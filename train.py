import contextlib
import datetime
import logging
import os, gc
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
from colorama import Fore, init as colorama_init
import munch
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils.logger_handler import get_logger
from eval.evaluator import Evaluator
from data_utils.dataset import BaseDataset
from model.base.basemodel import BaseModel
from trainer.trainConfig import BaseTrainerConfig
from trainer.trainer_utils import (
    TrainingCallback,
    set_seed,
    CallbackHandler,
    MetricConsolePrinterCallback,
    ProgressBarCallback,
    WandbCallback,
)
from utils import getters
import yaml
from collections import Counter
from potime import RunTime

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)
colorama_init(autoreset=True)


class BaseTrainer:
    """Base class to perform model training.

    Args:
        modelname (str): name of the model
        dataname (str): name of the dataset
        config (dict): The training arguments summarizing the main
            params used for training. If None, a basic training instance of
            :class:`BaseTrainerConfig` is used, and will update the setting from run.toml. Default: None.

        callbacks (List[~pythae.trainers.training_callbacks.TrainingCallbacks]):
            A list of callbacks to use during training.
    """

    def __init__(
        self,
        modelname: str,
        dataname: str,
        config: munch.Munch = None,
    ):
        self.training_config = BaseTrainerConfig()
        if config is None:
            # load from run.toml
            config = self.training_config.update()
        else:
            self.training_config.update(params=config)

        if self.training_config.output_dir is None:
            dir_path = Path(os.path.abspath(__file__)).resolve().parents[-4]
            output_dir = os.path.join(dir_path, "model_results/")
            self.base_dir = dir_path
            output_dir = os.path.join(
                output_dir, self.training_config.project_name, dataname, modelname
            )
            self.training_config.output_dir = output_dir

        # for distributed training
        self.world_size = self.training_config.world_size
        self.local_rank = self.training_config.local_rank
        self.rank = self.training_config.rank
        self.dist_backend = self.training_config.dist_backend

        if self.world_size > 1:
            self.distributed = True
        else:
            self.distributed = False

        if self.distributed:
            device = self._setup_devices()

        else:
            device = (
                str("cuda:" + str(self.training_config.device))
                if torch.cuda.is_available() and not self.training_config.no_cuda
                else "cpu"
            )

        self.amp_context = (
            torch.autocast("cuda")
            if self.training_config.amp
            else contextlib.nullcontext()
        )

        # if (
        #     hasattr(model.model_config, "reconstruction_loss")
        #     and model.model_config.reconstruction_loss == "bce"
        # ):
        #     self.amp_context = contextlib.nullcontext()

        self.device = device

        self.model_name = modelname
        self.model = getters.get_model(modelname)()

        # load dataset and data info
        if self.model.model_config.train_type is not None:
            config.dataset.traintype = self.model.model_config.train_type
        self.dataname = dataname
        self.train_dataset, self.eval_dataset, self.test_dataset, self.dataInfo = (
            getters.get_dataset(dataname, config)
        )

        # Build models
        self._setup_model()

        # build the eval
        self.evaluate = Evaluator(config=config)

        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])

        # Define the loaders
        if isinstance(self.train_dataset, DataLoader):
            train_loader = self.train_dataset
            logger.warning(
                Fore.YELLOW
                + "Using the provided train dataloader! Carefull this may overwrite some "
                "params provided in your training config."
            )
        else:
            train_loader = self.get_train_dataloader(self.train_dataset)

        if self.eval_dataset is not None:
            if isinstance(self.eval_dataset, DataLoader):
                eval_loader = self.eval_dataset
                logger.warning(
                    Fore.YELLOW
                    + "Using the provided eval dataloader! Carefull this may overwrite some "
                    "params provided in your training config."
                )
            else:
                eval_loader = self.get_eval_dataloader(self.eval_dataset)
        else:
            logger.info(
                Fore.RED
                + "! No eval dataset provided ! -> keeping best model on train.\n"
            )
            self.training_config.keep_best_on_train = True
            eval_loader = None

        if self.test_dataset is not None:
            if isinstance(self.test_dataset, DataLoader):
                test_loader = self.test_dataset
                logger.warning(
                    Fore.YELLOW
                    + "Using the provided test dataloader! Carefull this may overwrite some "
                    "params provided in your training config."
                )
            else:
                test_loader = self.get_test_dataloader(self.test_dataset)
        else:
            logger.info(Fore.RED + "! No test dataset provided !\n")
            self.training_config.keep_best_on_train = True
            test_loader = None

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.callbacks = None

        # run sanity check on the model
        self._best_model = None
        self.cur_seed = None
        seed = self.prepare_training()
        self._run_model_sanity_check()

        if self.is_main_process:
            logger.info(
                Fore.GREEN + "Model passed sanity check !\n"
                "Ready for training.\n"
                f"Training loops will run {self.model_name} on dataset {self.dataname} "
                f"with random seeds {self.training_config.seed}\n"
            )

            logger.info(
                Fore.BLUE
                + "The dataset Info is: \n"
                + yaml.dump(self.dataInfo, sort_keys=True, default_flow_style=False)
            )

    @property
    def is_main_process(self):
        if self.rank == 0 or self.rank == -1:
            return True
        else:
            return False

    def _setup_devices(self):
        """Sets up the devices to perform distributed training."""

        if dist.is_available() and dist.is_initialized() and self.local_rank == -1:
            logger.warning(
                "torch.distributed process group is initialized, but local_rank == -1. "
            )
        if self.training_config.no_cuda:
            self._n_gpus = 0
            device = "cpu"

        else:
            torch.cuda.set_device(self.local_rank)
            device = torch.device("cuda", self.local_rank)

            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.dist_backend,
                    init_method="env://",
                    world_size=self.world_size,
                    rank=self.rank,
                )

        return device

    def _setup_model(self):
        self.model.model_config.update(self.dataInfo)
        self.model.device = self.device
        self.model.build()
        self.model.to(self.device)

    def get_train_dataloader(
        self, train_dataset: BaseDataset
    ) -> torch.utils.data.DataLoader:
        if self.distributed:
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=self.world_size, rank=self.rank
            )
        else:
            train_sampler = None
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.training_config.per_device_train_batch_size,
            num_workers=self.training_config.train_dataloader_num_workers,
            shuffle=bool(train_sampler is None),
            sampler=train_sampler,
            # collate_fn=collate_dataset_output,
        )

    def get_eval_dataloader(
        self, eval_dataset: BaseDataset
    ) -> torch.utils.data.DataLoader:
        if self.distributed:
            eval_sampler = DistributedSampler(
                eval_dataset, num_replicas=self.world_size, rank=self.rank
            )
        else:
            eval_sampler = None
        return DataLoader(
            dataset=eval_dataset,
            batch_size=(
                self.training_config.per_device_eval_batch_size
                if self.evaluate.type == "fullsort"
                else 100
            ),
            num_workers=self.training_config.eval_dataloader_num_workers,
            shuffle=True if self.evaluate.type == "fullsort" else False,
            sampler=eval_sampler,
            # collate_fn=collate_dataset_output,
        )

    def get_test_dataloader(
        self, test_dataset: BaseDataset
    ) -> torch.utils.data.DataLoader:
        if self.distributed:
            eval_sampler = DistributedSampler(
                test_dataset, num_replicas=self.world_size, rank=self.rank
            )
        else:
            eval_sampler = None
        return DataLoader(
            dataset=test_dataset,
            batch_size=(
                self.training_config.per_device_eval_batch_size
                if self.evaluate.type == "fullsort"
                else 100
            ),
            num_workers=self.training_config.eval_dataloader_num_workers,
            shuffle=(
                self.training_config.test_shuffle
                if self.evaluate.type == "fullsort"
                else False
            ),
            sampler=eval_sampler,
            # collate_fn=collate_dataset_output,
        )

    def set_optimizer(self):
        optimizer_cls = getattr(optim, self.training_config.optimizer_cls)

        if self.training_config.optimizer_params is not None:
            optimizer = optimizer_cls(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                **self.training_config.optimizer_params,
            )
        else:
            optimizer = optimizer_cls(
                self.model.parameters(), lr=self.training_config.learning_rate
            )

        self.optimizer = optimizer

    def set_scheduler(self):
        if self.training_config.scheduler_cls is not None:
            scheduler_cls = getattr(lr_scheduler, self.training_config.scheduler_cls)

            if self.training_config.scheduler_params is not None:
                scheduler = scheduler_cls(
                    self.optimizer, **self.training_config.scheduler_params
                )
            else:
                scheduler = scheduler_cls(self.optimizer)

        else:
            scheduler = None

        self.scheduler = scheduler

    def _set_output_dir(self):
        # Create folder
        if not os.path.exists(self.training_config.output_dir) and self.is_main_process:
            os.makedirs(self.training_config.output_dir, exist_ok=True)
            logger.info(
                Fore.YELLOW
                + f"Created {self.training_config.output_dir} folder since did not exist.\n"
            )

        self._training_signature = (
            str(datetime.datetime.now())[5:19].replace(" ", "_").replace(":", "-")
        )
        if self.training_config.log_on_file:
            training_dir = os.path.join(
                self.training_config.output_dir,
                f"{self._training_signature}",
            )
        else:
            training_dir = os.path.join(
                self.training_config.output_dir,
                "test_without_log_file",
                f"{self._training_signature}",
            )

        self.training_dir = training_dir

        self.best_model_dir = os.path.join(self.training_config.output_dir, "best")

        if (
            not os.path.exists(training_dir)
            and self.is_main_process
            and self.training_config.log_on_file
        ):
            os.makedirs(training_dir, exist_ok=True)
            logger.info(
                f"Created {training_dir}. \n"
                "Training config, checkpoints and final model will be saved here.\n"
            )

    def _get_file_logger(self, log_output_dir):
        log_dir = log_output_dir
        if self.training_config.log_on_file:
            # if dir does not exist create it
            if not os.path.exists(log_dir) and self.is_main_process:
                os.makedirs(log_dir, exist_ok=True)
                logger.info(f"Created {log_dir} folder since did not exists.")
                logger.info("Training logs will be recodered here.\n")
                logger.info(" -> Training can be monitored here.\n")

            # create and set logger
            log_file_path = os.path.join(log_dir, f"{self._training_signature}.log")
            file_logger = get_logger(file_path=log_file_path, mode="a", console=False)
        else:
            logger.info(
                f"Not using file logger to save training logs by setting log_on_file = {self.training_config.log_on_file}.\n"
            )
            file_logger = logger
        return file_logger

    def _setup_callbacks(self):
        if self.training_config.callback:
            if self.callbacks is None:
                self.callbacks = [WandbCallback(model_name=self.model_name)]
        else:
            self.callbacks = [TrainingCallback()]

        self.callback_handler = CallbackHandler(
            callbacks=self.callbacks, model=self.model
        )

        self.callback_handler.add_callback(ProgressBarCallback())
        self.callback_handler.add_callback(MetricConsolePrinterCallback())

    def set_model_hyper_params(self, params):
        self.model.model_config.update(params)
        self._setup_model()

    def _run_model_sanity_check(self):
        try:
            train_inputs = next(iter(self.train_loader))
            train_dataset = self._set_inputs_to_device(train_inputs)
            self.model(train_dataset)
            # self.eval_step(0)
            # self.predict(self.model)

        except Exception as e:
            raise NotImplementedError(
                "Error when calling forward method from model. Potential issues: \n"
                f"Exception raised: {type(e)} with message: " + str(e)
            ) from e

    def _set_optimizer_on_device(self, optim, device):
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

        return optim

    def _set_inputs_to_device(self, inputs: Dict[str, Any]):
        inputs_on_device = inputs

        if self.device[0:4] == "cuda":
            cuda_inputs = dict.fromkeys(inputs)

            for key in inputs.keys():
                if torch.is_tensor(inputs[key]):
                    cuda_inputs[key] = inputs[key].cuda(self.device)

                else:
                    cuda_inputs[key] = inputs[key]
            inputs_on_device = cuda_inputs

        return inputs_on_device

    def _optimizers_step(self, model_output=None):
        loss = model_output.loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _schedulers_step(self, metrics=None):
        if self.scheduler is None:
            pass

        elif isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metrics)

        else:
            self.scheduler.step()

    def prepare_training(self, seeds_id=0):
        """Sets up the trainer for training"""
        # set random seed
        seed = self.training_config.seed[seeds_id % len(self.training_config.seed)]
        set_seed(seed)

        # set optimizer
        self.set_optimizer()

        # set scheduler
        self.set_scheduler()

        # create folder for saving
        if seeds_id == 0:
            self._set_output_dir()

        # set callbacks
        self._setup_callbacks()
        self.cur_seed = seed

        return seed

    def train(self, log_verbose=True, file_logger=None, times: int = 1):
        """This function is the main training function

        Args:
            :param log_verbose:
            :param file_logger:
            :param times:
        """

        # set the best losses for early stopping
        # best_train_loss = self.training_config.best_train_loss
        best_eval_loss: float = self.training_config.best_eval_loss
        best_model = None
        rates: pd.DataFrame = None

        for epoch in range(1, self.training_config.num_epochs + 1):
            self.callback_handler.on_epoch_begin(
                training_config=self.training_config,
                epoch=epoch,
                train_loader=self.train_loader,
                eval_loader=self.eval_loader,
                times=times,
            )

            metrics = {}

            epoch_train_loss = self.train_step(epoch)
            metrics[f"train_{times}"] = epoch_train_loss

            if self.eval_dataset is not None:
                epoch_eval_loss = self.eval_step(epoch)
                metrics[f"eval_{times}"] = epoch_eval_loss
            else:
                epoch_eval_loss = self.evaluate.val_best
            is_break, is_update, best_eval_loss = self.evaluate.early_stop_controller(
                epoch_eval_loss
            )
            metrics[f"eval_best-{times}"] = best_eval_loss
            self._schedulers_step(best_eval_loss)
            if is_update and not self.training_config.keep_best_on_train:
                best_model = deepcopy(self.model.state_dict())
                self._best_model = best_model
            if (
                is_break or epoch >= self.training_config.num_epochs
            ) and self.is_main_process:
                if best_model is None:
                    pre_model = self.model
                else:
                    self.model.load_state_dict(best_model)
                    pre_model = self.model
                rates = self.predict(pre_model, epoch)["rates"]

                self.callback_handler.on_prediction_step(
                    self.training_config,
                    type="table",
                    table=rates,
                    name=f"test_result_{times}",
                )

            self.callback_handler.on_epoch_end(training_config=self.training_config)

            # save checkpoints
            if is_break or epoch % self.training_config.steps_saving == 0:
                if self.is_main_process:
                    self.save_checkpoint(
                        state_dict=best_model, dir_path=self.training_dir, epoch=epoch
                    )
                    # logger.info(f"Saved checkpoint at epoch {epoch}\n")

                    if log_verbose:
                        file_logger.info(f"Saved checkpoint at epoch {epoch}\n")

            self.callback_handler.on_log(
                self.training_config,
                metrics,
                logger=file_logger,
                step=epoch,
                rank=self.rank,
            )
            if is_break:
                file_logger.info(
                    Fore.GREEN
                    + f"up to the {self.evaluate.patience_max} not better, so the train will end at Epoch {epoch}"
                )
                break

        if self.distributed:
            dist.destroy_process_group()

        # logger current train loop results
        # if self.is_main_process:
        #     file_logger.info(
        #         Fore.GREEN
        #         + f"Test Completed, the current traing loop test result is \n {rates}"
        #     )

        return rates, best_eval_loss, best_model

    def eval_step(self, epoch: int):
        """Perform an evaluation step

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The evaluation loss
        """

        self.callback_handler.on_eval_step_begin(
            training_config=self.training_config,
            eval_loader=self.eval_loader,
            epoch=epoch,
            rank=self.rank,
        )

        self.model.eval()

        epoch_loss = 0

        with self.amp_context:
            for inputs in self.eval_loader:
                inputs = self._set_inputs_to_device(inputs)
                model_output = None
                ids = None
                mask = None
                ground_truth = None

                try:
                    with torch.no_grad():
                        if self.evaluate.type == "fullsort":
                            model_output = self.model.full_sort(
                                inputs,
                                epoch=epoch,
                                dataset_size=len(self.eval_loader.dataset),
                                uses_ddp=self.distributed,
                            )
                            ids = inputs["id"]
                            mask = self.eval_dataset.mask
                            ground_truth = self.eval_dataset.ground_truth
                        elif self.evaluate.type == "samplesort":
                            model_output = self.model.predict(
                                inputs,
                                epoch=epoch,
                                dataset_size=len(self.eval_loader.dataset),
                                uses_ddp=self.distributed,
                            )
                        else:
                            raise NotImplementedError(
                                Fore.RED + "Unsupported sort type "
                            )

                except RuntimeError as error:
                    print(Fore.RED + f"Something wrong happened while eval step")
                    print(error)
                    print("\n")
                loss = self.evaluate.eval_controller(
                    model_output.score, id=ids, mask=mask, ground_truth=ground_truth
                )

                epoch_loss += loss.item()

                if epoch_loss != epoch_loss:
                    raise ArithmeticError("NaN detected in eval loss")

                self.callback_handler.on_eval_step_end(
                    training_config=self.training_config
                )

        epoch_loss /= len(self.eval_loader)

        return epoch_loss

    def train_step(self, epoch: int, **kwargs):
        """The trainer performs training loop over the train_loader.

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The step training loss
        """
        # set model in train model
        self.model.train()

        epoch_loss = 0
        iter_loss = 0

        self.callback_handler.on_train_step_begin(
            training_config=self.training_config,
            train_loader=self.train_loader,
            epoch=epoch,
            rank=self.rank,
            loss=iter_loss,
        )

        for inputs in self.train_loader:
            inputs = self._set_inputs_to_device(inputs)

            with self.amp_context:
                model_output = self.model(
                    inputs,
                    epoch=epoch,
                    dataset_size=len(self.train_loader.dataset),
                    uses_ddp=self.distributed,
                )

            self._optimizers_step(model_output)

            loss = model_output.loss

            epoch_loss += loss.item()
            iter_loss = loss.item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError(
                    "NaN detected in train loss",
                    model_output,
                )

            self.callback_handler.on_train_step_end(
                training_config=self.training_config, loss=iter_loss
            )

        # Allows model updates if needed
        if self.distributed:
            self.model.module.update()
        else:
            self.model.update(epoch=epoch)

        epoch_loss /= len(self.train_loader)

        return epoch_loss

    def save_model(self, state_dict, dir_path: str):
        """This method saves the final model along with the config files

        Args:
            model (BaseAE): The model to be saved
            dir_path (str): The folder where the model and config files should be saved
        """

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # save model
        if self.distributed:
            torch.save(state_dict, os.path.join(dir_path, "model.pt"))

        else:
            torch.save(state_dict, os.path.join(dir_path, "model.pt"))

        # save training config
        self.training_config.save_json(dir_path, "training_config")

        self.callback_handler.on_save(self.training_config)

    def save_checkpoint(self, state_dict, dir_path, epoch: int):
        """Saves a checkpoint alowing to restart training from here

        Args:
            dir_path (str): The folder where the checkpoint should be saved
            epochs_signature (int): The epoch number"""

        checkpoint_dir = os.path.join(dir_path, f"checkpoint_epoch")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # save optimizer
        torch.save(
            deepcopy(self.optimizer.state_dict()),
            os.path.join(checkpoint_dir, "optimizer.pt"),
        )

        # save scheduler
        if self.scheduler is not None:
            torch.save(
                deepcopy(self.scheduler.state_dict()),
                os.path.join(checkpoint_dir, "scheduler.pt"),
            )

        # save model
        if self.distributed:
            torch.save(state_dict, os.path.join(checkpoint_dir, "model.pt"))

        else:
            torch.save(state_dict, os.path.join(checkpoint_dir, "model.pt"))

        # save training config
        self.training_config.save_json(checkpoint_dir, "training_config")

    def load_checkpoint_model(
        self, dir_path: str = None, epoch: int = None, full_path=None
    ):
        if full_path is not None:
            state_dict = torch.load(full_path)
        else:
            checkpoint_dir = os.path.join(dir_path, f"checkpoint_epoch/")
            state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(
        self,
        model: BaseModel = None,
        epoch: int = None,
        time_needed=False,
        out_topk=False,
        **kwargs,
    ):
        if model is not None:
            self.model = model
        if epoch is None:
            epoch = 0
        self.model.eval()
        rates = None

        self.callback_handler.on_test_step_begin(
            training_config=self.training_config,
            test_loader=self.test_loader,
            epoch=epoch,
            rank=self.rank,
        )
        test_data_loader = (
            self.test_loader if "test_loader" not in kwargs else kwargs["test_loader"]
        )
        times_cost = 0
        rec_items = None
        hit_items = None
        with self.amp_context:
            for inputs in test_data_loader:
                inputs = self._set_inputs_to_device(inputs)
                model_output = None
                ids = None
                mask = None
                ground_truth = None

                try:
                    with torch.no_grad():
                        if self.evaluate.type == "fullsort":
                            if time_needed:
                                start_T = time.perf_counter()
                            model_output = self.model.full_sort(
                                inputs,
                                # dataset_size=len(self.test_loader.dataset),
                                uses_ddp=self.distributed,
                            )
                            if time_needed:
                                end_T = time.perf_counter()
                                times_cost += end_T - start_T
                            ids = inputs["id"]
                            mask = self.test_dataset.mask
                            ground_truth = self.test_dataset.ground_truth
                            item_popularity = self.test_dataset.item_popularity
                        elif self.evaluate.type == "samplesort":
                            model_output = self.model.predict(
                                inputs,
                                # dataset_size=len(self.test_loader.dataset),
                                # uses_ddp=self.distributed,
                            )
                            item_popularity = inputs["item_pop"]
                        else:
                            raise NotImplementedError(
                                Fore.RED + "Unsupported sort type "
                            )

                except RuntimeError as error:
                    print(Fore.RED + f"Something wrong happened while test step")
                    print(error)
                    print("\n")
                back = self.evaluate.test_controller(
                    model_output.score,
                    id=ids,
                    mask=mask,
                    ground_truth=ground_truth,
                    item_popularity=item_popularity,
                    item_prices=(
                        torch.Tensor(kwargs["item_prices"]).to(self.device)
                        if "item_prices" in kwargs
                        else None
                    ),
                    out_topk=out_topk,
                )
                if out_topk:
                    print(back["rec_items"].shape)
                    rec_items = (
                        back["rec_items"]
                        if rec_items is None
                        else np.concatenate((rec_items, back["rec_items"]), axis=0)
                    )
                    hit_items = (
                        back["hit_items"]
                        if hit_items is None
                        else np.concatenate((hit_items, back["hit_items"]), axis=0)
                    )
                rates_back = np.array(back["result"]).reshape(
                    len(self.evaluate.testmetric), len(self.evaluate.testmetrick), -1
                )
                if rates is None:
                    rates = rates_back
                else:
                    rates = np.concatenate((rates, rates_back), axis=2)

                self.callback_handler.on_test_step_end(
                    training_config=self.training_config
                )
            rates = rates.mean(-1)
            rates = pd.DataFrame(
                rates, columns=self.evaluate.testmetrick, index=self.evaluate.testmetric
            )
            rates = rates.apply(lambda x: round(x, 5))

        return {
            "rates": rates.transpose(),
            "time_cost": times_cost,
            "rec_items": rec_items,
            "hit_items": hit_items,
        }

    def reinit(self, modelname=None, times=None, filelogger=None, reload_model=False):

        self.model_name = modelname
        # if reload_model:
        #     self.model = getters.get_model(modelname)()
        #     self._setup_model()
        self.model.build()
        self.model.to(self.device)
        self._best_model = None
        seed = self.prepare_training(times)
        self.evaluate.reset_patience()

        if self.is_main_process:
            filelogger.info(
                Fore.GREEN + "Model passed sanity check !\n"
                "Ready for training.\n"
                f"Current loops will run {self.model_name} on dataset {self.dataname} with random seed {seed}\n"
            )
        return None

    def extra_output(self):
        out_dir = os.path.join(self.training_config.output_dir, "extra_results")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        return out_dir

    def train_test_for_times(self, times: int = 1, log_output_dir: str = None):
        if times < 1:
            raise ValueError(
                Fore.RED
                + "the values of times must be greater than or equal to 1....\n"
            )

        # set up log file
        if self.is_main_process:
            log_verbose = True
            if log_output_dir is None:
                dir_path = Path(os.path.abspath(__file__)).resolve().parents[-4]
                output_dir = os.path.join(dir_path, "model_results/")
                log_output_dir = os.path.join(self.training_config.output_dir, "logs")
            file_logger = self._get_file_logger(log_output_dir=log_output_dir)
        else:
            file_logger = logger

        rates = None
        rates_array = []
        rate_upper = None
        rate_lower = None
        best_eval = None
        best_model = None
        best_loop = None

        log_verbose = False

        msg = (
            f"Training params:\n - max_epochs: {self.training_config.num_epochs}\n"
            " - per_device_train_batch_size: "
            f"{self.training_config.per_device_train_batch_size}\n"
            " - per_device_eval_batch_size: "
            f"{self.training_config.per_device_eval_batch_size}\n"
            f" - checkpoint saving every: {self.training_config.steps_saving}\n"
            f"Optimizer: {self.optimizer}\n"
            f"Scheduler: {self.scheduler}\n"
        )

        if self.is_main_process:
            log_verbose = True
            file_logger.info(msg)
            file_logger.info(self.model)
            file_logger.info(self.model.model_config.__dict__)
            file_logger.info(
                Fore.YELLOW + "All get ready, Successfully launched training !\n"
            )

        self.callback_handler.on_train_begin(
            training_config=self.training_config, model_config=self.model.model_config
        )

        bar = tqdm(
            range(times),
            unit="times",
            position=0,
            leave=True,
            ncols=100,
            colour="CYAN",
            postfix=f"run on {self.device}",
        )
        for i in bar:
            bar.set_description(
                desc=Fore.LIGHTRED_EX
                + f"Traing loops {i+1}/{times} of {self.model_name.upper()} on {self.dataname.upper()}"
            )
            file_logger.info("-" * 80)
            file_logger.info(
                Fore.LIGHTGREEN_EX
                + f"The new training loop will begin, current is the {i+1} th loops of total {times}"
            )
            file_logger.info("-" * 80)
            rate, eval_score, model_params = self.train(
                file_logger=file_logger, times=i + 1
            )
            if rates is None:
                rates = rate
                rates_array.append(rate)
                rate_lower = rate
                rate_upper = rate
                best_eval = eval_score
                best_model = model_params
                best_loop = i + 1
            else:
                rates = rates + rate
                rates_array.append(rate)
                rate_lower = np.minimum(rate_lower, rate)
                rate_upper = np.maximum(rate_upper, rate)
                if self.evaluate.sortType == "asc":
                    if best_eval < eval_score:
                        best_eval = eval_score
                        best_model = model_params
                        best_loop = i + 1
                elif self.evaluate.sortType == "desc":
                    if best_eval > eval_score:
                        best_eval = eval_score
                        best_model = model_params
                        best_loop = i + 1
                else:
                    raise ValueError(
                        Fore.RED + f"The sort type {best_eval} is not supported...\n"
                    )

            if i < times - 1:
                file_logger.info(f"The {i}-th training result is:\n{rate}")
                self.reinit(self.model_name, times=i + 1, filelogger=file_logger)

        # mean for finall result
        rates = rates / times
        final_dir = os.path.join(self.best_model_dir)

        if self.is_main_process:
            self.save_model(state_dict=best_model, dir_path=final_dir)
            self.model.load_state_dict(best_model)
            # self.extra_output()

            self.callback_handler.on_prediction_step(
                self.training_config,
                type="table",
                table=rates,
                name=f"final_result_mean",
            )
            self.callback_handler.on_prediction_step(
                self.training_config,
                type="table",
                table=rate_upper,
                name=f"final_result_upper",
            )
            self.callback_handler.on_prediction_step(
                self.training_config,
                type="table",
                table=rate_lower,
                name=f"final_result_lower",
            )
            file_logger.info("*" * 80)
            file_logger.info(
                f"\n After {times} loops traing, the total result are as follows:\n"
            )
            for t, record in enumerate(rates_array):
                file_logger.info(f"The {t+1}-th training result is:\n{record}")
                file_logger.info("*" * 80)
            file_logger.info(f"The mean results is:\n{rates}")
            file_logger.info("*" * 80)
            file_logger.info(f"The upper results is:\n{rate_upper}")
            file_logger.info("*" * 80)
            file_logger.info(f"The lower results is:\n{rate_lower}")
            file_logger.info("*" * 80)

        self.callback_handler.on_train_end(training_config=self.training_config)
        torch.cuda.empty_cache()
        logger.info(Fore.LIGHTGREEN_EX + "Train end" + "." * 20)
        print("\n" * 3)

        logging.shutdown()
        del file_logger

        return munch.Munch(
            {"rates": rates, "rate_upper": rate_upper, "rate_lower": rate_lower}
        )

    def save_results(self, data, hyper_tune=False, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(self.base_dir, "model_results/")
            output_dir = os.path.join(
                output_dir, self.training_config.project_name, self.dataname
            )
        if not isinstance(data, dict):
            data = {"data": data}
        for key, value in data.items():
            value = value.copy()
            value.reset_index(inplace=True)
            value.rename(columns={"index": "Top-K"}, inplace=True)
            extra_remark = self.model.extra_remarks()
            value["model_name"] = self.model_name
            value["sort_key"] = key
            specific_sheet_name = extra_remark["sheet_name"]
            if hyper_tune:
                value["remark"] = extra_remark["remark"]
                if "sheet_name" in extra_remark:
                    specific_sheet_name = "hyper_tune" + extra_remark["sheet_name"]
            self.evaluate.save_result_to_excel(
                value.copy(),
                path=output_dir,
                hyper_tune=hyper_tune,
                posix=str(key),
                specific_sheet_name=specific_sheet_name,
            )

    def get_model_size(self):
        param_size = 0
        for name, param in self.model.named_parameters():
            param_size += param.numel() * param.element_size()
        size_in_mb = param_size / (1024 * 1024)
        print(f"Model {self.model.__class__.__name__} size is {size_in_mb}MB")
        return size_in_mb
