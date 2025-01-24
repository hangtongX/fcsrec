import os.path
from pathlib import Path
import munch
import toml
import torch, gc
import warnings
from utils.logger_handler import get_logger
from utils.getters import get_trainer
from trainer.train import BaseTrainer

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)
if __name__ == "__main__":
    file_path = Path(os.path.abspath(__file__)).resolve()
    config = munch.munchify(toml.load(os.path.join(file_path.parents[-4], "run.toml")))
    logger = get_logger(
        file_path=os.path.join(
            file_path.parents[-4], "run_error_log-" + str(config.train.device) + ".log"
        )
    )
    config.train.log_on_file = True
    for dataname in config.projects.datasets:
        trainer = None
        for modelname in config.projects.models:
            try:
                try:
                    if config.projects.trainer == "":
                        trainer_name = None
                    else:
                        trainer_name = config.projects.trainer
                    back = get_trainer(model_name=modelname, trainer_name=trainer_name)
                    logger.info(back.info)
                    # if (
                    #     trainer is not None
                    #     and back.__class__.__name__ == trainer.__class__.__name__
                    # ):
                    #     trainer.reinit(modelname=modelname)
                    #     logger.info("not reinit trainer, just reinit the model")
                    # else:
                    #     trainer = back.trainer(modelname, dataname, config=config)
                    trainer = back.trainer(modelname, dataname, config=config)
                except Exception as e:
                    logger.info(e)
                    trainer = BaseTrainer(modelname, dataname, config=config)

                # begin train
                result = trainer.train_test_for_times(config.run.times)
                trainer.save_results(result, hyper_tune=False)
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
            except RuntimeError as e:
                logger.exception(e)
