from model.base.baseconfig import BaseRecConfig
from dataclasses import dataclass


@dataclass
class Config(BaseRecConfig):

    name: str = None
    dropout_ratio: float = None
    depth: int = None
    c_num: int = None
    backbone: str = None
    hid_dims: list = None
    output_func = None
    depth: int = None
    num_head: int = None
    exp_factor: float = None
    time_span: int = None
    max_seq_length: int = None
