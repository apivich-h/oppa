from typing import Callable, List, Union, Dict
import yaml
import os

import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch

from ..strategies.parallel_strategy import ParallelisationStrategy
from ..optimise_parallelism.baseline import ParallelismOptimiser


class RandomSelection(ParallelismOptimiser):
    
    def __init__(
        self,
        batch_size: int = 1,
        num_hosts: int = 1,
        num_gpus_per_host: int = 1,
        max_dp_size: Union[int, None] = None,
        max_tp_size: Union[int, None] = None, 
        max_pp_size: Union[int, None] = None, 
        max_sp_size: Union[int, None] = 1, 
        max_num_microbatches: Union[int, None] = None,
        tp_divisible_by: int = 4,
        pp_divisible_by: int = 4,
        max_num_model_chunks: int = 4,
        gpu_max_mem_GB: float = 40.,
        max_out_gpus: bool = True,
        max_out_machines: bool = False,
        fixed_args: Dict = None,
        out_folder: str = './results',
        training_num_steps: int = 100,
        training_process_args: Dict = None,
        runner_additional_args_dir: str = None,
    ):
        super().__init__(
            batch_size=batch_size,
            num_hosts=num_hosts,
            num_gpus_per_host=num_gpus_per_host,
            max_dp_size=max_dp_size,
            max_tp_size=max_tp_size,
            max_pp_size=max_pp_size,
            max_sp_size=max_sp_size,
            max_num_microbatches=max_num_microbatches,
            pp_divisible_by=pp_divisible_by,
            tp_divisible_by=tp_divisible_by,
            max_num_model_chunks=max_num_model_chunks,
            gpu_max_mem_GB=gpu_max_mem_GB,
            max_out_gpus=max_out_gpus,
            max_out_machines=max_out_machines,
            fixed_args=fixed_args,
            use_throughput=True,
            out_folder=out_folder,
            runner_additional_args_dir=runner_additional_args_dir,
            training_num_steps=training_num_steps,
            training_process_args=training_process_args,
        )
        
    def _select_sample(self, i) -> ParallelisationStrategy:
        s = torch.randint(low=0, high=self.candidates_count, size=(1,))[0]
        return self.candidates_list[s], dict()
