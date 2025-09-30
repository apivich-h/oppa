from typing import Callable, List, Union, Dict
import yaml
import os

import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import random

from ..strategies.parallel_strategy import ParallelisationStrategy
from ..optimise_parallelism.baseline import ParallelismOptimiser


class XGBoost(ParallelismOptimiser):
    
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
        runner_additional_args_dir: str = None, 
        training_num_steps: int = 100,
        training_process_args: Dict = None,
        init_random_rounds: int = 1,
        drop_oom_timings: bool = True,
        xgb_loss_type: str = 'rank',
        exploration: float = 0.2,
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
            tp_divisible_by=tp_divisible_by,
            pp_divisible_by=pp_divisible_by,
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
        
        assert init_random_rounds >= 1
        self.init_random_rounds = init_random_rounds
        self.drop_oom_timings = drop_oom_timings

        self.init_samples = None
        self.learned_surrogate = None
        self.bst = None
        self.y_max = None

        # default to rank to match original codes at
        # https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/autotuning/tuner/model_based_tuner.py#L19
        if xgb_loss_type == "reg":
            self.xgb_params = {
                "max_depth": 3,
                "gamma": 0.0001,
                "min_child_weight": 1,
                "subsample": 1.0,
                "eta": 0.3,
                "lambda": 1.0,
                "alpha": 0,
                "objective": "reg:linear",
            }
        elif xgb_loss_type == "rank":
            self.xgb_params = {
                "max_depth": 3,
                "gamma": 0.0001,
                "min_child_weight": 1,
                "subsample": 1.0,
                "eta": 0.3,
                "lambda": 1.0,
                "alpha": 0,
                "objective": "rank:pairwise",
            }

        self._x_upper_bound = torch.tensor(ParallelisationStrategy.get_list_log_transform_upper_bound(
            num_hosts=self.num_hosts,
            num_gpus_per_host=self.num_gpus_per_host,
            batch_size=self.batch_size,
            max_dp_bucket_size_mb=2**12,
            max_model_chunks=self.max_num_microbatches,
        ))
        self.random_exploration_ratio = exploration  # do random exploration

    def xgb_fit(self, xs, ys):
        x_train = np.array(xs, dtype=np.float32)
        y_train = np.array(ys, dtype=np.float32)
        y_max = np.max(y_train)
        self.y_max = max(y_max, 1e-9)
        y_train = y_train / self.y_max

        dtrain = xgb.DMatrix(x_train, y_train)
        self.bst = xgb.train(self.xgb_params, dtrain)
        
        # def _fn(x):
        #     return self.y_max * self.bst.predict(xgb.DMatrix(x))
        
        self.learned_surrogate = {
            'throughput': None,
            'mem': None,
            'x_train': x_train,
            'y_train': y_train,
            'y_max': y_max,
            'xgb_params': self.xgb_params,
            'x_upper_bound': self._x_upper_bound,
            'gpu_max_GB': self.gpu_max_mem_GB,
        }

    def xgb_predict(self, xs):
        features = xgb.DMatrix(xs)
        return self.bst.predict(features)
        
    def _select_sample(self, i) -> ParallelisationStrategy:

        x_upper_bound = self._x_upper_bound
        candidates_set = list(set(self.candidates_list) - set(self.queried_strat))
        assert len(candidates_set) == len(self.candidates_list) - i, (len(candidates_set), len(self.candidates_list), i)
        candidates = torch.tensor(np.array([x.to_list() for x in candidates_set]))
        candidates = ParallelisationStrategy.log_transform(candidates) / x_upper_bound[..., :]

        if (i < self.init_random_rounds) or (np.random.rand() < self.random_exploration_ratio):
            c = random.choice(list(candidates_set))
            aux = {'rand_explore': True}
        else:
            train_x = ParallelisationStrategy.log_transform(torch.tensor(np.array(self.queried_X)).double()) / x_upper_bound[...,:]
            train_y = torch.tensor(np.array(self.queried_y)).reshape(-1, 1).double()
            train_y_isnan = train_y.isnan().reshape(-1)
            if train_y_isnan.all():
                c = random.choice(list(candidates_set))
                aux = {'rand_explore': True}
            else:
                # train_x = train_x[~train_y_isnan]
                # train_y = train_y[~train_y_isnan]
                train_y[train_y_isnan] = 0.
                self.xgb_fit(train_x, train_y)
                pred_y = self.xgb_predict(candidates)
                best_idx = np.argmax(pred_y)
                c = candidates_set[best_idx]
                aux = {'y_pred': pred_y[best_idx], 'rand_explore': False}
            
        return c, aux
