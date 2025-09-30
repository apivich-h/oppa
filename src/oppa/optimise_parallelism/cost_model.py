from typing import Callable, List, Union, Dict
import yaml
import os
import random
import math

import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch

from torch.optim import SGD, Adam
from torch.quasirandom import SobolEngine

from ..strategies.parallel_strategy import ParallelisationStrategy, STRATEGY_EMBEDDING_DIM, STRATEGY_PARALLELISM_DIM_SIZE_INDEXS
from ..runners import run_training_subprocess
from ..optimise_parallelism.baseline import ParallelismOptimiser
from ..bo_gp.custom_components.parallelism_prior import OOM_ADD_FACTOR, ParallelCommCostMean, MaxMemoryMean

# to make the same costs be selected randomly
SMALL_PERTURB = 1e-6


class CostModelBased(ParallelismOptimiser):
    
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
        runner_additional_args_dir: str = './additional_args.yaml',
        training_num_steps: int = 100,
        training_process_args: Dict = None,
        fit_steps: int = 5000,
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
        
        self.fit_steps = fit_steps
        self.explore_mode_phase = True
        self.best_strat = None
        self.learned_surrogate = None
        
    def _fit_model(self, train_x, train_y_quantity, train_y, trial_mean_module=None):
        
        assert self.max_out_gpus  # TODO: adjust for when not
        
        if trial_mean_module is not None:
            mean_module = trial_mean_module
        elif train_y_quantity == 'time':
            # cost for the reciprocal of time GP
            mean_module = ParallelCommCostMean(
                x_upper_bound=self._x_upper_bound,
                consider_comm=True,
            )
        else:
            # cost for the max mem consumption GP
            assert train_y_quantity == 'mem'
            mean_module = MaxMemoryMean(
                x_upper_bound=self._x_upper_bound,
                max_mem_GB=self.gpu_max_mem_GB,
            )
        
        # TODO: probably make this part more adjustable from the outside
        optimizer = Adam([{'params': mean_module.parameters()}], lr=0.1)
        mean_module.train()
        for epoch in range(self.fit_steps + 1):
            optimizer.zero_grad()
            output = mean_module(train_x)
            loss = ((output.flatten() - train_y.flatten())**2).mean()
            loss.backward()
            if epoch % 1000 == 0:
                print(f"Step {epoch:>3}/{self.fit_steps} - Loss: {loss.item():>4.3f} ")
            optimizer.step()
            
        return mean_module, loss.item()
        
    def _select_sample(self, i) -> ParallelisationStrategy:
        
        candidates = torch.tensor(np.array([x.to_list() for x in self.candidates_list]))
        candidates = ParallelisationStrategy.log_transform(candidates) / self._x_upper_bound[..., :]
        print(f'Pre BO: {len(candidates)=}')
        
        if self.explore_mode_phase:
            s = torch.randint(low=0, high=self.candidates_count, size=(1,))[0]
            c = self.candidates_list[s]
            aux = dict()
            
        else:
            
            train_x = ParallelisationStrategy.log_transform(torch.tensor(np.array(self.queried_X)).double()) / self._x_upper_bound[...,:]
            train_y = torch.tensor(np.array(self.queried_y)).reshape(-1, 1).double()
            train_y_isnan = train_y.isnan().reshape(-1)
            train_x = train_x[~train_y_isnan]
            train_y = train_y[~train_y_isnan]
            print(f'Time GP sizes: {train_x.shape=}, {train_y.shape=}')
            throughput_fn, l = self._fit_model(
                train_x=train_x,
                train_y_quantity='time',
                train_y=train_y,
            )
            
            train_x_mem = ParallelisationStrategy.log_transform(torch.tensor(np.array(self.queried_X)).double()) / self._x_upper_bound[...,:]
            train_mem = torch.tensor(np.array(self.queried_mem)).reshape(-1, 1).double()
            train_mem_isinf = train_mem.isinf().reshape(-1)
            train_mem[train_mem_isinf] = self.gpu_max_mem_GB * (1. + OOM_ADD_FACTOR)
            train_mem = (train_mem - self.gpu_max_mem_GB) / self.gpu_max_mem_GB
            print(f'Mem GP sizes: {train_x_mem.shape=}, {train_mem.shape=}')
            mem_fn, l = self._fit_model(
                train_x=train_x_mem,
                train_y_quantity='mem',
                train_y=train_mem,
            )
            
            throughput_pred = throughput_fn(candidates).reshape(-1)
            throughput_pred += SMALL_PERTURB * torch.randn(size=throughput_pred.size())
            mem_pred = mem_fn(candidates).reshape(-1)
            oom = (mem_pred >= 0.).float()
            best_idx = torch.argmax(throughput_pred * oom)
            x_next = candidates[best_idx]
            print(f'{candidates.shape=} {throughput_pred.shape=} {mem_pred.shape=}')
            self.learned_surrogate = {
                'throughput': throughput_fn,
                'mem': mem_fn,
                'x_upper_bound': self._x_upper_bound,
                'gpu_max_GB': self.gpu_max_mem_GB,
            }
            
            print(f'From BO: {x_next=}')
            # selected_set = {x.to_string() for x in self.queried_strat}
            # closest_x_idx = torch.argmax(model_score.covar_module.forward(candidates, x_next.reshape(1, -1)).reshape(-1))
            # best = candidates[closest_x_idx] * x_upper_bound
            # print(f'From SCBO: {candidates[closest_x_idx]=}')
            
            best = x_next * self._x_upper_bound
            print(f'From BO: selected encoding strat {best=}')
            c_list = ParallelisationStrategy.undo_log_transform(best.cpu().detach()).tolist()

            print('Selected raw params =', c_list)
            c = ParallelisationStrategy.from_list(c_list)
            self.best_strat = c
                        
            aux = {
                'best_strat_throughput_pred': throughput_pred[best_idx].tolist(),
                'best_strat_mem_pred': mem_pred[best_idx].tolist(),
                'throughput_params': {k: v.tolist() for (k, v) in throughput_fn.named_parameters()},
                'mem_params': {k: v.tolist() for (k, v) in mem_fn.named_parameters()},
            }
        
        return c, aux
    
    def _should_stop(self, r, aux, min_rounds, max_rounds, max_time) -> bool:
        if not self.explore_mode_phase:
            print(f'Done exploring, found the best strategy already -- stopping...')
            return True
        elif sum(self.queried_total_time) > max_time:
            print(f'Runtime is {sum(self.queried_total_time)} seconds -- stop exploring...')
            self.explore_mode_phase = False
            return False
        else:
            return False
