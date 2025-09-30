from typing import Callable, List, Union, Dict
import yaml
import os

import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch

from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from botorch.models.transforms import Normalize, Standardize, Log
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound, LogExpectedImprovement, LogNoisyExpectedImprovement
from botorch.optim import optimize_acqf_discrete

from ..strategies.parallel_strategy import ParallelisationStrategy
from ..runners import run_training_subprocess
from ..optimise_parallelism.baseline import ParallelismOptimiser

botorch.settings.debug(True)


class KMeans(ParallelismOptimiser):
    
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
        # grid_dist_scale: float = 0.05,
        max_candidates_num: int = 50,
        use_throughput: bool = True,
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
            use_throughput=use_throughput,
            out_folder=out_folder,
            runner_additional_args_dir=runner_additional_args_dir,
            training_num_steps=training_num_steps,
            training_process_args=training_process_args,
        )
        # self.grid_dist_scale = grid_dist_scale
        self.max_candidates_num = max_candidates_num
        self._strat_order = []
        
    def _select_sample(self, i) -> ParallelisationStrategy:
        
        if i == 0:
            
            x_upper_bound = torch.tensor(ParallelisationStrategy.get_list_log_transform_upper_bound(
                num_hosts=self.num_hosts,
                num_gpus_per_host=self.num_gpus_per_host,
                batch_size=self.batch_size,
                max_dp_bucket_size_mb=2**12,
                max_model_chunks=self.max_num_microbatches,
            )) + 0.001
            
            candidates = torch.tensor(np.array([x.to_list() for x in self.candidates_list]))
            candidates = ParallelisationStrategy.log_transform(candidates) / x_upper_bound[..., :]
            
            n_samples, n_features = candidates.shape
            centroids = candidates[torch.randint(0, n_samples, (1,))]
            for _ in range(1, self.max_candidates_num):
                distances = torch.min(torch.cdist(candidates, centroids[:,:], p=2), dim=1).values
                probabilities = (distances ** 2)
                probabilities /= torch.sum(probabilities)
                next_centroid_idx = torch.multinomial(probabilities, 1).item()
                centroids = torch.cat([centroids, candidates[next_centroid_idx].unsqueeze(0)], dim=0)
                
            init_samples = centroids
            for c in init_samples:
                c = ParallelisationStrategy.undo_log_transform((c * x_upper_bound).cpu().detach())
                c = ParallelisationStrategy.from_list(c.tolist())
                self._strat_order.append(c)
                print(f'Grid - round {len(self._strat_order)} : {c}')
            
            
            # soboleng = torch.quasirandom.SobolEngine(dimension=candidates.shape[1])
            
            # while len(self._strat_order) < self.max_candidates_num:
            #     k = soboleng.draw(n=1)[0]
            #     abs_diff = torch.abs(candidates - k)
            #     closest_values, _ = torch.min(abs_diff, dim=0)
            #     closest_values = candidates[torch.argmin(abs_diff, dim=0), torch.arange(k.size(0))]
            #     try:
            #         c = ParallelisationStrategy.undo_log_transform((closest_values * x_upper_bound).cpu().detach())
            #         c = ParallelisationStrategy.from_list(c.tolist())
            #     except AssertionError:
            #         continue
            #     self._strat_order.append(c)
            #     print(f'Grid - round {len(self._strat_order)} : {c}')
                # distances = torch.norm(candidates - k, dim=1)
                # closest_idx = torch.argmin(distances)
                # closest_vector = candidates[closest_idx]
                # if ((k - closest_vector).abs()[2:6] < self.grid_dist_scale).all():
                #     c = ParallelisationStrategy.undo_log_transform((closest_vector * x_upper_bound).cpu().detach())
                #     c = ParallelisationStrategy.from_list(c.tolist())
                #     self._strat_order.append(c)
                #     print(f'Grid - round {len(self._strat_order)} : {c}')
            
        return self._strat_order[i], dict()
