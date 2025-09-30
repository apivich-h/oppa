from typing import Callable, List, Union, Dict
import yaml
import os
import random
import math

import logging
logger = logging.getLogger(__name__)

import numpy as np
import torch

from ..strategies.parallel_strategy import ParallelisationStrategy
from .baseline import ParallelismOptimiser
from ..bo_gp.gp import fit_gp
from ..bo_gp.bo import choose_next_point
from ..bo_gp.custom_components.parallelism_prior import OOM_ADD_FACTOR


MEM_VARIANCE = 1e-6


class BayesianOptimisation(ParallelismOptimiser):
    
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
        gp_kernel: str = 'matern52',
        gp_fit_steps: int = 2000,
        use_cost_informed_mean: bool = False,
        init_random_rounds: int = None,
        init_random_time: int = None,
        init_random_method: str = 'br',
        drop_oom_timings: bool = True,
        oom_factor_for_rand_explore: int = 3,
        early_terminate_bad_trials: bool = False,
        early_terminate_crit_improvement_bound: float = 0.001,
        early_terminate_score_improvement_bound: float = 0.,
        early_terminate_n_inlier: int = 5,
        init_best: int = 1.,
        gp_fit_timeout_sec: float = 10.,
        acq_fn: str = 'ucb',
        acq_fn_ucb_beta: float = 1.,
        rand_explore_proportion: float = 0.1,
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
            # transform_fn=(lambda t: (1. / t)),
            use_throughput=True,
            out_folder=out_folder,
            runner_additional_args_dir=runner_additional_args_dir,
            training_num_steps=training_num_steps,
            training_process_args=training_process_args,
        )
        
        assert init_random_rounds is None or init_random_time is None
        self.init_random_rounds = init_random_rounds
        self.init_random_time = init_random_time
        self.gp_kernel = gp_kernel
        self.gp_fit_steps = gp_fit_steps
        self.use_cost_informed_mean = use_cost_informed_mean
        self.drop_oom_timings = drop_oom_timings
        self.oom_factor_for_rand_explore = oom_factor_for_rand_explore
        self.init_random_method = init_random_method
        self.early_terminate_bad_trials = early_terminate_bad_trials
        self.early_terminate_crit_improvement_bound = early_terminate_crit_improvement_bound
        self.early_terminate_score_improvement_bound = early_terminate_score_improvement_bound
        self.early_terminate_n_inlier = early_terminate_n_inlier
        self.init_best = init_best
        self.gp_fit_timeout_sec = gp_fit_timeout_sec
        self.acq_fn = acq_fn
        self.acq_fn_ucb_beta = acq_fn_ucb_beta
        self.rand_explore_proportion = rand_explore_proportion
        
        self.init_samples = []
        self.oom_count = 0
        self.sobol = None
        self.learned_surrogate = dict()
        self.last_score_gp = None
        self.last_mem_gp = None
        
    def _fit_gp(self, train_x, train_y_quantity, train_y, train_y_var, gp_kernel, step=0, last_round_gp=None):
        
        if '12' in gp_kernel:
            nu = 0.5
        elif '32' in gp_kernel:
            nu = 1.5
        elif '52' in gp_kernel:
            nu = 2.5
        
        if gp_kernel == 'rbf':
            print(f'Using RBF kernel')
            kernel_name = 'rbf'
            gp_kernel_args = dict()
        elif gp_kernel.startswith('matern'):
            print(f'Using Matern kernel with {nu=}')
            kernel_name = 'matern'
            gp_kernel_args = {'nu': nu}
        elif gp_kernel.startswith('dk'):
            kernel_name = 'dk'
            base_kernel = gp_kernel.removeprefix('dk')
            gp_kernel_args = {
                'hidden_dim_seq': (64, 8),
                'freeze_nn': False
            }
            if base_kernel == 'r':
                gp_kernel_args['base_kernel'] = 'rbf'
            elif base_kernel.startswith('m'):
                gp_kernel_args['base_kernel'] = 'matern'
                gp_kernel_args['nu'] = nu
        else:
            raise ValueError(f'Invalid {gp_kernel=}')
        
        if self.use_cost_informed_mean:
            mean_name = train_y_quantity
            mean_args = {
                'x_upper_bound': self._x_upper_bound,
                'max_mem_GB': self.gpu_max_mem_GB,
                'step': step,
            }
        else:
            mean_name = 'constant'
            mean_args = dict()
            
        gp_args = dict(
            gp_kernel=kernel_name,
            gp_kernel_args=gp_kernel_args,
            gp_mean=mean_name,
            gp_mean_args=mean_args,
            gp_fit_timeout_sec=self.gp_fit_timeout_sec,
        )
        print('gp_args =', gp_args)
        model, res = fit_gp(
            train_x=train_x,
            train_y=train_y,
            train_y_var=train_y_var,
            restore_gp=last_round_gp,
            separately_fit_mean=self.use_cost_informed_mean,
            **gp_args,
        )
        # print(f'{model.mean_module=}')
        # print(f'{model.covar_module=}')
        # print(f'Fitting done -- res = {res}')
        return model, res
    
    def _select_sample(self, i) -> ParallelisationStrategy:
        
        # x_upper_bound = torch.tensor(ParallelisationStrategy.get_list_log_transform_upper_bound(
        #     num_hosts=self.num_hosts,
        #     num_gpus_per_host=self.num_gpus_per_host,
        #     batch_size=self.batch_size,
        #     max_dp_bucket_size_mb=2**12,
        #     max_model_chunks=self.max_num_microbatches,
        # )) + 0.001
        # # x_upper_bound = x_upper_bound.max() * torch.ones_like(x_upper_bound)
        # print(f'SCBO start: {x_upper_bound=}')
        
        aux = dict()
        
        # extra test for OOM cases
        # if face OOM too often, might need to explore a bit more
        if (i > 0) and (self.queried_mem[-1] > self.gpu_max_mem_GB):
            self.oom_count += 1
            print(f'Found {self.oom_count} consecutive OOM cases')
        else:
            self.oom_count = 0
            
        rand_sel_flag = (
            ((self.init_random_rounds is None) and (sum(self.queried_total_time) < self.init_random_time)) or
            ((self.init_random_time is None) and (i < self.init_random_rounds)) or
            (np.random.rand() < self.rand_explore_proportion)
        )
        
        # candidates_set = set(self.candidates_list)
        candidates_set = set(self.candidates_list) - set(self.queried_strat)
        assert len(candidates_set) == len(self.candidates_list) - i, (len(candidates_set), len(self.candidates_list), i, self.queried_strat, set(self.queried_strat))
        candidates = torch.tensor(np.array([x.to_list() for x in candidates_set]))
        candidates = ParallelisationStrategy.log_transform(candidates) / self._x_upper_bound[..., :]
        print(f'Pre BO: {len(candidates)=}')
        
        # selected_set = {x.to_string() for x in self.queried_strat}
        # candidates = torch.tensor(np.array([x.to_list() for x in self.candidates_list if x.to_string() not in selected_set]))
        # candidates = ParallelisationStrategy.log_transform(candidates) / x_upper_bound[..., :]
        # print(f'Pre SCBO: {len(candidates)=}')
        
        run_bo_flag = (i > 0) and (not np.isnan(self.queried_y).all())
        
        if run_bo_flag:
        
            train_x = ParallelisationStrategy.log_transform(torch.tensor(np.array(self.queried_X)).double()) / self._x_upper_bound[...,:]
            train_y = torch.tensor(np.array(self.queried_y)).reshape(-1, 1).double()
            train_y_std = torch.tensor(np.array(self.queried_y_std)).reshape_as(train_y).double()
            train_y_var = train_y_std ** 2
            train_y_isnan = train_y.isnan().reshape(-1)
            
            if train_y_isnan.all():
                train_y = torch.zeros_like(train_y)
                train_y_var = None
                max_train_y = train_y[-1]
                max_train_x = train_x[-1]
            else:
                if self.drop_oom_timings:
                    train_x = train_x[~train_y_isnan]
                    train_y = train_y[~train_y_isnan]
                    train_y_var = train_y_var[~train_y_isnan]
                else:
                    train_y[train_y_isnan] = 0.
                    train_y_var[train_y_isnan] = torch.max(train_y_var[~train_y_isnan])
                max_train_idx = torch.argmax(train_y.reshape(-1))
                max_train_y = train_y[max_train_idx]
                max_train_x = train_x[max_train_idx]
            
            print(f'Time GP sizes: {train_x.shape=}, {train_y.shape=}, {train_y_var.shape=}')
            print(f'Time GP maxval: {max_train_x=}, {max_train_y=}')
            model_score, res_score = self._fit_gp(
                train_x=train_x,
                train_y_quantity=('throughput-' + ('sh' if self.num_hosts == 1 else 'mh')),
                train_y=train_y,
                train_y_var=train_y_var,
                gp_kernel=self.gp_kernel,
                step=i,
                last_round_gp=self.last_score_gp,
            )
            self.last_score_gp = model_score
            
            train_x_mem = ParallelisationStrategy.log_transform(torch.tensor(np.array(self.queried_X)).double()) / self._x_upper_bound[...,:]
            train_mem = torch.tensor(np.array(self.queried_mem)).reshape(-1, 1).double()
            train_mem_isinf = train_mem.isinf().reshape(-1)
            print(f'Mem GP sizes: {train_x_mem.shape=}, {train_mem.shape=}')
            
            if train_mem_isinf.any():
                print('Training model for memory constraints')
                train_mem[train_mem_isinf] = self.gpu_max_mem_GB * (1. + OOM_ADD_FACTOR)
                train_mem = (train_mem - self.gpu_max_mem_GB) / self.gpu_max_mem_GB
                model_constraint, res_constraint = self._fit_gp(
                    train_x=train_x_mem,
                    train_y_quantity='memory',
                    train_y=train_mem,
                    train_y_var=None,
                    gp_kernel=self.gp_kernel,
                    last_round_gp=self.last_mem_gp,
                )
                self.last_mem_gp = model_constraint
            else:
                print('NOT training model for memory constraints')
                train_mem = (train_mem - self.gpu_max_mem_GB) / self.gpu_max_mem_GB
                model_constraint = None
                res_constraint = None
                
            save_covar_params = not (self.gp_kernel.startswith('dk'))
            extractor = lambda module: {k: v.tolist() for (k, v) in module.named_parameters()}
            aux.update({
                'gp_score_mean_params': extractor(model_score.mean_module),
                'gp_score_covar_params': extractor(model_score.covar_module) if save_covar_params else None,
                'gp_score_llh_params': extractor(model_score.likelihood),
                'gp_score_res': res_score,
            })
            if model_constraint is not None:
                aux.update({
                    'gp_mem_mean_params': extractor(model_constraint.mean_module),
                    'gp_mem_covar_params': extractor(model_constraint.covar_module) if save_covar_params else None,
                    'gp_mem_llh_params': extractor(model_constraint.likelihood),
                    'gp_mem_res': res_constraint,
                })
            if (i + 1) % 5 == 0:
                self.learned_surrogate[i+1] = {
                    'throughput': model_score,
                    'mem': model_constraint,
                    'x_upper_bound': self._x_upper_bound,
                    'gpu_max_GB': self.gpu_max_mem_GB,
                }
        
        # random selection
        if rand_sel_flag or (self.oom_count >= self.oom_factor_for_rand_explore):
            
            acq_score = None
                
            if self.init_random_method == 'br':
                if len(self.init_samples) == 0:
                    self.init_samples = []
                    unique_prefix_dict = dict()
                    for c in list(set(self.candidates_list) - set(self.queried_strat)):
                        prefix = (c.num_gpus, c.num_hosts, c.dp_size, c.tp_size, c.pp_size, c.sp_size)
                        if prefix in unique_prefix_dict.keys():
                            unique_prefix_dict[prefix].append(c)
                        else:
                            unique_prefix_dict[prefix] = [c]
                    for _ in range(len(unique_prefix_dict.keys())):
                        prefix = random.choice(list(unique_prefix_dict.keys()))
                        c_with_prefix = unique_prefix_dict.pop(prefix)
                        c = random.choice(c_with_prefix)
                        self.init_samples.append(c)
                    print(f'{self.init_samples=}')
                c = self.init_samples.pop(0)
                
            elif self.init_random_method == 'kmbr':
                unique_prefix_dict = dict()
                for c in list(set(self.candidates_list) - set(self.queried_strat)):
                    prefix = (c.num_gpus, c.num_hosts, c.dp_size, c.tp_size, c.pp_size, c.sp_size)
                    if prefix in unique_prefix_dict.keys():
                        unique_prefix_dict[prefix].append(c)
                    else:
                        unique_prefix_dict[prefix] = [c]
                prefix = random.choice(list(unique_prefix_dict.keys()))
                print(f'{prefix=}')
                c_with_prefix = unique_prefix_dict.pop(prefix)
                candidates_subspace = torch.tensor(np.array([x.to_list() for x in c_with_prefix]))
                candidates_subspace = ParallelisationStrategy.log_transform(candidates_subspace) / self._x_upper_bound[..., :]
                n_samples, n_features = candidates_subspace.shape
                if i == 0:
                    self.centroids = candidates_subspace[torch.randint(0, n_samples, (1,))]
                else:
                    distances = torch.min(torch.cdist(candidates_subspace, self.centroids, p=2), dim=1).values
                    probabilities = distances ** 2
                    probabilities /= torch.sum(probabilities)
                    next_centroid_idx = torch.multinomial(probabilities, 1).item()
                    self.centroids = torch.cat([self.centroids, candidates_subspace[next_centroid_idx].unsqueeze(0)], dim=0)  
                print(f'{self.centroids[-1]=}')
                x = self.centroids[-1] * self._x_upper_bound[..., :]
                c = ParallelisationStrategy.undo_log_transform((x).cpu().detach())
                c = ParallelisationStrategy.from_list(c.tolist())
            
            elif self.init_random_method == 'kmeans':
                n_samples, n_features = candidates.shape
                if i == 0:
                    self.centroids = candidates[torch.randint(0, n_samples, (1,))]
                else:
                    distances = torch.min(torch.cdist(candidates, self.centroids, p=2), dim=1).values
                    probabilities = distances ** 2
                    probabilities /= torch.sum(probabilities)
                    next_centroid_idx = torch.multinomial(probabilities, 1).item()
                    self.centroids = torch.cat([self.centroids, candidates[next_centroid_idx].unsqueeze(0)], dim=0)  
                print(self.centroids[-1])
                x = self.centroids[-1] * self._x_upper_bound[..., :]
                c = ParallelisationStrategy.undo_log_transform((x).cpu().detach())
                c = ParallelisationStrategy.from_list(c.tolist())
            
            elif self.init_random_method == 'random':
                s = torch.randint(low=0, high=self.candidates_count, size=(1,))[0]
                c = self.candidates_list[s]
                
            elif self.init_random_method == 'sobol':
                if self.sobol is None:
                    self.sobol = torch.quasirandom.SobolEngine(dimension=candidates.shape[1], scramble=False)
                pt = self.sobol.draw(1)
                print(f'{pt.tolist()=}')
                dists = torch.cdist(pt.float(), candidates.float(), p=1)
                idx = torch.argmin(dists, dim=1)
                print(f'closest={candidates[idx].tolist()}')
                c = ParallelisationStrategy.undo_log_transform((candidates[idx].reshape(-1) * self._x_upper_bound).cpu().detach())
                c = ParallelisationStrategy.from_list(c.tolist())
                
            else:
                raise ValueError('Bad init_random_method')
            
           
        # BO based method
        else:
            
            if self.acq_fn == 'ei':
                acq_fn_args = {
                    'best_f': max_train_y,
                }
            elif self.acq_fn == 'ucb':
                acq_fn_args = {
                    'beta': self.acq_fn_ucb_beta,
                }
            
            x_next, acq_score = choose_next_point(
                model_objective=model_score,
                model_constraint=model_constraint,
                acq_fn=self.acq_fn,
                acq_fn_args=acq_fn_args,
                acq_candidates=candidates,
                acq_opt_args=dict(),
            )
            # batch_aux = {'acq_score': acq_score}
            batch_aux = dict()
            
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
            
            aux.update(batch_aux)

        run_extra_args = dict()
        if run_bo_flag:
            x_next = torch.tensor(np.array(c.to_log_list())) /  self._x_upper_bound
            print(f'Next strat_emb = {c.to_log_list()}')
            print(f'Next strat_log_norm_emb = {x_next.tolist()}')
            model_score.eval()
            posterior = model_score(x_next.reshape(1, -1))
            x_next_prior_mean = float(posterior.mean.reshape(-1).detach().cpu().numpy()[0])
            x_next_prior_cov = float(posterior.variance.reshape(-1).detach().cpu().numpy()[0])
            # x_next_prior_mean = model_score.mean_module()[0].tolist()
            # x_next_prior_cov = model_score.covar_module(x_next.reshape(1, -1))[0][0].tolist()
            if self.early_terminate_bad_trials:
                run_extra_args['early_terminate'] = {
                    'do_early_terminate': True,
                    'method': self.acq_fn,
                    'use_throughput': True,
                    'curr_best': max(self.init_best, max_train_y[0].tolist()),
                    'prior_mean': x_next_prior_mean,
                    'prior_var': x_next_prior_cov,
                    'crit_threshold': self.early_terminate_crit_improvement_bound,
                    'mean_threshold': self.early_terminate_score_improvement_bound,
                    'ucb_beta': self.acq_fn_ucb_beta,
                    'n_inlier': self.early_terminate_n_inlier,
                }
        else:
            x_next_prior_mean = None
            x_next_prior_cov = None
            if self.early_terminate_bad_trials:
                # to get a rough scale first
                run_extra_args['early_terminate'] = {
                    'do_early_terminate': True,
                    'method': self.acq_fn,
                    'use_throughput': True,
                    'curr_best': self.init_best,
                    'prior_mean': 0.,
                    'prior_var': 1.,
                    'crit_threshold': self.early_terminate_crit_improvement_bound,
                    'mean_threshold': self.early_terminate_score_improvement_bound,
                    'ucb_beta': self.acq_fn_ucb_beta,
                }

        aux.update({
            'rand_sel_triggered': rand_sel_flag,
            'x_next': c.to_string(),
            'x_next_prior_mean': x_next_prior_mean,
            'x_next_prior_cov': x_next_prior_cov,
            'x_next_acq_score': acq_score,
            'run_extra_args': run_extra_args,
        })
        
        return c, aux
