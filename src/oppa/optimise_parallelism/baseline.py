from typing import Callable, List, Union, Dict, Tuple
import yaml
import os
import time
import math
import itertools
import shutil

import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch

from ..strategies.parallel_strategy import ParallelisationStrategy, check_implementation_specific_strategy
from ..runners import run_training_subprocess

def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)


class ParallelismOptimiser:
    
    def __init__(
        self,
        batch_size: int = 1,
        num_hosts: int = 1,
        num_gpus_per_host: int = 1,
        max_dp_size: Union[int, None] = None,
        max_tp_size: Union[int, None] = None, 
        max_pp_size: Union[int, None] = None, 
        max_sp_size: Union[int, None] = 1, 
        max_num_model_chunks: int = 4,
        max_num_microbatches: Union[int, None] = None,
        tp_divisible_by: int = 4,
        pp_divisible_by: int = 4,
        gpu_max_mem_GB: float = 40.,
        fixed_args: Dict = None,
        max_out_gpus: bool = True,
        max_out_machines: bool = False,
        # transform_fn: Callable = None,
        use_throughput: bool = True,
        out_folder: str = './results',
        runner_additional_args_dir: str = './additional_args.yaml',
        training_num_steps: int = 100,
        training_process_args: Dict = None,
    ):
        
        self.batch_size = batch_size
        self.num_hosts = num_hosts
        self.num_gpus_per_host = num_gpus_per_host
        num_gpus = self.num_hosts * self.num_gpus_per_host
        self.max_dp_size = num_gpus if max_dp_size is None else max_dp_size
        self.max_tp_size = num_gpus if max_tp_size is None else max_tp_size
        self.max_pp_size = num_gpus if max_pp_size is None else max_pp_size
        self.max_sp_size = num_gpus if max_sp_size is None else max_sp_size
        self.max_num_microbatches = batch_size if max_num_microbatches is None else max_num_microbatches
        self.max_num_model_chunks = max_num_model_chunks
        self.gpu_max_mem_GB = gpu_max_mem_GB
        self.tp_divisible_by = tp_divisible_by
        self.pp_divisible_by = pp_divisible_by
        self.max_out_gpus = max_out_gpus
        self.max_out_machines = max_out_machines
        # self.transform_fn = transform_fn
        self.use_throughput = use_throughput
        self.out_folder = os.path.abspath(out_folder)
        # self.model_out = os.path.abspath(os.path.join(out_folder, 'model_out'))
        # self.aux_out = os.path.abspath(os.path.join(out_folder, 'aux_out'))
        self.training_num_steps = training_num_steps
        self.training_process_args = training_process_args
        self.runner_additional_args_dir = os.path.abspath(runner_additional_args_dir)
        if fixed_args is None:
            self.fixed_args = {
                'use_grad_checkpoint': True,
                'dp_outside': True,
                'overlap_communication_for_zero': True,
                'overlap_allgather_for_zero': False,
                'overlap_p2p_for_pp': True,
            }
        else:
            self.fixed_args = fixed_args

        self._prep_opt_states()

    def _prep_opt_states(self):
        self.candidates_list = self._generate_all_candidates()
        self.candidates_count = len(self.candidates_list)
        self.queried_count = 0
        self.queried_X = []
        self.queried_y = []
        self.queried_y_std = []
        self.queried_mem = []
        self.queried_mem_std = []
        self.queried_strat = []
        self.queried_all_data = []
        self.queried_valid_trial = []
        self.queried_raw_time = []
        self.queried_raw_mem = []
        self.queried_runtime = []
        self.queried_selection_time = []
        self.queried_total_time = []
        self.queried_aux = []
        self.best_config = None
        self.best_config_time = float('inf')
        self.best_config_time_cumulative = []
        self._x_upper_bound = torch.tensor(ParallelisationStrategy.get_list_log_transform_upper_bound(
            num_hosts=self.num_hosts,
            num_gpus_per_host=self.num_gpus_per_host,
            batch_size=self.batch_size,
            max_dp_bucket_size_mb=2**12,
            max_model_chunks=self.max_num_microbatches,
        ))
                        
    def _generate_all_candidates(self):
        # I know this whole function is inefficient
        # I will refactor this later (but watch me end up never ending up refactoring this)
                
        num_gpus_total = self.num_hosts * self.num_gpus_per_host
                
        # generate all allowed gpu configs
        gpu_configs = []
        dp_iter = range(1, self.max_dp_size+1)
        for dp in dp_iter:
            if self.batch_size % dp != 0:
                # DP should divide into the batch size
                continue
            tp_iter = range(1, min(self.max_tp_size, num_gpus_total // dp)+1)
            tp_iter = [tp for tp in tp_iter if (self.tp_divisible_by % tp == 0)]
            for tp in tp_iter:
                if not is_power_of_two(tp):
                    # TP should divide matrices properly
                    # currently assume that the matrices are a power of 2
                    # TODO: change that assumption
                    continue
                pp_iter = range(1, min(self.max_pp_size, num_gpus_total // (dp * tp))+1)
                pp_iter = [pp for pp in pp_iter if (self.pp_divisible_by % pp == 0)]
                for pp in pp_iter:
                    sp_iter = range(1, min(self.max_sp_size, num_gpus_total // (dp * tp * pp))+1)
                    for sp in sp_iter:
                        gpu_configs.append((dp, tp, pp, sp))
                        
        # generate parameter for microbatches
        # number of microbatches should be at least the degree of pipeline parallelism
        gpu_microbatch_configs = []
        for conf in gpu_configs:
            dp, _, pp, _ = conf
            allowable_microbatches = [i for i in range(1, self.batch_size+1) 
                                      if ((self.batch_size // dp) % i == 0) and (i <= self.max_num_microbatches)]
            # should have more microbatches than PP
            # number of microbatches should not exceed the number of samples we have
            gpu_microbatch_configs.extend([conf + (b,) for b in allowable_microbatches if b >= pp])
        del gpu_configs
                
        # generate parameter for model chunking for sequence parallelism
        full_configs = []
        mc_list = [m for m in range(1, self.max_num_model_chunks+1)]
        for conf in gpu_microbatch_configs:
            _, _, pp, sp, bsz = conf
            for mc in mc_list:
                if (mc * pp <= bsz) and (mc * pp <= self.pp_divisible_by):
                    full_configs.append(conf + (mc,))
        del gpu_microbatch_configs
        
        full_configs_with_hostcount = []
        for conf in full_configs:
            
            dp, tp, pp, sp, mb, mc = conf
            num_gpus_required = dp * tp * pp * sp
            
            if not (is_power_of_two(num_gpus_required) or num_gpus_required == num_gpus_total):
                continue
            
            if self.num_hosts == 1:
                if num_gpus_required == self.num_gpus_per_host:
                    # only one host, and we want to use all GPUs
                    full_configs_with_hostcount.append(conf + (num_gpus_required, 1))
                elif (not self.max_out_gpus) and (num_gpus_required < self.num_gpus_per_host):
                    # only one host, but it is okay if we don't use all GPUs
                    full_configs_with_hostcount.append(conf + (num_gpus_required, 1))
                else:
                    continue
                
            elif self.max_out_gpus and not self.max_out_machines:
                if num_gpus_required == num_gpus_total:
                    # many hosts, and we want to use all GPUs
                    full_configs_with_hostcount.append(conf + (num_gpus_required, self.num_hosts))
                elif self.max_out_gpus and (num_gpus_required % self.num_gpus_per_host == 0):
                    num_hosts_required = num_gpus_required // self.num_gpus_per_host
                    if (num_hosts_required == self.num_hosts):
                        full_configs_with_hostcount.append(conf + (num_gpus_required, num_hosts_required))
                    elif not self.max_out_machines and (num_hosts_required <= self.num_hosts):
                        full_configs_with_hostcount.append(conf + (num_gpus_required, num_hosts_required))
                    else:
                        continue
                else:
                    continue
            
            elif self.max_out_gpus and self.max_out_machines:
                
                if num_gpus_required == self.num_gpus_per_host * self.num_hosts:
                    # only one host, and we want to use all GPUs
                    full_configs_with_hostcount.append(conf + (num_gpus_required, self.num_hosts))
                else:
                    continue
            
            else:
                raise NotImplementedError
            
        del full_configs

        # add parallelism for ZeRO optimiser
        full_configs_objects = []
        for dp, tp, pp, sp, mb, mc, ng, nh in full_configs_with_hostcount:
            for zs in [0, 1, 2, 3]:  # TODO: make stage 3 optimisation also compatible
                for dp_bk in 2 ** np.arange(0, 13, 2):
                    for z_bk in 2 ** np.arange(0, 13, 2):
                        for use_gc, ocz, oaz, opp in itertools.product([True, False], repeat=4):
                            try:
                                d = dict(
                                    num_gpus=ng,
                                    num_hosts=nh,
                                    dp_size=dp,
                                    tp_size=tp,
                                    pp_size=pp,
                                    sp_size=sp,
                                    dp_bucket_size_mb=dp_bk,
                                    zero_stage=zs,
                                    zero_bucket_size_mb=z_bk,
                                    num_microbatches=mb,
                                    num_model_chunks=mc,
                                    use_grad_checkpoint=use_gc,
                                    overlap_allgather_for_zero=oaz,
                                    overlap_communication_for_zero=ocz,
                                    overlap_p2p_for_pp=opp,
                                )
                                d.update(self.fixed_args)
                                conf = ParallelisationStrategy(**d)
                                check_implementation_specific_strategy(
                                    p=conf,
                                    implementation=self.training_process_args['training_framework'],
                                )
                                full_configs_objects.append(conf)
                            except AssertionError:
                                pass
                   
        full_configs_objects = sorted(list(set(full_configs_objects)) ) # remove duplicates
        assert len(full_configs_objects) > 0
        print(f'Generated all candidates. Total number is {len(full_configs_objects)}.')
        return full_configs_objects
    
    def _run_timing_process(self, para_strategy: ParallelisationStrategy, additional_args=dict(), rerun=True, only_setup=False):
        
        time.sleep(1)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = os.path.join(self.out_folder, 'trials', f'{timestamp}--{para_strategy.to_string()}')
        os.makedirs(prefix)
        
        with open(self.runner_additional_args_dir, 'r') as f:
            d = yaml.safe_load(f)
        d.update(additional_args)
        runner_args = os.path.join(prefix, 'runner_args.yaml')
        with open(runner_args, 'w') as f:
            yaml.safe_dump(d, f, sort_keys=False)
        
        model_out_path, aux_out_path, reran = run_training_subprocess(
            parallelisation_strategy=para_strategy,
            batch_size=self.batch_size,
            n_steps=self.training_num_steps,
            n_epochs=(1 if self.training_num_steps is None else None),
            model_out=os.path.join(prefix, 'model_out'),
            aux_out=os.path.join(prefix, 'train_profile.yaml'),
            save_step_timing=True,
            verbose=False,
            capture_outputs=False,
            always_rerun=rerun,
            only_setup=only_setup,
            only_benchmark=(not only_setup),
            model_additional_args_path=runner_args,
            **self.training_process_args,
        )
        return prefix, model_out_path, aux_out_path, reran
    
    def running_setup(self):
        prefix, _, _, _ = self._run_timing_process(
            para_strategy=ParallelisationStrategy(
                num_gpus=self.num_hosts,
                num_hosts=self.num_hosts,
                dp_size=self.num_hosts,
                tp_size=1,
                pp_size=1,
                sp_size=1,
                dp_bucket_size_mb=1,
                zero_stage=0,
                zero_bucket_size_mb=1,
                num_microbatches=1,
                num_model_chunks=1,
                use_grad_checkpoint=True,
                overlap_communication_for_zero=False,
                overlap_allgather_for_zero=False,
                overlap_p2p_for_pp=False,
            ),
            only_setup=True,
        )
        shutil.rmtree(prefix)
        
    def _get_timing(self, para_strategy: ParallelisationStrategy, additional_args=dict()):
        
        # try to do it a few times in case of some weird error
        attempts = 0
        success = False

        while not success:
            
            try:
                _, _, aux_out_path, _ = self._run_timing_process(
                    para_strategy=para_strategy,
                    additional_args=additional_args,
                    rerun=True,
                )
                with open(aux_out_path, 'r') as f:
                    d = yaml.safe_load(f)
                success = True

            except Exception as e:
                attempts += 1
                if attempts >= 3:
                    raise e
            
        if d['ran_successfully'] :
            # remove the first timestep since it probably contains some additional overhead,
            # and is often much larger time than the rest of the timsteps
            # time_per_step = np.array(d['time_results']['time_per_step'])[1:]
            # N_count = time_per_step.shape[0]
            # if self.transform_fn is not None:
            #     transformed_time_per_step = self.transform_fn(time_per_step)
            #     time_mean = np.mean(transformed_time_per_step)
            #     time_std = np.std(transformed_time_per_step) / math.sqrt(N_count)
            # else:
            #     time_mean = np.mean(time_per_step)  # negative to make it maximisation
            #     time_std = np.std(time_per_step) / math.sqrt(N_count)
            if self.use_throughput:
                time_mean = d['time_results']['throughput_mean']
                time_std = d['time_results']['throughput_std']
            else:
                time_mean = d['time_results']['time_mean']
                time_std = d['time_results']['time_std']
            N_count = d['time_results']['n_inlier']
            ts = d['time_results']['all_time_steps']
            is_full_trial = (ts is not None) and (len(ts) >= self.training_num_steps)
                
        else:
            time_mean = float('nan')
            time_std = float('nan')
            N_count = 0
            is_full_trial = False
                
        mem_constraints = d['memory_results']['peak_mem']
        # TODO: definitely change how the std is computed
        # since the GPU usage can heavily vary between GPUs, but we just std() on all currently
        mem_std = 0.  # np.std(d['memory_results']['peak_mem_per_gpu'])
                
        return {
            'ran_successfully': d['ran_successfully'], 
            'time_mean': time_mean, 
            'time_std': time_std,
            'mem_mean': mem_constraints,
            'mem_std': mem_std,
            'N_count': N_count,
            'is_full_trial': is_full_trial,
            'all_data': d,
        }
    
    
    def _query_sample(self, para_strategy: ParallelisationStrategy, additional_args=dict()):
        
        print(f'Round {self.queried_count+1} running: {para_strategy}')
        data = self._get_timing(para_strategy=para_strategy, additional_args=additional_args)
        ran_successfully = data['ran_successfully'] 
        time_mean = data['time_mean']
        time_std = data['time_std'] 
        mem_mean = data['mem_mean']
        mem_std = data['mem_std']
        N_count = data['N_count']
        
        # if ran_successfully:
            
        self.queried_X.append(para_strategy.to_list())
        self.queried_y.append(time_mean)
        self.queried_y_std.append(time_std)
        # self.queried_mem.append(min(self.gpu_max_mem_GB, mem_mean))
        self.queried_mem.append(mem_mean)
        self.queried_mem_std.append(mem_std)
        print(f'Round {self.queried_count+1} score: {time_mean} +- {time_std} (count={N_count}) | mem = {mem_mean}')
            
        # else:
            
        #     self.queried_X.append(para_strategy.to_list())
        #     self.queried_y.append(float('nan'))
        #     self.queried_y_std.append(float('nan'))
        #     self.queried_mem.append(self.gpu_max_mem_GB)
        #     self.queried_mem_std.append(0.)
        #     print(f'Round {self.queried_count+1} score: None (run had an error)')

        self.queried_strat.append(para_strategy)
        self.queried_all_data.append(data)
        if data['all_data']['time_results'] is None:
            self.queried_raw_time.append([float('nan'), float('nan')])
        else:
            self.queried_raw_time.append(data['all_data']['time_results']['all_time_steps'])
        self.queried_raw_mem.append(data['all_data']['memory_results']['peak_mem_per_gpu'])
        
        self.queried_count += 1
        return time_mean, time_std
        
    def _select_sample(self, i) -> Tuple[ParallelisationStrategy, Dict]:
        raise NotImplementedError

    def _should_stop(self, r, aux, min_rounds, max_rounds, max_time) -> bool:
        if (r > min_rounds) and (sum(self.queried_total_time) > max_time):
            print(f'Runtime is {sum(self.queried_total_time)} seconds -- time limit hit...')
            return True
        elif r > max_rounds:
            print(f'Currently in round {r+1} -- round limit hit...')
            return True
        else:
            return False
        
    def run(self, min_rounds: int = 10, max_rounds: int = 50, max_time: float = float('inf')):
        for r in range(max_rounds):
            t1 = time.time()
            x, aux = self._select_sample(r)
            if 'run_extra_args' in aux.keys():
                additional_args = aux['run_extra_args']
            else:
                additional_args = dict()
            t2 = time.time()
            self._query_sample(para_strategy=x, additional_args=additional_args)
            valid_trial = self.queried_all_data[r]['is_full_trial']
            time_mean = self.queried_all_data[r]['all_data']['time_results']['time_mean']
            mem_mean = self.queried_all_data[r]['all_data']['memory_results']['peak_mem']
            if valid_trial and (mem_mean < self.gpu_max_mem_GB) and (time_mean < self.best_config_time):
                self.best_config = x
                self.best_config_time = time_mean
            t3 = time.time()
            self.queried_selection_time.append(t2 - t1)
            self.queried_runtime.append(t3 - t2)
            self.queried_total_time.append(t3 - t1)
            self.queried_aux.append(aux)
            self.queried_valid_trial.append(valid_trial)
            self.best_config_time_cumulative.append(self.best_config_time)
            self.get_queries_info().to_csv(os.path.join(self.out_folder, 'ran_queries.csv'), float_format='%.6f')
            if self._should_stop(
                r=r, 
                aux=aux, 
                min_rounds=min_rounds, 
                max_rounds=max_rounds, 
                max_time=max_time,
            ):
                break

    def get_best_parallel_strategy(self):
        return self.best_config
            
    def pre_query_all_timings(self, runs=None):
        c = 0
        for x in self.candidates_list:
            _, _, _, reran = self._run_timing_process(x, rerun=False)
            if reran:
                c += 1
            if (runs is not None) and (c >= runs):
                break

    def get_queries_info(self):
        data = []
        for i, x in enumerate(self.queried_X):
            strat = ParallelisationStrategy.from_list(x)
            strat_dict = strat.to_dict()
            # try:
            #     # remove first element as outlier
            #     time_mean = float(np.mean(self.queried_raw_time[i][1:]))
            #     time_std = float(np.std(self.queried_raw_time[i][1:])) / math.sqrt(len(self.queried_raw_time[i][1:]))
            # except:
            #     time_mean = float('nan')
            #     time_std = float('nan')
            strat_dict.update({
                'strat_str': strat.to_string(),
                'strat_emb': strat.to_list(),
                'strat_logemb': strat.to_log_list(),
                'strat_logemb_norm': list(np.array(strat.to_log_list()) / self._x_upper_bound.cpu().numpy()),
                'score_mean': self.queried_y[i],
                'score_std': self.queried_y_std[i],
                # 'time_mean': time_mean, 
                # 'time_std': time_std,
                'mem_mean': self.queried_mem[i],
                'mem_std': self.queried_mem_std[i],
                'considered': self.queried_valid_trial[i],
                'current_best_time': self.best_config_time_cumulative[i],
                'current_best_throughput': 1. / self.best_config_time_cumulative[i],
                'algtime_cumulative': sum(self.queried_total_time[:i+1]),
                'algtime_round': self.queried_total_time[i],
                'algtime_select': self.queried_selection_time[i],
                'algtime_trial': self.queried_runtime[i],
                'raw_time': self.queried_raw_time[i],
                'raw_mem': self.queried_raw_mem[i],
            })
            strat_dict.update({f'aux_{k}': v for (k, v) in self.queried_aux[i].items()})
            data.append(strat_dict)
        return pd.DataFrame.from_records(data)
