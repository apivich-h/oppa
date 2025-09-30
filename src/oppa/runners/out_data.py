
import datetime
import yaml
from typing import List, Dict

from oppa.strategies.parallel_strategy import ParallelisationStrategy
from oppa.utils.robust_stats import robust_mean_and_std, robust_reciprocal_mean_and_std


def save_training_info(
    out_path: str,
    para_strategy: ParallelisationStrategy,
    ran_successfully: bool,
    execution_time,
    time_per_step: List[float] = None,
    memory_results: List[float] = None,
    other_results: Dict = dict(),
    do_save: bool = True,
):
    
    ts = datetime.datetime.now()
    
    # discard first step in computation because usually abnormally higher
    if (time_per_step is not None) and (len(time_per_step) > 2):
        time_mean, time_std, n_count_time = robust_mean_and_std(time_per_step[1:])
        thr_mean, thr_std, n_count_thr = robust_reciprocal_mean_and_std(time_per_step[1:])
        assert n_count_time == n_count_thr
        time_results = {
            'time_mean': time_mean,
            'time_std': time_std,
            'throughput_mean': thr_mean,
            'throughput_std': thr_std,
            'n_inlier': n_count_time,
            'all_time_steps': time_per_step,
        }
    else:
        time_results = {
            'time_mean': float('nan'),
            'time_std': float('nan'),
            'throughput_mean': float('nan'),
            'throughput_std': float('nan'),
            'n_inlier': float('nan'),
            'all_time_steps': [float('nan')],
        }
    
    if ran_successfully:
        
        data_to_store = {
            'timestamp': ts,
            'input_hyperparams': para_strategy.to_dict(),
            'ran_successfully': True,
            'execution_time': execution_time,
            'time_results': time_results,
            'memory_results': {
                'peak_mem': max(memory_results),
                'peak_mem_per_gpu': memory_results,
            }
        }
    
    else:
        data_to_store = {
            'timestamp': ts,
            'input_hyperparams': para_strategy.to_dict(),
            'ran_successfully': False,
            'execution_time': execution_time,
            'time_results': time_results,
            'memory_results': {
                'peak_mem': float('inf'),
                'peak_mem_per_gpu': [float('inf') for _ in range(para_strategy.num_gpus)],
            }
        }
        
    data_to_store.update(other_results)
    
    if do_save:
        with open(out_path, 'w') as f:
            yaml.dump(data_to_store, f, sort_keys=False)
        
    return data_to_store