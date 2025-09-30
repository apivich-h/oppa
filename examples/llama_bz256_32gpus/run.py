import os
import sys

alg = sys.argv[1]
seed = int(sys.argv[2])
out_folder = os.path.join('.', 'results', alg, f'seed_{seed}')
N_STEPS = 30
ROUNDS = 5
TIME = 60 * 60
INIT_BEST = 0.14
CRIT_EARLY_TERMINATE_BOUND = 0.001
SCORE_EARLY_TERMINATE_BOUND = 0.
INIT_RAND_ROUNDS = 8 if alg.startswith('pm') else 5


if os.path.exists(out_folder):
    exit()
print('='*50 + '\n', f'Running to {out_folder}', '\n' + '='*50)


import logging
import random
import numpy as np
import torch

logging.getLogger('optimise_parallelism').setLevel(logging.INFO)

with open(os.environ['PBS_NODEFILE']) as file:
    HOSTNAMES = list(dict.fromkeys([line.rstrip() for line in file]))

print('Hosts =', HOSTNAMES)

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

po_args = dict(
    batch_size=256,
    num_hosts=8,
    num_gpus_per_host=4,
    max_dp_size=16,
    max_tp_size=4, 
    max_pp_size=None,
    max_sp_size=1,
    tp_divisible_by=4,
    pp_divisible_by=32,
    max_num_microbatches=128,
    max_num_model_chunks=4,
    gpu_max_mem_GB=40.,
    max_out_gpus=True,
    max_out_machines=True,
    fixed_args={
        # 'overlap_communication_for_zero': False,
        # 'overlap_allgather_for_zero': False,
        # 'overlap_p2p_for_pp': False,   
    },
    out_folder=out_folder,
    training_num_steps=N_STEPS,
    runner_additional_args_dir=os.path.abspath('./config.yaml'),
    training_process_args=dict(
        training_framework='colossal',
        mpi_implementation='pmi',
        hostnames=HOSTNAMES,
        seed=seed,
    ),
)

########
# DONT NEED TO CHANGE PARTS BELOW HERE
########

from oppa.optimise_parallelism import RandomSelection, KMeans, XGBoost, BayesianOptimisation, CostModelBased

if 'dkm52' in alg:
    kernel = 'dkm52'
elif 'dkm32' in alg:
    kernel = 'dkm32'
elif 'dkm12' in alg:
    kernel = 'dkm12'
elif 'dkr' in alg:
    kernel = 'dkr'
elif 'rbf' in alg:
    kernel = 'rbf'
elif 'm12' in alg:
    kernel = 'matern12'
elif 'm32' in alg:
    kernel = 'matern32'
elif 'm52' in alg:
    kernel = 'matern52'

early_terminate = 'et' in alg
acq_fn = 'ucb' if ('_ucb' in alg) else ('ei' if '_ei' in alg else None)

if alg == 'random':
    po = RandomSelection(**po_args)
    
elif alg == 'kmeans':
    po = KMeans(max_candidates_num=10*ROUNDS, **po_args)
    
elif alg == 'xgb':
    po = XGBoost(**po_args)

elif alg == 'cost':
    po = CostModelBased(**po_args)
    
elif alg.startswith('pm'):
    print(f'BO with Prior and {kernel=}')
    po = BayesianOptimisation(
        init_random_rounds=INIT_RAND_ROUNDS,
        gp_kernel=kernel,
        use_cost_informed_mean=True,
        early_terminate_bad_trials=early_terminate,
        early_terminate_crit_improvement_bound=CRIT_EARLY_TERMINATE_BOUND,
        early_terminate_score_improvement_bound=SCORE_EARLY_TERMINATE_BOUND,
        acq_fn=acq_fn,
        init_best=INIT_BEST,
        **po_args,
    )
    
elif alg.startswith('bo'):
    print(f'Pure BO with {kernel=}')
    po = BayesianOptimisation(
        init_random_rounds=INIT_RAND_ROUNDS,
        gp_kernel=kernel,
        use_cost_informed_mean=False,
        early_terminate_bad_trials=early_terminate,
        early_terminate_crit_improvement_bound=CRIT_EARLY_TERMINATE_BOUND,
        early_terminate_score_improvement_bound=SCORE_EARLY_TERMINATE_BOUND,        
        acq_fn=acq_fn,
        init_best=INIT_BEST,
        **po_args,
    )

else:
    raise ValueError

po.running_setup()
po.run(
    min_rounds=ROUNDS,
    max_rounds=200,
    max_time=TIME,
)

if isinstance(po, (BayesianOptimisation, CostModelBased, XGBoost)):
    import pickle as pkl
    with open(os.path.join(out_folder, 'learned_surrogate.pkl'), 'wb+') as f:
        pkl.dump(po.learned_surrogate, f)
