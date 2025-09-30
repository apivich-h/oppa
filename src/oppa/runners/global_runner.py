import subprocess
import os
import sys
import json
import random
import shlex
import time

from ..strategies.parallel_strategy import ParallelisationStrategy, check_implementation_specific_strategy

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def run_training_subprocess(
    parallelisation_strategy: ParallelisationStrategy,
    training_framework='colossal',
    batch_size=1,
    n_epochs=None,
    n_steps=None,
    model_additional_args_path=None,
    model_out='../results/model_out',
    aux_out='../results/aux.yaml',
    save_step_timing=False,
    hostnames=None,
    master_port=29400,
    mpi_implementation='pmi',
    mpi_env_variables=dict(),
    seed=42,
    capture_outputs=False,
    verbose=False,
    always_rerun=True,
    only_setup=False,
    only_benchmark=False,
    only_short_training=False,
):
    assert n_epochs is not None or n_steps is not None
    assert n_epochs is None or n_steps is None
    assert int(only_setup) + int(only_benchmark) + int(only_short_training) <= 1
    check_implementation_specific_strategy(parallelisation_strategy, implementation=training_framework)
    
    os.makedirs(model_out, exist_ok=True)
    aux_out_path = aux_out
    
    reran = False

    if always_rerun or not os.path.exists(aux_out_path):

        reran = True
    
        """ Set up base command for all frameworks """
    
        base_command = [
            "--batch-size", str(batch_size),
            "--model-out-path", model_out,
            "--aux-out-path", aux_out_path,
            "--seed", str(seed)
        ]
        if n_epochs is not None:
            base_command += ["--n-epochs", str(n_epochs)]
        else:
            base_command += ["--n-steps", str(n_steps)]
        base_command += parallelisation_strategy.to_args()
        
        if save_step_timing:
            base_command.append("--save-step-timing")
            
        if model_additional_args_path:
            base_command.extend(["--model-additional-args-path", model_additional_args_path])
            
        if only_setup:
            base_command.append("--only-setup") 
        elif only_benchmark:
            base_command.append("--only-benchmark")
        elif only_short_training:
            base_command.append("--only-short-training")
        
        """ Get command specific to framework we're testing on """
        
        if training_framework == 'mock':
            commands = [['python', os.path.join(DIR_PATH, 'mock', 'mock_runner.py')] + base_command]
            
        elif training_framework == 'nemo':
            
            gpus_per_host = parallelisation_strategy.num_gpus // parallelisation_strategy.num_hosts

            if parallelisation_strategy.num_hosts == 1:
                commands = [['python', os.path.join(DIR_PATH, 'nemo', 'nemo_runner.py')] + base_command]
            
            else:
                raise NotImplementedError
            
        elif training_framework == 'lightning':
            
            gpus_per_host = parallelisation_strategy.num_gpus // parallelisation_strategy.num_hosts
            
            if parallelisation_strategy.num_hosts == 1:
                commands = [[
                    # 'torchrun', 
                    # '--nproc_per_node', str(gpus_per_host),
                    # "--master-port", str(master_port),
                    'python',
                    os.path.join(DIR_PATH, 'lightning', 'lightning_runner.py')
                ] + base_command]
            
            else:
                raise NotImplementedError
            
        elif training_framework == 'hf':
            base_command.extend([
                '--master-port', str(master_port),
            ])
            commands = [[
                'python',
                os.path.join(DIR_PATH, 'hf', 'hf_runner.py')
            ] + base_command]
        
        elif training_framework == 'colossal':
            
            gpus_per_host = parallelisation_strategy.num_gpus // parallelisation_strategy.num_hosts

            if parallelisation_strategy.num_hosts > 1:
                assert hostnames is not None
                assert master_port is not None
                base_command.extend([
                    "--master-addr", hostnames[0],
                    "--master-port", str(master_port) 
                ])

                if mpi_implementation == 'pmi':
                    colossal_command = [
                        'mpirun',
                        '--hosts', ','.join(hostnames),
                        '--ppn', str(gpus_per_host),
                        '--np', str(parallelisation_strategy.num_gpus),
                        '-v',
                        '--abort-on-failure',
                    ]
                    for k, v in mpi_env_variables.items():
                        colossal_command.extend(['--env', f'{k}={v}'])

                elif mpi_implementation == 'ompi':
                    colossal_command = [
                        'mpirun',
                        '--host', ','.join([f'{h}:{gpus_per_host}' for h in hostnames]),
                        '-np', str(parallelisation_strategy.num_gpus),
                        '--abort-on-failure',
                    ]
                    for k, v in mpi_env_variables.items():
                        colossal_command.extend(['-x', f'{k}={v}'])
                
                colossal_command.extend([sys.executable, os.path.join(DIR_PATH, 'colossal', 'colossal_runner.py')])
                base_command.extend(['--mpi-implementation', mpi_implementation])
                commands = [colossal_command + base_command]
                
            else:
                assert master_port is not None
                commands = [[
                    'torchrun',
                    '--nproc_per_node', str(gpus_per_host),
                    "--master-port", str(master_port),
                    os.path.join(DIR_PATH, 'colossal', 'colossal_runner.py'), 
                    '--mpi-implementation', 'ompi',
                    "--master-port", str(master_port),
                ] + base_command]
            
        else:
            raise ValueError(f'Invalid {training_framework = }')
        
        """ Do some actual running """
        
        assert not (verbose and capture_outputs)
        
        try:

            for cmd in commands:
                print('Running command:', shlex.join(cmd))
                process = subprocess.Popen(cmd)
                process.wait()
                
        except subprocess.CalledProcessError as e:
            print("An error occurred while running the subprocess.")

    if parallelisation_strategy.num_hosts > 1:
        time.sleep(5)
    return model_out, aux_out_path, reran
