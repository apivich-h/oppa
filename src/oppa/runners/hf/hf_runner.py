import sys, subprocess, shlex, os

from oppa.runners.parser import parse_args
from oppa.strategies.parallel_strategy import ParallelisationStrategy

def main():

    arg_options = sys.argv[1:]
    args = parse_args(impl='hf')
    para_strategy = ParallelisationStrategy.from_args(args)

    if para_strategy.num_hosts == 1:

        cmd_train = (
            [
                'accelerate', 'launch',
                '--num_processes', str(para_strategy.num_gpus),
                '--num_machines', str(para_strategy.num_hosts),
                '--mixed_precision', 'bf16',
                '--dynamo_backend', 'no',
                '--main_process_port', str(args.master_port),
            ] + 
            [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hf_train.py')] + 
            arg_options
        )
        print('================== DO TRAINING ==================')
        print('Train launch command =', shlex.join(cmd_train))
        process = subprocess.Popen(cmd_train)
        process.wait()

        if not (args.only_setup or args.only_benchmark or args.only_short_training):
            cmd_eval = (
                [
                    'accelerate', 'launch',
                    '--num_processes','1',
                    '--num_machines', '1',
                    # '--num_processes', str(para_strategy.num_gpus),
                    # '--num_machines', str(para_strategy.num_hosts),
                    '--mixed_precision', 'bf16',
                    '--dynamo_backend', 'no',
                    # '--main_process_port', '0',
                ] + 
                [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hf_eval.py')] + 
                arg_options
            )
            print('================== DO EVAL ==================')
            print('Eval launch command =', shlex.join(cmd_eval))
            process = subprocess.Popen(cmd_eval)
            process.wait()

    else:
        raise NotImplementedError('Only can have num_hosts=1')


if __name__ == '__main__':
    main()

