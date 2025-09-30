import argparse


def parse_args(impl):
    
    parser = argparse.ArgumentParser(description="Parse distributed computing arguments.")
    
    parser.add_argument('--model', help='Model')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--n-epochs', type=int, default=None, help='Training epochs')
    parser.add_argument('--n-steps', type=int, default=None, help='Training steps')
    parser.add_argument('--only-setup', action='store_true', default=False, help='Only do model setup (e.g., downloading in cache first)')
    parser.add_argument('--only-benchmark', action='store_true', default=False, help='Only do model benchmarking (for training step times)')
    parser.add_argument('--only-short-training', action='store_true', default=False, help='Only do shorter model training')

    parser.add_argument('--model-additional-args-path', type=str, default=None, help='Additional model arguments parsed as JSON')
    parser.add_argument('--model-out-path', type=str, help='Location to store the trained model checkpoints')
    parser.add_argument('--aux-out-path', type=str, help='Location to store the training auxillary information')
    parser.add_argument('--save-step-timing', action='store_true', default=False, help='Whether to save the fine-grained timing for each train step')
    
    parser.add_argument('--dp-size', type=int, default=1, help='Data parallelism size')
    parser.add_argument('--tp-size', type=int, default=1, help='Tensor parallelism size')
    parser.add_argument('--pp-size', type=int, default=1, help='Pipeline parallelism size')
    parser.add_argument('--sp-size', type=int, default=1, help='Sequence parallelism size')
    parser.add_argument('--dp-bucket-size-mb', type=float, default=1., help='DP bucket sizes')
    parser.add_argument('--zero-stage', type=int, default=1, help='ZeRO stage')
    parser.add_argument('--zero-bucket-size-mb', type=float, default=1., help='ZeRO all-scatter and reduce-gather bucket sizes')
    parser.add_argument('--num-microbatches', type=int, default=1, help='Number of microbatches')
    parser.add_argument('--num-model-chunks', type=int, default=1, help='Number of model chunks')
    parser.add_argument('--use-grad-checkpoint', type=int, default=1, help='Use gradient checkpointing or not')
    parser.add_argument('--overlap-communication-for-zero', type=int, default=0)
    parser.add_argument('--overlap-allgather-for-zero', type=int, default=0)
    parser.add_argument('--overlap-p2p-for-pp', type=int, default=0)
    parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--num-hosts', type=int, default=1, help='Number of hosts to use')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    parser.add_argument('--master-addr', type=str, default=None, help='Master IP address')
    parser.add_argument('--master-port', type=int, default=None, help='Master port')

    if impl == 'colossal':
        parser.add_argument('--mpi-implementation', type=str, default='pmi', help='MPI implementation used')
    
    return parser.parse_args()