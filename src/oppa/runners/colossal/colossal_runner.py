import os
import yaml
from pathlib import Path
import time


def colossal_runner():

    start_time = time.time()

    from oppa.runners.parser import parse_args
    args = parse_args(impl='colossal')
    additional_args = yaml.safe_load(Path(args.model_additional_args_path).read_text())
    model = additional_args.pop('model')
    
    if 'hf_home' in additional_args.keys():
        os.environ['HF_HOME'] = additional_args.pop('hf_home')
        print(f'{os.environ["HF_HOME"]=}')
        os.makedirs(os.environ['HF_HOME'], exist_ok=True)

    assert args.n_epochs is not None or args.n_steps is not None
    assert args.n_epochs is None or args.n_steps is None
    assert args.only_setup or args.only_benchmark

    # currently using this hack to make sure each node loads NCCL properly, and not try to load GLOO
    # TODO: make this part more proper
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(args.num_gpus)])
    if 'LOCAL_WORLD_SIZE' not in os.environ.keys():
        os.environ['LOCAL_WORLD_SIZE'] = str(args.num_gpus)
        print(f'Setting {os.environ["LOCAL_WORLD_SIZE"]=}')
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    # os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'

    import torch
    import torch.distributed as dist
    import colossalai
    from colossalai.accelerator import get_accelerator
    from colossalai.utils import set_seed
    from colossalai.cluster import DistCoordinator

    from oppa.runners.colossal.parallel_helper import set_up_parallelism
    from oppa.runners.colossal.training_helper import train_epochs
    from oppa.runners.colossal.model_setup import (
        generate_huggingface_classification_model,
        generate_huggingface_generation_model,
        generate_random_huggingface_generation_model
    )
    from oppa.strategies.parallel_strategy import ParallelisationStrategy
    from oppa.runners.out_data import save_training_info
    from oppa.utils.normdist import normal_posterior, expected_improvement, upper_confidence_bound_improvement
    
    accelerator = get_accelerator()
    
    if args.num_hosts > 1:
    
        if args.mpi_implementation == 'ompi':
            print('Using OMPI env variables')
            rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
            local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
            world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])

        elif args.mpi_implementation == 'pmi':
            print('Using PMI env variables')
            rank = int(os.environ["PMI_RANK"])
            local_rank = int(os.environ["PMI_LOCAL_RANK"])
            world_size = int(os.environ["PMI_SIZE"])

        else:
            raise ValueError('Invalid MPI implementation')

        print(f'Launching on {rank=}, {local_rank=}, {world_size=}')

        colossalai.launch(
            local_rank=local_rank,
            rank=rank,
            world_size=world_size,
            host=args.master_addr,
            port=args.master_port,
            backend='nccl',
            seed=args.seed,
            verbose=True,
        )
        print(f'Finished launch on {rank=}, {local_rank=}, {world_size=}')
        
    else:
        
        colossalai.launch_from_torch(
            backend='nccl',
            seed=args.seed,
            verbose=True,
        )
    
    coordinator = DistCoordinator()
    
    # if 'hf_access_token_path' in additional_args.keys():
    #     with open(os.path.join(
    #         os.path.dirname(args.model_additional_args_path), additional_args['hf_access_token_path']), 
    #         'r') as f:
    #         hf_token = f.read().strip()
    #     additional_args.pop('hf_access_token_path') 
    # else:
    hf_token = ''
    
    booster_args = additional_args.pop('booster_args', dict())
    if 'use_flash_attn' in additional_args.keys():
        booster_args['use_flash_attention'] = additional_args['use_flash_attn']
    if coordinator.is_master():
        print(f'{booster_args=}') 
    para_strategy = ParallelisationStrategy.from_args(args)
    booster = set_up_parallelism(para_strategy=para_strategy, **booster_args)
    
    ran_successfully = False
    if coordinator.is_master():
        print(f'{additional_args=}')
        
    batch_size = args.batch_size // para_strategy.dp_size
    lr = (additional_args.pop('lr', 3e-4)) / para_strategy.dp_size
    
    early_terminate_args = additional_args.pop('early_terminate', {'do_early_terminate': False})
    
    def load_model(only_do_predownload):
        if model == 'hf_class':
            components = generate_huggingface_classification_model(
                para_strategy=para_strategy,
                booster=booster, 
                batch_size=batch_size, 
                lr=lr,
                use_grad_checkpoint=para_strategy.use_grad_checkpoint,
                hf_access_token=hf_token,
                only_do_predownload=only_do_predownload,
                **additional_args
            )
        elif model == 'hf_gen':
            components = generate_huggingface_generation_model(
                para_strategy=para_strategy,
                booster=booster, 
                batch_size=batch_size, 
                lr=lr,
                use_grad_checkpoint=para_strategy.use_grad_checkpoint,
                hf_access_token=hf_token,
                only_do_predownload=only_do_predownload,
                **additional_args
            )
        elif model == 'hf_gen_rand':
            components = generate_random_huggingface_generation_model(
                para_strategy=para_strategy,
                booster=booster, 
                batch_size=batch_size, 
                lr=lr,
                use_grad_checkpoint=para_strategy.use_grad_checkpoint,
                hf_access_token=hf_token,
                only_do_predownload=only_do_predownload,
                **additional_args
            )
        else:
            raise ValueError(f'Invalid model -- {model}')
        return components
    
    if args.only_setup:
        load_model(only_do_predownload=True)
        return
    
    if coordinator.is_master():
            
        finish_time = time.time()
        
        # data_to_store = {
        #     'input_hyperparams': para_strategy.to_dict(),
        #     'ran_successfully': False,
        #     'execution_time': finish_time - start_time,
        #     'time_results': None,
        #     'memory_results': {
        #         'peak_mem': float('inf'),
        #         'peak_mem_per_gpu': [float('inf') for _ in range(coordinator.world_size)],
        #     }
        # }
        
        # with open(args.aux_out_path, 'w') as f:
        #     yaml.dump(data_to_store, f, sort_keys=False)
            
        save_training_info(
            out_path=args.aux_out_path,
            para_strategy=para_strategy,
            ran_successfully=False,
            execution_time=(finish_time - start_time),
        )

    try:
    
        components = load_model(only_do_predownload=False)
        
        
        # if args.load_checkpoint is not None:
        #     if "modeling" in args.load_checkpoint:
        #         coordinator.print_on_master(f"Continued pretrain from checkpoint {args.load_checkpoint}")
        #         booster.load_model(model, args.load_checkpoint)
        #     else:
        #         coordinator.print_on_master(f"Load model checkpoint from {args.load_checkpoint}")
        #         start_epoch, start_step, sampler_start_idx = load_checkpoint(
        #             load_dir=args.load_checkpoint,
        #             booster=booster,
        #             model=model,
        #             optimizer=optimizer,
        #             lr_scheduler=lr_scheduler,
        #         )
        #         coordinator.print_on_master(
        #             f"Loaded checkpoint {args.load_checkpoint} at epoch {start_epoch} step {start_step}"
        #         )
        #         coordinator.print_on_master(f"Loaded sample at index {sampler_start_idx}")
        
        t_start = time.time()
        time_per_step = []
        for t in train_epochs(
            n_epochs=args.n_epochs,
            n_steps=args.n_steps,
            model=components['model'],
            optimizer=components['optimiser'],
            criterion=components['criterion'],
            lr_scheduler=components['lr_scheduler'],
            train_dataloader=components['train_dataloader'],
            accelerator=accelerator,
            booster=booster,
            coordinator=coordinator,
            store_time_per_step=args.save_step_timing,
        ):
            
            time_per_step.append(t)
            other_results = {
                "steps_ran": len(time_per_step),
                "overall_time": time.time() - t_start,
            }
            
            finish_time = time.time()
        
            if coordinator.is_master():
                
                d = save_training_info(
                    out_path=args.aux_out_path,
                    para_strategy=para_strategy,
                    ran_successfully=False,
                    execution_time=(finish_time - start_time),
                    time_per_step=time_per_step,
                    other_results=other_results,
                    do_save=(len(time_per_step) % 5 == 0),
                )
                
                if early_terminate_args['do_early_terminate']:
                    et_method = early_terminate_args['method']
                    curr_best = early_terminate_args['curr_best']
                    prior_mean = early_terminate_args['prior_mean']
                    prior_var = early_terminate_args['prior_var']
                    crit_threshold = early_terminate_args['crit_threshold']
                    mean_threshold = early_terminate_args['mean_threshold']
                    assert early_terminate_args['use_throughput']
                    llh_mean = d['time_results']['throughput_mean']
                    llh_var = d['time_results']['throughput_std'] ** 2
                    n_inlier = d['time_results']['n_inlier']
                    posterior_mean, posterior_var = normal_posterior(
                        prior_mean=prior_mean,
                        prior_var=prior_var,
                        llh_mean=llh_mean,
                        llh_var=llh_var,
                    )
                    if et_method == 'ei':
                        s_posterior = expected_improvement(
                            mean=posterior_mean,
                            std=posterior_var**0.5,
                            best=curr_best,
                        )
                        s_llh = expected_improvement(
                            mean=llh_mean,
                            std=llh_var**0.5,
                            best=curr_best,
                        )
                    elif et_method == 'ucb':
                        s_posterior = upper_confidence_bound_improvement(
                            mean=posterior_mean,
                            std=posterior_var**0.5,
                            best=curr_best,
                            beta=early_terminate_args.get('ucb_beta', 0.01),
                        )
                        s_llh = upper_confidence_bound_improvement(
                            mean=llh_mean,
                            std=llh_var**0.5,
                            best=curr_best,
                            beta=early_terminate_args.get('ucb_beta', 0.01),
                        )
                    else:
                        raise ValueError(f'Invalid {et_method=}')
                    terminate = (
                        (s_posterior < crit_threshold) and 
                        (s_llh < crit_threshold) and 
                        (llh_mean - curr_best < mean_threshold) and  # make sure that the mean is not too below best
                        (n_inlier >= early_terminate_args.get('n_inlier', 5))
                    )
                else:
                    terminate = False
                    
            else:
                terminate = None
                
            dist.barrier()
            flag = [None]
            dist.scatter_object_list(
                scatter_object_output_list=flag,
                scatter_object_input_list=[terminate for _ in range(coordinator.world_size)],
                src=0,
            )
            terminate = flag[0]
            if terminate:
                if coordinator.is_master():
                    print(f'{llh_mean=:.5e} {llh_var=:.5e} {posterior_mean=:.5e} {posterior_var=:.5e} {curr_best=:.5e}')
                    print(f'{s_llh=:.5e} {s_posterior=:.5e} {crit_threshold=:.3e} {mean_threshold=:.3e} -- terminating trial...')
                break  
        
        max_use_GB = accelerator.max_memory_allocated(accelerator.get_current_device()) / 1024**3
        all_mem_vals = [None for _ in range(coordinator.world_size)]
        dist.gather_object(
            max_use_GB,
            object_gather_list=(all_mem_vals if coordinator.is_master() else None),
            dst=0,
        )
        
        if coordinator.is_master():
            
            finish_time = time.time()
            
            save_training_info(
                out_path=args.aux_out_path,
                para_strategy=para_strategy,
                ran_successfully=True,
                execution_time=(finish_time - start_time),
                time_per_step=time_per_step,
                memory_results=all_mem_vals,
                other_results=other_results
            )
            
            
        
    except torch.OutOfMemoryError as e:
    # except Exception as e:
        
        # time_results = None
        # ran_successfully = False
        # all_mem_vals = [float('inf') for _ in range(coordinator.world_size)]
        print(f'torch.OutOfMemoryError reached in rank {coordinator.rank}')
        raise e
    
    except Exception as e:
        
        os.remove(args.aux_out_path)
        raise e
        
    # finally:
        
        # if ran_successfully:
        #     save_checkpoint(
        #         save_dir=args.save_dir,
        #         booster=booster,
        #         model=model,
        #         optimizer=optimizer,
        #         lr_scheduler=lr_scheduler,
        #         epoch=epoch,
        #         step=step + 1,
        #         batch_size=args.batch_size,
        #         coordinator=coordinator,
        #     )
        
        
                    

if __name__ == '__main__':
    colossal_runner()
