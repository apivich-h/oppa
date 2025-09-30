import os
import yaml
from pathlib import Path
import time

import torch
import lightning.pytorch as pl
from nemo import lightning as nl
from nemo.collections import llm
import nemo_run as run
from nemo.collections import nlp as nemo_nlp
from megatron.core.optimizer import OptimizerConfig
from datasets import load_dataset
from lightning.pytorch.utilities import rank_zero_only

from ...parallel_strategy import ParallelisationStrategy
from ..out_data import save_training_info
from ..parser import parse_args
from .helpers.squad_v2 import make_squad_v2_hf_dataset
from .profiling_callback import PerformanceCallback


def nemo_runner():
    
    start_time = time.time()
    
    args = parse_args(impl='nemo')
    additional_args = yaml.safe_load(Path(args.model_additional_args_path).read_text())
    para_strategy = ParallelisationStrategy.from_args(args)
    
    seq_length = additional_args.get('seq_length', 2048)
    global_batch_size = args.batch_size
    micro_batch_size = global_batch_size // (para_strategy.num_microbatches * para_strategy.dp_size)
    do_finetuning = additional_args.get('do_finetuning', True)

    dset_name = additional_args.get('dataset_name', None)
    dset_path = additional_args.get('dataset_path', None)
    
    
    if do_finetuning:
    
        ## do LoRA
        if additional_args.get('do_lora', False):
            model_transform = llm.peft.LoRA(
                dim=additional_args['lora_dim'],
                alpha=additional_args['lora_dim'],  # check if alpha needs to be adjusted somehow
                dropout=additional_args['lora_dropout']
            )
        else:
            model_transform = None
            
    else:
        model_transform = None

    model_base = additional_args['model_base']
    
    if model_base == 'llama2-7b':
        model_hf = 'meta-llama/Llama-2-7b-hf'
        model = llm.LlamaModel(
            config=llm.Llama2Config7B(),
            tokenizer=nemo_nlp.modules.get_tokenizer(tokenizer_name=model_hf),
        )
    elif model_base == 't5-base':
        model_hf = 'google-t5/t5-base'
        model = llm.T5Model(
            config=llm.t5_220m.T5Config220M(),
            tokenizer=nemo_nlp.modules.get_tokenizer(tokenizer_name=model_hf),
        )
    else:
        raise ValueError(f'Invalid {model_base=}')
    
    if do_finetuning:
        resume = nl.AutoResume(resume_from_directory=f'hf://{model_hf}')
    else:
        resume = None
        
    if dset_name == 'dummy':
        data = llm.MockDataModule(
            seq_length=seq_length, 
            global_batch_size=global_batch_size, 
            micro_batch_size=micro_batch_size,
        )
    elif dset_name == 'squad-v2':
        data = make_squad_v2_hf_dataset(
            tokenizer=model.tokenizer, 
            seq_length=seq_length, 
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size, 
            seed=args.seed,
        )
    else:
        raise NotImplementedError
    
    ## initialize the strategy
    if para_strategy.zero_stage > 0:
        from lightning.pytorch.plugins import FSDPPrecision
        # there are no ZeRO implementation so just use FSDP
        # with the different sharding strategies for each ZeRO stage
        sharding_strategy_list = ["NO_SHARD", "SHARD_GRAD_OP", "HYBRID_SHARD", "FULL_SHARD"]
        strategy = nl.FSDPStrategy(
            sharding_strategy=sharding_strategy_list[para_strategy.zero_stage],
            device_mesh=(para_strategy.dp_size, para_strategy.tp_size),
        )
        precision_plugin = FSDPPrecision(
            precision="bf16-mixed",
        )
        optim = torch.optim.Adam(model.parameters(), lr=additional_args.get('lr', 6e-4))
    else:
        mc = para_strategy.num_model_chunks
        strategy = nl.MegatronStrategy(
            overlap_p2p_comm=para_strategy.overlap_p2p_for_pp,
            batch_p2p_comm=(not para_strategy.overlap_p2p_for_pp),
            tensor_model_parallel_size=para_strategy.tp_size,
            pipeline_model_parallel_size=para_strategy.pp_size,
            virtual_pipeline_model_parallel_size=(mc if (mc > 1) else None),
            # microbatch_group_size_per_vp_stage=micro_batch_size,
            context_parallel_size=para_strategy.sp_size,
            pipeline_dtype=torch.bfloat16,
            gradient_accumulation_fusion=False,
            use_distributed_optimizer=True,
        )
        precision_plugin = nl.MegatronMixedPrecision(
            precision="bf16-mixed",
        )
        ## setup the optimizer
        optim = nl.MegatronOptimizerModule(
            config=OptimizerConfig(
                optimizer='adam',
                lr=additional_args.get('lr', 6e-4),
                bf16=True,
                # use_distributed_optimizer=True,
                # para strategy items
                # bucket_cap_mb=para_strategy.dp_bucket_size_mb,
                # overlap_param_gather_with_optimizer_step=para_strategy.overlap_allgather_for_zero,
            )
        )
        
    callbacks = []
    if args.only_setup:
        print('Only running setup')
        nemo_logger = None
        other_args = dict(
            enable_checkpointing=False,
            limit_val_batches=0,
            limit_test_batches=0,
        )
    elif args.only_benchmark:
        print('Only running benchmarking of train step times')
        nemo_logger = None
        other_args = dict(
            enable_checkpointing=False,
            limit_val_batches=0,
            limit_test_batches=0,
        )
        profiler = PerformanceCallback()
        callbacks.append(profiler)
    else:
        print('Performing training with validation')
        nemo_logger = nl.NeMoLogger(
            log_dir=os.path.join(args.model_out_path, 'logs')
        )
        other_args = dict(
            enable_checkpointing=True,
        )
        
    if args.only_setup:
        trainer = nl.Trainer(
            devices=para_strategy.num_gpus//para_strategy.num_hosts,
            num_nodes=para_strategy.num_hosts,
            max_steps=1,
            max_epochs=(None if args.n_epochs is None else args.n_epochs),
            accelerator="gpu",
            strategy=strategy,
            plugins=precision_plugin,
            callbacks=callbacks,
            **other_args,
        )
    else:
        trainer = nl.Trainer(
            devices=para_strategy.num_gpus//para_strategy.num_hosts,
            num_nodes=para_strategy.num_hosts,
            max_steps=(-1 if args.n_steps is None else args.n_steps),
            max_epochs=(None if args.n_epochs is None else args.n_epochs),
            accelerator="gpu",
            strategy=strategy,
            plugins=precision_plugin,
            callbacks=callbacks,
            **other_args,
        )
                
    if args.only_benchmark:
        rank_zero_only(save_training_info)(
            out_path=args.aux_out_path,
            para_strategy=para_strategy,
            ran_successfully=False,
            execution_time=(time.time() - start_time),
        )
    
    if do_finetuning:
        llm.finetune(
            model=model,
            data=data,
            trainer=trainer,
            log=nemo_logger,
            optim=optim,
            resume=resume,
            peft=model_transform,
        )
    else:
        llm.train(
            model=model,
            data=data,
            trainer=trainer,
            log=nemo_logger,
            optim=optim,
        )
        
    if args.only_benchmark:
        benchmark_results = profiler.get_training_stats()
        if benchmark_results is not None:
            save_training_info(
                out_path=args.aux_out_path,
                para_strategy=para_strategy,
                ran_successfully=True,
                execution_time=(time.time() - start_time),
                time_results=benchmark_results['time'],
                memory_results=benchmark_results['mem'],
            )
        
if __name__ == '__main__':
    nemo_runner()