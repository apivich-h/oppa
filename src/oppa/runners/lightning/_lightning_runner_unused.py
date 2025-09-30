import os
import yaml
from pathlib import Path
import time
import datetime
import tqdm

def lightning_runner():
    
    start_time = time.time()
    
    from oppa.runners.parser import parse_args
    
    args = parse_args(impl='lightning')
    additional_args = yaml.safe_load(Path(args.model_additional_args_path).read_text())
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(args.num_gpus // args.num_hosts)])
    
    if 'hf_access_token_path' in additional_args['framework_args']:
        with open(additional_args['framework_args']['hf_access_token_path'], 'r') as f:
            hf_token = f.read().strip()
        os.environ['HF_TOKEN'] = hf_token
    
    
    import torch
    import torch.distributed
    from torch.utils.data import DataLoader, random_split
    import lightning as L
    import lightning.pytorch as pl
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    from datasets import load_dataset, load_from_disk
    from peft import get_peft_model, LoraConfig, TaskType
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    from lightning.pytorch.utilities import rank_zero_only
    from lightning.pytorch.utilities.types import STEP_OUTPUT
    from lightning.pytorch import Callback, Trainer
    from lightning.fabric.strategies import DeepSpeedStrategy, FSDPStrategy, ModelParallelStrategy
        
    from oppa.strategies.parallel_strategy import ParallelisationStrategy
    from oppa.runners.out_data import save_training_info
        
    assert torch.cuda.device_count() > 0
    
    para_strategy = ParallelisationStrategy.from_args(args)
    
    seq_length = additional_args['tokenizer_args'].get('max_length', None)
    global_batch_size = args.batch_size
    dp_batch_size = global_batch_size // (para_strategy.dp_size)
    do_finetuning = additional_args.get('do_finetuning', True)

    val_dset_path = additional_args.get('val_data_path', None)

    dset_path = additional_args.get('dset_path', None)
    dset_name = additional_args.get('dset_name', None)
    model_hf = additional_args['model_base']
    
    if do_finetuning:
        finetuning_params = additional_args['finetuning_params']
        if finetuning_params.get('do_lora', False):
            model_transform = LoraConfig(
                r=finetuning_params['lora_dim'],
                lora_alpha=finetuning_params['lora_dim'],  # check if alpha needs to be adjusted somehow
                lora_dropout=finetuning_params['lora_dropout'],
                layers_to_transform=sorted(finetuning_params['lora_layers']),
            )
        else:
            model_transform = None
    else:
        model_transform = None
        
    tokenizer = AutoTokenizer.from_pretrained(model_hf)
    if 'special_tokens' in additional_args['framework_args']:
        tokenizer.add_special_tokens(additional_args['framework_args']['special_tokens'])
    
    if dset_path is not None:
        train_dataset = load_from_disk(dset_path).select_columns(['input_ids', 'attention_mask', 'labels'])
        val_dataset = load_from_disk(val_dset_path).select_columns(['input_ids', 'attention_mask', 'labels'])
        # train_dataset.set_format("torch")
        # val_dataset.set_format("torch")
        
    elif dset_name is not None:
        
        raise NotImplementedError

        # def tokenize_function(examples):
        #     return tokenizer(
        #         examples["source"], 
        #         padding="max_length", 
        #         truncation=True,
        #         max_length=seq_length,
        #     )

        train_dataset = dataset['train'].map(tokenize_function, batched=False)
        if 'validation' in dataset:
            val_dataset = dataset['validation'].map(tokenize_function, batched=False)
        else:
            val_dataset = None
        
    else:
        raise ValueError(f"Unsupported combination of {dset_path=} and {dset_name=}")
    
    def _print_data_stats():
        print(f'{len(train_dataset)=} -- {len(val_dataset)=}')
        print(f'max seq length = {max(len(d["input_ids"]) for d in train_dataset)}')
    rank_zero_only(_print_data_stats)()
            
    ## initialize the strategy
    if True:
        
        # strategy = DeepSpeedStrategy(
        #     zero_optimization=(para_strategy.zero_stage > 0),
        #     stage=para_strategy.zero_stage,
        #     offload_optimizer=para_strategy.zero_offload,
        #     offload_parameters=para_strategy.zero_offload,
        #     overlap_comm=para_strategy.overlap_communication_for_zero,
        #     reduce_bucket_size=para_strategy.zero_bucket_size_mb,
        #     allgather_bucket_size=para_strategy.zero_bucket_size_mb,
        # )
        
        strategy = FSDPStrategy(
            sharding_strategy=["NO_SHARD", "SHARD_GRAD_OP", "HYBRID_SHARD", "FULL_SHARD"][para_strategy.zero_stage],
            cpu_offload=para_strategy.zero_offload, 
            device_mesh=(2, 4),
            # precision='bf16',
        )
        
    else:
        raise NotImplementedError
                
        
    # Custom collate_fn to handle padding for input_ids, attention_mask, and labels
    # from transformers import DataCollatorWithPadding
    # collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    def collate_fn(batch):
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        
        input_ids = [x["input_ids"] for x in batch]
        max_len = max(len(x) for x in input_ids)
        input_ids = [x + [pad_id] * (max_len - len(x)) for x in input_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        attention_mask = [x["attention_mask"] for x in batch]
        attention_mask = [x + [0] * (max_len - len(x)) for x in attention_mask]
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        labels = [x["labels"] for x in batch]
        labels = [x + [pad_id] * (max_len - len(x)) for x in labels]
        labels = torch.tensor(labels, dtype=torch.long)

        d = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        return d
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=dp_batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=(2 * para_strategy.num_gpus // para_strategy.num_hosts),
        collate_fn=collate_fn,
    )
    
    val_dataloader = (DataLoader(
        val_dataset, 
        batch_size=(additional_args.get('val_batch_size', 128) // para_strategy.num_gpus), 
        shuffle=False,
        pin_memory=True,
        num_workers=(2 * para_strategy.num_gpus // para_strategy.num_hosts),
        collate_fn=collate_fn,
    ) if val_dataset else None)
    
    do_validation = not (args.only_setup or args.only_benchmark)
    val_check_interval = max(min(1, (args.n_steps // 10)), len(train_dataloader))
    if do_validation:
        print(f'Performing validation and checkpointing every {val_check_interval} steps')
        def save_yaml_(y):
            with open(args.aux_out_path, 'w') as f:
                yaml.dump(y, f, sort_keys=False)
        rank_zero_only(save_yaml_)([])
        
    do_profiling = args.only_benchmark
    if do_profiling:
        rank_zero_only(save_training_info)(
            out_path=args.aux_out_path,
            para_strategy=para_strategy,
            ran_successfully=False,
            execution_time=(time.time() - start_time),
        )
        
    from lightning.fabric import Fabric
    fabric = Fabric(
        devices=para_strategy.num_gpus//para_strategy.num_hosts,
        num_nodes=para_strategy.num_hosts,
        accelerator="cuda", 
        strategy=strategy,
        precision='bf16-mixed',
    )
    fabric.launch()
    
    with fabric.init_module(empty_init=True):
        model = AutoModelForCausalLM.from_pretrained(
            model_hf, 
            # device_map='auto', 
            # ignore_mismatched_sizes=True,
        )
        if model_transform is not None:
            model = get_peft_model(model, model_transform)
            
    model = fabric.setup(model)
    model.resize_token_embeddings(len(tokenizer))
            
    optim_type = additional_args.get('optim_type', 'adamw')
    lr = additional_args.get('optim_lr', 6e-4)
    # if para_strategy.zero_offload:
    #     import deepspeed
    #     if optim_type in {'adam', 'adamw'}:
    #         use_adamw = optim_type == 'adamw'
    #         optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
    #             model.parameters(), 
    #             lr=lr,
    #             adamw_mode=use_adamw,
    #         )
    # else:
    if optim_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    optimizer = fabric.setup(optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    val_dataloader = (
        fabric.setup_dataloaders(val_dataloader) 
        if val_dataloader else None
    )
    
    n_epochs = args.n_epochs if (args.n_epochs is not None) else 1
    n_steps = args.n_steps if (args.n_steps is not None) else float('inf')
    
    global_step = 0
    train_time = 0.
    
    validation_results = []
    training_step_time = []
    
    torch.cuda.reset_max_memory_allocated()
    
    for epoch in range(n_epochs):
        
        step_start_time = time.time()
        model.train()
        train_iter = iter(train_dataloader)
        
        for s in tqdm.trange(len(train_dataloader), desc=f'Epoch {epoch}'):
            
            if do_validation and (global_step % val_check_interval == 0):
                # put this first so loss on step 0 is also recorded
                model.eval()
                validation_steps = 0
                validation_batch_loss = 0.
                with torch.no_grad():
                    for batch in tqdm.tqdm(val_dataloader, desc=f'Validation for step {global_step}'):
                        output = model(**batch)
                        validation_batch_loss += output.loss.item()
                        validation_steps += 1
                mean_validation = validation_batch_loss / validation_steps
                print(f'Training step {global_step} validation loss = {mean_validation}')
                validation_results.append({
                    'train_steps': global_step,
                    'train_time': train_time,
                    'val_loss': mean_validation,
                })
                rank_zero_only(save_yaml_)(validation_results)
                model.train()
                step_start_time = time.time()
            
            t0 = time.time()
            
            batch = next(train_iter)
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            fabric.backward(loss)
            optimizer.step()
            global_step += 1
            
            dt = time.time() - t0
            train_time += dt           
            if do_profiling:
                training_step_time.append(dt)
                
            if global_step > n_steps:
                break
            
        if global_step > n_steps:
            break
        
    if do_profiling:
        
        max_use_GB = torch.cuda.max_memory_allocated() / 1024**3
        all_mem_vals = [None for _ in range(fabric.world_size)]
        torch.distributed.gather_object(
            max_use_GB,
            object_gather_list=(all_mem_vals if (fabric.global_rank == 0) else None),
            dst=0,
        )
        max_mem_all_machines = all_mem_vals
        
        rank_zero_only(save_training_info)(
            out_path=args.aux_out_path,
            para_strategy=para_strategy,
            ran_successfully=True,
            execution_time=(time.time() - start_time),
            time_results=training_step_time,
            memory_results=max_mem_all_machines,
        )
        
if __name__ == '__main__':
    lightning_runner()
