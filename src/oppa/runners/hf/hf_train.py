import os
import yaml
from pathlib import Path
import time
import datetime
import tqdm
import math

def hf_train():
    
    start_time = time.time()
    
    from oppa.runners.parser import parse_args
    
    args = parse_args(impl='lightning')
    additional_args = yaml.safe_load(Path(args.model_additional_args_path).read_text())
    
    if not os.environ.get('CUDA_VISIBLE_DEVICES', None):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(args.num_gpus // args.num_hosts)])
    else:
        print(f"{os.environ['CUDA_VISIBLE_DEVICES']=}")

    if 'hf_access_token_path' in additional_args:
        with open(additional_args['hf_access_token_path'], 'r') as f:
            hf_token = f.read().strip()
        os.environ['HF_TOKEN'] = hf_token
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    import numpy as np
    import torch
    import torch.distributed
    from torch.utils.data import DataLoader, random_split
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    from datasets import load_dataset, load_from_disk, disable_caching
    from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    from accelerate import Accelerator, init_empty_weights, DeepSpeedPlugin
    from accelerate.utils import broadcast
        
    from oppa.strategies.parallel_strategy import ParallelisationStrategy
    from oppa.runners.out_data import save_training_info
    from oppa.utils.dict_utils import map_nested_dicts
        
    assert torch.cuda.device_count() > 0
    disable_caching()

    para_strategy = ParallelisationStrategy.from_args(args)
    
    seq_length = additional_args['tokenizer_args'].get('max_length', None)
    global_batch_size = args.batch_size
    dp_batch_size = global_batch_size // (para_strategy.dp_size)
    do_finetuning = additional_args.get('do_finetuning', True)

    val_dset_path = additional_args.get('val_data_path', None)

    dset_path = additional_args.get('dset_path', None)
    dset_name = additional_args.get('dset_name', None)
    model_hf = additional_args['model_base']
    
    ## initialize the strategy
    if para_strategy.zero_stage > 0:
        assert para_strategy.zero_stage < 3  # because of saving params error
        import deepspeed
        strat_type = 'deepspeed'
        # offload_device = 'cpu' if para_strategy.zero_offload else None
        offload_device = 'cpu'
        acc_kwargs = {
            'deepspeed_plugin': DeepSpeedPlugin(
                zero_stage=para_strategy.zero_stage,
                offload_optimizer_device=offload_device,
                offload_param_device=offload_device,
                # overlap_comm=para_strategy.overlap_communication_for_zero,
                # reduce_bucket_size=para_strategy.zero_bucket_size_mb,
                # allgather_bucket_size=para_strategy.zero_bucket_size_mb,
            )
        }
        
    else:
        raise NotImplementedError
    
    accelerator = Accelerator(
        mixed_precision='bf16',
        **acc_kwargs,
    )

    if accelerator.is_main_process:
        model_save_path = os.path.join(args.model_out_path, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(model_save_path, exist_ok=True)
    
    if not accelerator.is_main_process:
        import logging
        logging.disable(logging.WARN);
    
    if do_finetuning:
        
        finetuning_params = additional_args['finetuning_params']
        
        if finetuning_params.get('do_lora', False):
            target = []
            if finetuning_params['lora_add_to_kqvo']:
                target.extend(['q_proj', 'v_proj', 'k_proj', 'o_proj'])
            if finetuning_params['lora_add_to_mlp']:
                target.extend(['down_proj', 'up_proj'])
            assert len(target) > 0
            model_transform = LoraConfig(
                r=finetuning_params['lora_dim'],
                lora_alpha=(finetuning_params['lora_alpha_div_dim'] * finetuning_params['lora_dim']),
                lora_dropout=finetuning_params['lora_dropout'],
                target_modules=target,
                layers_to_transform=sorted(finetuning_params['lora_layers']),
            )
            
        else:
            raise ValueError('Only allow PEFT for now')
    else:
        raise ValueError('Only do_finetuning == True possible for now')
        
    tokenizer = AutoTokenizer.from_pretrained(model_hf)
    if 'special_tokens' in additional_args:
        print(f"Add extra tokens: {additional_args['special_tokens']}")
        tokenizer.add_special_tokens(additional_args['special_tokens'])
    
    if dset_path is not None:
        train_dataset = load_from_disk(dset_path).select_columns(['input_ids', 'attention_mask', 'labels'])
        val_dataset = load_from_disk(val_dset_path).select_columns(['input_ids', 'attention_mask', 'labels'])
        # train_dataset.set_format("torch")
        # val_dataset.set_format("torch")
        
    elif dset_name is not None:
        raise NotImplementedError
        
    else:
        raise ValueError(f"Unsupported combination of {dset_path=} and {dset_name=}")
    
    if accelerator.is_main_process:
        accelerator.print(f'{len(train_dataset)=} -- {len(val_dataset)=}')
        accelerator.print(f'max seq length = {max(len(d["input_ids"]) for d in train_dataset)}') 
        accelerator.print(f'Will save checkpoints to {model_save_path}')       
        
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
    
    val_dp_batch_size = additional_args.get('val_batch_size', 128) // para_strategy.num_gpus
    val_dataloader = (DataLoader(
        val_dataset, 
        batch_size=val_dp_batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=(2 * para_strategy.num_gpus // para_strategy.num_hosts),
        collate_fn=collate_fn,
    ) if val_dataset else None)
    
    
    model = AutoModelForCausalLM.from_pretrained(model_hf)
    model.resize_token_embeddings(len(tokenizer))
    if model_transform is not None:
        model = get_peft_model(model, model_transform)

    def count_trainable_parameters(model):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters])
        return int(num_params)
    
    num_trainable_params = count_trainable_parameters(model)
    accelerator.print(f'Number of trainable params = {num_trainable_params}')
    # accelerator.print('Trainable params include:')
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         accelerator.print(f'   {name}')
            
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
        
    train_dataloader, model, optimizer = accelerator.prepare(train_dataloader, model, optimizer)
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)
        
    do_validation = not (args.only_setup or args.only_benchmark)
    do_test_loss = do_validation and not args.only_short_training
        
    n_steps = args.n_steps if (args.n_steps is not None) else float('inf')
    n_epochs = args.n_epochs if (args.n_epochs is not None) else (int(n_steps // len(train_dataloader)) + 1)
    max_train_time = additional_args.get('max_train_time', float('inf'))
    accelerator.print(f'== Training terminated after {n_epochs} epochs OR {n_steps} steps OR {max_train_time} sec. ==')
    
    val_check_interval = max(1, min(args.n_steps, 10000) // 10) if do_test_loss else 20
    val_time_check_interval = 600. if do_test_loss else max_train_time
    next_val_check_time = val_time_check_interval
    if do_validation and accelerator.is_main_process:
        def save_training_yaml(y):
            with open(args.aux_out_path, 'w') as f:
                yaml.dump(y, f, sort_keys=False)
        accelerator.print(f'== Performing validation every {val_check_interval} steps or every {val_time_check_interval} sec. ==')
        save_training_yaml([])
        
    do_profiling = args.only_benchmark
    if do_profiling and accelerator.is_main_process:
        save_training_info(
            out_path=args.aux_out_path,
            para_strategy=para_strategy,
            ran_successfully=False,
            execution_time=(time.time() - start_time),
        )

    global_epoch = 0
    global_step = 0
    train_time = 0.
    do_next_epoch = True
    thread_tokens_count = 0
    
    validation_results = []
    training_step_time = []


    def val_loop():

        if (not do_validation) or (len(validation_results) and (global_step == validation_results[-1]['train_steps'])):
            # don't do validation if not needed, or the same step validation is already done
            return

        model.eval()  # comment out if this makes bug in saving weights
        validation_steps = 0
        validation_batch_loss = 0.
        with torch.no_grad():
            for batch in tqdm.tqdm(
                val_dataloader, 
                desc=f'Validation for step {global_step}',
                disable=(not accelerator.is_main_process),
                mininterval=2,
            ):
                output = model(**batch)
                validation_batch_loss += output.loss.item()
                validation_steps += 1
        mean_validation = float(validation_batch_loss / validation_steps)
        all_mean_validation = [None for _ in range(para_strategy.num_gpus)]
        torch.distributed.gather_object(
            mean_validation,
            object_gather_list=(all_mean_validation if accelerator.is_main_process else None),
            dst=0,
        )
        
        all_tokens_count = [None for _ in range(para_strategy.num_gpus)]
        torch.distributed.gather_object(
            int(thread_tokens_count),
            object_gather_list=(all_tokens_count if accelerator.is_main_process else None),
            dst=0,
        )
        
        if accelerator.is_main_process:
            gathered_mean = float(np.mean(all_mean_validation))
            d = {
                'train_steps': global_step,
                'train_time': train_time,
                'train_loss': train_loss.item() if global_step > 0 else None,
                'val_loss': gathered_mean,
                'num_trainable_params': num_trainable_params,
                'num_tokens_trained': sum(all_tokens_count),
            }
        else:
            d = {
                'train_steps': global_step,
            }

        if not do_next_epoch:
            max_use_GB = torch.cuda.max_memory_allocated() / 1024**3
            all_mem_vals = [None for _ in range(para_strategy.num_gpus)]
            torch.distributed.gather_object(
                max_use_GB,
                object_gather_list=(all_mem_vals if accelerator.is_main_process else None),
                dst=0,
            )
            if accelerator.is_main_process:
                d['all_mem_vals'] = all_mem_vals
            accelerator.wait_for_everyone()

        do_eval = (
            (not do_next_epoch) 
            or (not (global_step % len(train_dataloader))) 
            or (train_time > next_val_check_time)
        )
        if do_test_loss and do_eval:
            if strat_type == 'deepspeed':
                ctx_ = deepspeed.zero.GatheredParameters((
                    p for n, p in model.named_parameters() if (p.requires_grad or ('embed' in n))
                ))
            else:
                raise ValueError
            with ctx_:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ckpt_dir = os.path.join(model_save_path, f'epoch{global_epoch:04}-gstep{global_step:06}') 
                    model.save_pretrained(ckpt_dir)
                    accelerator.print(f'Finetuned weights saved to {ckpt_dir}.')
                    d['ckpt_dir'] = ckpt_dir if do_test_loss else None
            accelerator.wait_for_everyone()
            
        d['last_train_step'] = not do_next_epoch
        validation_results.append(d)
        if accelerator.is_main_process:
            accelerator.print(f'Training step {global_step} after {train_time:.3f}s validation loss = {gathered_mean}')
            save_training_yaml(validation_results)
        accelerator.wait_for_everyone()
            
        model.train()
    
    torch.cuda.reset_peak_memory_stats()

    val_loop()
    
    for _ in range(n_epochs):
        
        model.train()
        train_iter = iter(train_dataloader)
                
        for s in tqdm.trange(
            len(train_dataloader), 
            desc=f'Epoch {global_epoch}',
            disable=(not accelerator.is_main_process),
            mininterval=5,
        ):
            
            t0 = time.time()
            
            batch = next(train_iter)
            optimizer.zero_grad()
            output = model(**batch)
            train_loss = output.loss
            accelerator.backward(train_loss)
            optimizer.step()
            global_step += 1
            
            dt = time.time() - t0
            train_time += dt
            if do_profiling:
                training_step_time.append(dt)
            thread_tokens_count += np.sum(batch['attention_mask'].tolist())
            
            # checkpoint and/or break if time reached
            if global_step % 10 == 0:
                train_time = broadcast(torch.tensor(train_time).cuda(), from_process=0).tolist()
                if train_time > max_train_time:
                    do_next_epoch = False
                elif train_time > next_val_check_time:
                    val_loop()
                    next_val_check_time += val_time_check_interval

            # checkpoint and/or break if steps reached
            if (not do_next_epoch) or global_step > n_steps:
                do_next_epoch = False
            elif global_step % val_check_interval == 0:
                val_loop()

            if not do_next_epoch:
                break
            
        if not do_next_epoch:
            break
        else:
            val_loop()
            global_epoch += 1

    val_loop()

    if do_profiling:
        
        max_use_GB = torch.cuda.max_memory_allocated() / 1024**3
        all_mem_vals = [None for _ in range(para_strategy.num_gpus)]
        torch.distributed.gather_object(
            max_use_GB,
            object_gather_list=(all_mem_vals if accelerator.is_main_process else None),
            dst=0,
        )
        max_mem_all_machines = all_mem_vals
        
        if accelerator.is_main_process:
            save_training_info(
                out_path=args.aux_out_path,
                para_strategy=para_strategy,
                ran_successfully=True,
                execution_time=(time.time() - start_time),
                time_results=training_step_time,
                memory_results=max_mem_all_machines,
            )

        
if __name__ == '__main__':
    hf_train()
