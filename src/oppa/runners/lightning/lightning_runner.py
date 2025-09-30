import os
import yaml
from pathlib import Path
import time
import datetime

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
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    from datasets import load_dataset, load_from_disk
    from peft import get_peft_model, LoraConfig, TaskType
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    from lightning.pytorch.utilities import rank_zero_only
    from lightning.pytorch.utilities.types import STEP_OUTPUT
    from lightning.pytorch import Callback, Trainer
    from lightning.pytorch.strategies import DeepSpeedStrategy, FSDPStrategy, ModelParallelStrategy
        
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
        strategy = DeepSpeedStrategy(
            zero_optimization=(para_strategy.zero_stage > 0),
            stage=para_strategy.zero_stage,
            offload_optimizer=para_strategy.zero_offload,
            offload_parameters=para_strategy.zero_offload,
            overlap_comm=para_strategy.overlap_communication_for_zero,
            reduce_bucket_size=para_strategy.zero_bucket_size_mb,
            allgather_bucket_size=para_strategy.zero_bucket_size_mb,
        )
        # strategy = FSDPStrategy(
        #     sharding_strategy=["NO_SHARD", "SHARD_GRAD_OP", "HYBRID_SHARD", "FULL_SHARD"][para_strategy.zero_stage],
        #     cpu_offload=para_strategy.zero_offload,
        # )
    else:
        raise NotImplementedError
        # strategy = ModelParallelStrategy(
        #     data_parallel_size=para_strategy.dp_size,
        #     tensor_parallel_size=para_strategy.tp_size,
        # )
        # mc = para_strategy.num_model_chunks
        # strategy = nl.MegatronStrategy(
        #     overlap_p2p_comm=para_strategy.overlap_p2p_for_pp,
        #     batch_p2p_comm=(not para_strategy.overlap_p2p_for_pp),
        #     tensor_model_parallel_size=para_strategy.tp_size,
        #     pipeline_model_parallel_size=para_strategy.pp_size,
        #     virtual_pipeline_model_parallel_size=(mc if (mc > 1) else None),
        #     # microbatch_group_size_per_vp_stage=micro_batch_size,
        #     context_parallel_size=para_strategy.sp_size,
        #     pipeline_dtype=torch.bfloat16,
        #     gradient_accumulation_fusion=False,
        #     use_distributed_optimizer=True,
        # )
        # precision_plugin = nl.MegatronMixedPrecision(
        #     precision="bf16-mixed",
        # )        
        
    # check against https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#hooks if unsure

    class ValidationCallback(Callback):

        def __init__(self):
            super().__init__()
            self.start_time = None
            self.train_total_time = 0.
            self.train_steps = 0
            self.validation_steps = 0
            self.validation_batch_loss = 0.
            self.validation_results = []

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self.batch_start_time = time.time()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            dt = time.time() - self.batch_start_time
            self.train_total_time += dt
            self.train_steps += 1
            self.batch_start_time = None
            
        def on_validation_start(self, trainer, pl_module):
            self.validation_steps = 0
            self.validation_batch_loss = 0.
            
        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
            self.validation_steps += 1
            self.validation_batch_loss += outputs.item()

        def on_validation_end(self, trainer, pl_module):
            self.validation_results.append({
                'train_steps': self.train_steps,
                'train_time': self.train_total_time,
                'val_loss': self.validation_batch_loss / float(self.validation_steps),
            })
            self.validation_steps = 0
            self.validation_batch_loss = 0.
            
            lora_state_dict = get_peft_model_state_dict(pl_module.model, adapter_name="default")
            torch.save(lora_state_dict, './test.pt')
            
            def save_():
                with open(args.aux_out_path, 'w') as f:
                    yaml.dump(self.get_validation_stats(), f, sort_keys=False)
            rank_zero_only(save_)()

        def get_validation_stats(self):
            return self.validation_results


    class PerformanceCallback(Callback):

        def __init__(self):
            super().__init__()
            self.batch_start_time = None
            self.training_step_time = []
            self.max_mem_all_machines = []

        def on_train_start(self, trainer, pl_module):
            torch.cuda.reset_max_memory_allocated()

        @rank_zero_only
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self.batch_start_time = time.time()

        @rank_zero_only
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            t = time.time() - self.batch_start_time
            self.training_step_time.append(t)
            
        def on_train_end(self, trainer, pl_module):
            max_use_GB = torch.cuda.max_memory_allocated() / 1024**3
            all_mem_vals = [None for _ in range(trainer.world_size)]
            torch.distributed.gather_object(
                max_use_GB,
                object_gather_list=(all_mem_vals if (trainer.global_rank == 0) else None),
                dst=0,
            )
            self.max_mem_all_machines = all_mem_vals
            
            benchmark_results = self.get_training_stats()
            rank_zero_only(save_training_info)(
                out_path=args.aux_out_path,
                para_strategy=para_strategy,
                ran_successfully=True,
                execution_time=(time.time() - start_time),
                time_results=benchmark_results['time'],
                memory_results=benchmark_results['mem'],
            )

        @rank_zero_only
        def get_training_stats(self):
            return {
                'time': self.training_step_time, 
                'mem': self.max_mem_all_machines,
            }
        
    class WrappedModule(pl.LightningModule):
        
        def __init__(self, model_, tokenizer_):
            super().__init__()
            self.tokenizer = tokenizer_
            self.model = model_
        
        def forward(self, input_ids, attention_mask, decoder_input_ids=None, labels=None):
            return self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)

        def training_step(self, batch, batch_idx):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            return loss

        def validation_step(self, batch, batch_idx):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss = outputs.loss
            return val_loss

        def configure_optimizers(self):
            optim_type = additional_args.get('optim_type', 'adamw')
            lr = additional_args.get('optim_lr', 6e-4)
            if para_strategy.zero_offload:
                import deepspeed
                if optim_type in {'adam', 'adamw'}:
                    use_adamw = optim_type == 'adamw'
                    return deepspeed.ops.adam.DeepSpeedCPUAdam(
                        self.model.parameters(), 
                        lr=lr,
                        adamw_mode=use_adamw,
                    )
            else:
                if optim_type == 'adam':
                    return torch.optim.Adam(self.model.parameters(), lr=lr)
                elif optim_type == 'adamw':
                    return torch.optim.AdamW(self.model.parameters(), lr=lr)
            raise ValueError  # in the case that optimiser did not get chosen
        
    callbacks = []
    
    if args.only_setup:
        print('Only running setup and check.')
        # nemo_logger = None
        other_args = dict(
            enable_checkpointing=False,
            fast_dev_run=True,
            logger=False,
        )
        profiler = None
        do_val = False
        
    elif args.only_benchmark:
        assert args.n_steps is not None
        print('Only running benchmarking of train step times.')
        # nemo_logger = None
        other_args = dict(
            enable_checkpointing=False,
            limit_val_batches=0,
            limit_test_batches=0,
            max_steps=args.n_steps,
            fast_dev_run=False,
            logger=False,
        )
        profiler = PerformanceCallback()
        callbacks.append(profiler)
        do_val = False
        
    elif args.only_short_training:
        assert args.n_steps is not None
        print('Performing shorter training with validation.')
        train_steps_per_epoch = len(train_dataset) // global_batch_size
        assert args.n_steps > 10
        val_check_interval = min(1., min(100, args.n_steps // 10) / train_steps_per_epoch)
        print(f'Doing validation check every {val_check_interval} epochs, '
              f' or every {int(train_steps_per_epoch * val_check_interval)} train steps.')
        # nemo_logger = pl.Lo(
        #     log_dir=os.path.join(args.model_out_path, 'logs')
        # )
        profiler = ValidationCallback()
        callbacks.append(profiler)
        other_args = dict(
            enable_checkpointing=False,
            max_steps=args.n_steps,
            val_check_interval=val_check_interval,
            limit_test_batches=0,
            fast_dev_run=False,
            logger=True,
        )
        do_val = True
        
    else:
        print('Performing training with validation.')
        # nemo_logger = pl.Lo(
        #     log_dir=os.path.join(args.model_out_path, 'logs')
        # )
        train_steps_per_epoch = len(train_dataset) // global_batch_size
        profiler = ValidationCallback()
        callbacks.append(profiler)
        other_args = dict(
            enable_checkpointing=True,
            max_steps=(-1 if args.n_steps is None else args.n_steps),
            max_epochs=(None if args.n_epochs is None else args.n_epochs),
            # val_check_interval=min(1., (args.n_steps // 10) / train_steps_per_epoch),
            val_check_interval=(10. / train_steps_per_epoch),
            limit_test_batches=0,
            fast_dev_run=False,
            logger=True,
        )
        do_val = True
        
    max_time = (
        datetime.timedelta(seconds=additional_args['max_train_time'])
        if 'max_train_time' in additional_args.keys() else None
    )
    
    trainer = pl.Trainer(
        devices=para_strategy.num_gpus//para_strategy.num_hosts,
        num_nodes=para_strategy.num_hosts,
        accelerator="gpu",
        strategy=strategy,
        callbacks=callbacks,
        precision='bf16',
        accumulate_grad_batches=1,
        default_root_dir=args.model_out_path,
        max_time=max_time,
        **other_args,
    )
    
    with trainer.init_module(empty_init=True):
        model = AutoModelForCausalLM.from_pretrained(model_hf)
        model.resize_token_embeddings(len(tokenizer))
        if model_transform is not None:
            model = get_peft_model(model, model_transform)
        module = WrappedModule(model_=model, tokenizer_=tokenizer)
        
    
                
    if args.only_benchmark:
        rank_zero_only(save_training_info)(
            out_path=args.aux_out_path,
            para_strategy=para_strategy,
            ran_successfully=False,
            execution_time=(time.time() - start_time),
        )
        
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
        
        
    # num_workers and pin_memory set according to https://lightning.ai/docs/pytorch/stable/advanced/speed.html
    trainer.fit(
        model=module,
        train_dataloaders=DataLoader(
            train_dataset, 
            batch_size=dp_batch_size, 
            shuffle=True,
            pin_memory=True,
            num_workers=(2 * para_strategy.num_gpus // para_strategy.num_hosts),
            collate_fn=collate_fn,
        ), 
        val_dataloaders=(DataLoader(
            val_dataset, 
            batch_size=(additional_args.get('val_batch_size', 128) // para_strategy.num_gpus), 
            shuffle=False,
            pin_memory=True,
            num_workers=(2 * para_strategy.num_gpus // para_strategy.num_hosts),
            collate_fn=collate_fn,
        ) if do_val else None),
    )
        
    # if args.only_benchmark:
    #     benchmark_results = profiler.get_training_stats()
    #     if benchmark_results is not None:
    #         rank_zero_only(save_training_info)(
    #             out_path=args.aux_out_path,
    #             para_strategy=para_strategy,
    #             ran_successfully=True,
    #             execution_time=(time.time() - start_time),
    #             time_results=benchmark_results['time'],
    #             memory_results=benchmark_results['mem'],
    #         )
    # elif args.only_short_training:
    #     def save_():
    #         with open(args.aux_out_path, 'w') as f:
    #             yaml.dump(profiler.get_validation_stats(), f, sort_keys=False)
    #     rank_zero_only(save_)()
        
        
if __name__ == '__main__':
    lightning_runner()
