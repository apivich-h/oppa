import lm_eval
import os
import yaml
from pathlib import Path
import time
import datetime
import tqdm
import math
    

def hf_eval():
    
    start_time = time.time()
    
    from oppa.runners.parser import parse_args
    
    args = parse_args(impl='hf')
    additional_args = yaml.safe_load(Path(args.model_additional_args_path).read_text())
    
    print(f"{os.environ.get('CUDA_VISIBLE_DEVICES', None)=}")
    
    if 'hf_access_token_path' in additional_args:
        with open(additional_args['hf_access_token_path'], 'r') as f:
            hf_token = f.read().strip()
        os.environ['HF_TOKEN'] = hf_token
    
    import numpy as np
    import torch
    import torch.distributed
    from torch.utils.data import DataLoader, random_split
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    from datasets import load_dataset, load_from_disk, disable_caching
    from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    from accelerate import Accelerator, init_empty_weights, DeepSpeedPlugin, FullyShardedDataParallelPlugin

    from oppa.strategies.parallel_strategy import ParallelisationStrategy
    from oppa.runners.out_data import save_training_info
    from oppa.utils.dict_utils import map_nested_dicts
        
    assert torch.cuda.device_count() > 0
    disable_caching()
    
    val_batch_size = additional_args.get('lm_eval_batch_size', 128)
    only_eval_last = additional_args.get('only_eval_last', False)

    tokenizer = AutoTokenizer.from_pretrained(additional_args['model_base'])
    if 'special_tokens' in additional_args:
        print(f"Add extra tokens: {additional_args['special_tokens']}")
        tokenizer.add_special_tokens(additional_args['special_tokens'])

    # para_strategy = ParallelisationStrategy.from_args(args)
    
    # ## initialize the strategy
    # acc_kwargs = {
    #     'fsdp_plugin': FullyShardedDataParallelPlugin(),
    # }
    
    accelerator = Accelerator(mixed_precision='bf16')
    # # if strat_type == 'deepspeed':
    # #     accelerator.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = val_batch_size
    
    if not accelerator.is_main_process:
        import logging
        logging.disable(logging.WARN)
            
    with open(args.aux_out_path, 'r') as f:
        d = yaml.safe_load(f)

    with open(os.path.join(os.path.dirname(args.aux_out_path), 'aux_out--noeval.yaml'), 'w') as f:
        yaml.dump(d, f, sort_keys=False)

    updated_d = []
    
    for i, x in enumerate(d):

        eval_this_checkpoint = (
            ((not only_eval_last) and ('ckpt_dir' in x.keys())) or
            (only_eval_last and x['last_train_step'])
        )

        if eval_this_checkpoint:

            ckpt = x['ckpt_dir']
            print(f'Testing checkpoint {ckpt}')
            peft_config = PeftConfig.from_pretrained(ckpt)
            base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
            base_model.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(base_model, ckpt)
            # model = model.cuda()
            model = accelerator.prepare(model)
            accelerator.wait_for_everyone()
        
            results = lm_eval.simple_evaluate(
                model=lm_eval.models.huggingface.HFLM(
                    pretrained=model, 
                    backend='causal',
                    tokenizer=tokenizer, 
                    dtype=torch.bfloat16, 
                    max_length=tokenizer.model_max_length,
                    batch_size=val_batch_size, 
                    trust_remote_code=True,
                ),
                tasks=additional_args['val_tasks'],
                task_manager=lm_eval.tasks.TaskManager(),
                batch_size=val_batch_size,
                max_batch_size=val_batch_size,
                num_fewshot=0,
                # limit=additional_args.get('val_prop', None),
                apply_chat_template=True,
                # cache_requests=False,
            )
            x['eval'] = {
                'results': map_nested_dicts(results['results'], lambda x: float(x)), 
                'higher_is_better': results['higher_is_better'], 
            }

        updated_d.append(x)

        if accelerator.is_main_process:
            with open(args.aux_out_path, 'w') as f:
                # import pickle
                # pickle.dump(updated_d, f)
                yaml.dump(updated_d, f, sort_keys=False)


if __name__ == '__main__':
    hf_eval()
