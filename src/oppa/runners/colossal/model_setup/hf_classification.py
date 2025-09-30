from contextlib import nullcontext
from typing import Callable, List, Union, Tuple, Optional, Iterator
import inspect
import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from huggingface_hub import login
from datasets import load_dataset

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.lazy import LazyInitContext
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device

from .utils.dist_sampler import StatefulDistributedSampler
from oppa.strategies.parallel_strategy import ParallelisationStrategy


def generate_huggingface_classification_model(
    para_strategy: ParallelisationStrategy,
    booster: Booster, 
    batch_size, 
    model_name='bert-base-uncased',
    dataset_name='imdb',
    cache_dir=None,
    hf_access_token=None,
    lr=3e-4,
    use_flash_attn=False,
    mixed_precision="fp16",
    use_grad_checkpoint=False,
    freeze_non_embeds_params=False,
    weight_decay=0.1,
    num_warmup_steps=None,
    num_epochs=100,
    accumulation_steps=1,
    seq_max_length=512,
    only_do_predownload=False,
    special_tokens=None,
):
    
    os.environ['HF_TOKEN'] = hf_access_token
    # login(token=hf_access_token)

    raw_dataset = load_dataset(
        *dataset_name.split(','), 
        cache_dir=cache_dir, 
        # token=hf_access_token
    ).rename_column("label", "labels")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        cache_dir=cache_dir, 
        # token=hf_access_token,
    )
    
    if special_tokens is not None:
        tokenizer.add_special_tokens(special_tokens)

    def tokenize(batch):
        return tokenizer(
            batch['text'], 
            padding='max_length', 
            truncation=True,
            max_length=seq_max_length,
        )

    train_dataset = raw_dataset['train'].map(tokenize, batched=True).remove_columns(["text"])
    test_dataset = raw_dataset['test'].map(tokenize, batched=True).remove_columns(["text"])
    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    train_dataloader = booster.plugin.prepare_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        seed=42,
        pin_memory=True,
        num_workers=(2 * para_strategy.num_gpus // para_strategy.num_hosts),
        distributed_sampler_cls=StatefulDistributedSampler,
    )
    
    context = nullcontext() if only_do_predownload else LazyInitContext(default_device=get_current_device())

    with context:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            # num_labels=num_labels, 
            # label2id=label2id, 
            # id2label=id2label,
            attn_implementation=("flash_attention_2" if use_flash_attn else None),
            torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16,
            trust_remote_code=True,
            cache_dir=cache_dir, 
            # use_auth_token=hf_access_token,
        )
        model.train()
        # model.resize_token_embeddings(len(tokenizer))
        
    if only_do_predownload:
        return
    
    if use_grad_checkpoint:
        model.gradient_checkpointing_enable()

    # optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = HybridAdam(optimizer_grouped_parameters, lr=lr, eps=1e-8)

    if num_warmup_steps is None:
        num_warmup_steps = int(num_epochs * 0.025 * (len(train_dataloader) // accumulation_steps))

    # lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    output_transform_fn = lambda x: x
    criterion = lambda x: x.loss

    def _criterion(outputs, inputs):
        outputs = output_transform_fn(outputs)
        loss = criterion(outputs)
        return loss

    # ==============================
    # Boost with ColossalAI
    # ==============================
    default_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    torch.set_default_dtype(default_dtype)
    model, optimizer, _, train_dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=train_dataloader,
    )
    
    return {
        "model": model,
        "optimiser": optimizer,
        "criterion": _criterion,
        "lr_scheduler": lr_scheduler,
        'train_dataloader': train_dataloader,
    }