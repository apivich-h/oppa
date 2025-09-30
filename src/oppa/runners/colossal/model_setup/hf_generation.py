from contextlib import nullcontext
from typing import Callable, List, Union, Tuple, Optional, Iterator
from pathlib import Path
import inspect
import os

import evaluate
import torch
from accelerate import PartialState
from accelerate.utils import set_seed

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from transformers import AutoTokenizer
from huggingface_hub import login
from datasets import load_dataset, Dataset, load_from_disk

from colossalai.booster import Booster
from colossalai.lazy import LazyInitContext
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device

from .utils.dist_sampler import StatefulDistributedSampler
from oppa.strategies.parallel_strategy import ParallelisationStrategy



CONFIG_MAP = {
    "toy": transformers.LlamaConfig(num_hidden_layers=4),
    "llama-7b": transformers.LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=32,
        max_position_embeddings=2048,
    ),
    "llama-13b": transformers.LlamaConfig(
        hidden_size=5120,
        intermediate_size=13824,
        num_attention_heads=40,
        num_hidden_layers=40,
        num_key_value_heads=40,
        max_position_embeddings=2048,
    ),
    "llama2-7b": transformers.LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=32,
        max_position_embeddings=4096,
    ),
    "llama2-13b": transformers.LlamaConfig(
        hidden_size=5120,
        intermediate_size=13824,
        num_attention_heads=40,
        num_hidden_layers=40,
        num_key_value_heads=40,
        max_position_embeddings=4096,
    ),
    "llama3-8b": transformers.LlamaConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=8,
        max_position_embeddings=8192,
    ),
    "llama3-70b": transformers.LlamaConfig(
        hidden_size=8192,
        intermediate_size=28672,
        num_attention_heads=64,
        num_hidden_layers=80,
        num_key_value_heads=8,
        max_position_embeddings=8192,
    ),
}


def freeze_non_embeds_parameters(model) -> None:
    """Freeze all parameters except embeddings."""
    for name, params in model.named_parameters():
        if "embed_tokens" not in name and "lm_head" not in name:
            params.requires_grad = False
        else:
            params.requires_grad = True


def unfreeze_parameters(model) -> None:
    for name, params in model.named_parameters():
        params.requires_grad = False


def generate_huggingface_generation_model(
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
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        cache_dir=cache_dir,
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
    
    # Duplicate the input_ids column to create labels
    def add_labels_column(example):
        example["labels"] = example["input_ids"]
        return example

    # Apply the transformation
    train_dataset = train_dataset.map(add_labels_column)
    test_dataset = test_dataset.map(add_labels_column)

    train_dataloader = booster.plugin.prepare_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        seed=42,
        pin_memory=True,
        num_workers=(2 * para_strategy.num_gpus // para_strategy.num_hosts),
        distributed_sampler_cls=StatefulDistributedSampler,
        # use_auth_token=hf_access_token,
    )
    
    context = nullcontext() if only_do_predownload else LazyInitContext(default_device=get_current_device())

    with context:
        if use_flash_attn:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
        # model.resize_token_embeddings(len(tokenizer))
            
    if only_do_predownload:
        return

        # # Freeze part of parameters.
        # if freeze_non_embeds_params:
        #     freeze_non_embeds_parameters(model=model)

    # this is essential, otherwise the grad checkpoint will not work.
    model.train()
    
    if use_grad_checkpoint:
        model.gradient_checkpointing_enable()

    optimizer = HybridAdam(
        model_params=(
            filter(lambda p: p.requires_grad, model.parameters())
            if freeze_non_embeds_params
            else model.parameters()
        ),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        adamw_mode=True,
    )

    warmup_steps = int(num_epochs * 0.025 * (len(train_dataloader) // accumulation_steps))

    lr_scheduler = CosineAnnealingWarmupLR(
        optimizer=optimizer,
        total_steps=num_epochs * (len(train_dataloader) // accumulation_steps),
        warmup_steps=warmup_steps,
        eta_min=0.1 * lr,
    )

    # Flash attention will be disabled because it does NOT support fp32.
    default_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    torch.set_default_dtype(default_dtype)
    model, optimizer, _, train_dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=train_dataloader,
    )

    torch.set_default_dtype(torch.float)
    
    output_transform_fn = lambda x: x
    criterion = lambda x: x.loss

    def _criterion(outputs, inputs):
        # return outputs[0]
        outputs = output_transform_fn(outputs)
        loss = criterion(outputs)
        return loss
    
    return {
        "model": model,
        "optimiser": optimizer,
        "criterion": _criterion,
        "lr_scheduler": lr_scheduler,
        'train_dataloader': train_dataloader,
    }

def generate_random_huggingface_generation_model(
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

    def data_gen(batch_size: int = 4, seq_len: int = 512):
        input_ids = torch.randint(10, 30000, (batch_size, seq_len), device=get_current_device())
        return input_ids.tolist()
    
    def make_synthetic_hf_dataset(num_samples: int,
                              batch_size: int,
                              seq_len: int) -> Dataset:
        """
        Build a HuggingFace Dataset with columns:
        - input_ids: List[int]
        - attention_mask: List[int] (all ones)
        - labels: copy of input_ids
        """
        input_ids_buf: List[List[int]] = []
        attention_buf: List[List[int]] = []
        # generate until we have num_samples examples
        while len(input_ids_buf) < num_samples:
            batch = data_gen(batch_size=batch_size, seq_len=seq_len)
            for seq in batch:
                if len(input_ids_buf) < num_samples:
                    input_ids_buf.append(seq)
                    attention_buf.append([1] * seq_len)

        # create the Dataset
        ds = Dataset.from_dict({
            "input_ids":      input_ids_buf,
            "attention_mask": attention_buf,
        })
        # add labels = input_ids
        ds = ds.map(lambda ex: {"labels": ex["input_ids"]}, 
                    remove_columns=[])
        # set PyTorch tensor format
        ds.set_format("torch")
        return ds

    def get_synthetic_hf_dataset(cache_dir: Union[str, Path],
                             num_samples: int,
                             batch_size: int,
                             seq_len: int) -> Dataset:
        cache_dir = Path(cache_dir)
        if cache_dir.exists():
            print(f"Loading synthetic dataset from cache at {cache_dir}")
            ds = load_from_disk(str(cache_dir))
        else:
            print(f"Generating synthetic dataset and saving to {cache_dir}")
            ds = make_synthetic_hf_dataset(num_samples, batch_size, seq_len)
            cache_dir.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(cache_dir))
        return ds

    train_dataset = get_synthetic_hf_dataset(
        cache_dir="~/.cache/random-dataset",
        num_samples=50000,
        batch_size=batch_size,
        seq_len=seq_max_length,
    )

    train_dataloader = booster.plugin.prepare_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        seed=42,
        pin_memory=True,
        num_workers=(para_strategy.num_gpus // para_strategy.num_hosts),
        distributed_sampler_cls=StatefulDistributedSampler,
        # use_auth_token=hf_access_token,
    )
    
    context = nullcontext() if only_do_predownload else LazyInitContext(default_device=get_current_device())

    with context:
        config = CONFIG_MAP[model_name]
        model = transformers.LlamaForCausalLM(config)
        if mixed_precision == "fp16":
            model = model.half()
        elif mixed_precision == "bf16":
            model = model.to(torch.bfloat16)
        # model.resize_token_embeddings(len(tokenizer))
            
    if only_do_predownload:
        return

        # # Freeze part of parameters.
        # if freeze_non_embeds_params:
        #     freeze_non_embeds_parameters(model=model)

    # this is essential, otherwise the grad checkpoint will not work.
    model.train()
    
    if use_grad_checkpoint:
        model.gradient_checkpointing_enable()

    optimizer = HybridAdam(
        model_params=(
            filter(lambda p: p.requires_grad, model.parameters())
            if freeze_non_embeds_params
            else model.parameters()
        ),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        adamw_mode=True,
    )

    warmup_steps = int(num_epochs * 0.025 * (len(train_dataloader) // accumulation_steps))

    lr_scheduler = CosineAnnealingWarmupLR(
        optimizer=optimizer,
        total_steps=num_epochs * (len(train_dataloader) // accumulation_steps),
        warmup_steps=warmup_steps,
        eta_min=0.1 * lr,
    )

    # Flash attention will be disabled because it does NOT support fp32.
    default_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    torch.set_default_dtype(default_dtype)
    model, optimizer, _, train_dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=train_dataloader,
    )

    torch.set_default_dtype(torch.float)
    
    output_transform_fn = lambda x: x
    criterion = lambda x: x.loss

    def _criterion(outputs, inputs):
        # return outputs[0]
        outputs = output_transform_fn(outputs)
        loss = criterion(outputs)
        return loss
    
    return {
        "model": model,
        "optimiser": optimizer,
        "criterion": _criterion,
        "lr_scheduler": lr_scheduler,
        'train_dataloader': train_dataloader,
    }
