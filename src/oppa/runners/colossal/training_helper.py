from contextlib import nullcontext
from typing import Callable, List, Union
import time
import numpy as np
import gc

import evaluate
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, GPT2ForSequenceClassification, get_linear_schedule_with_warmup

import colossalai
from colossalai.shardformer.layer.utils import Randomizer
from colossalai.accelerator import BaseAccelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator


def move_to_cuda(batch):
    return {k: v.cuda() for k, v in batch.items()}


def train_epochs(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: Callable,
    lr_scheduler: LRScheduler,
    train_dataloader: DataLoader,
    accelerator: BaseAccelerator,
    booster: Booster,
    coordinator: DistCoordinator,
    n_steps: int = None,
    n_epochs: int = None,
    store_time_per_step: bool = False,
):
    use_pipeline = isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1
    # is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()

    model.train()
    optimizer.zero_grad()
    train_dataloader_iter = iter(train_dataloader)
    
    time_each_step = []
    t_start = time.time()

    if n_steps is None:
        n_steps = len(train_dataloader_iter) * n_epochs
    else: 
        n_epochs = (n_steps // len(train_dataloader_iter)) + 1
    
    steps = 0
    with tqdm(
        range(n_steps),
        # disable=not (coordinator.is_master() or is_pp_last_stage),
        disable=(not coordinator.is_master()),
    ) as pbar:

        for _ in range(n_epochs):

            # Forward pass
            for s in range(len(train_dataloader_iter)):
                
                if store_time_per_step:
                    accelerator.synchronize()
                    t_step = time.time()
                              
                if use_pipeline:
                    outputs = booster.execute_pipeline(
                        train_dataloader_iter, model, criterion, optimizer, return_loss=True
                    )
                    loss = outputs["loss"]
                    # # Backward and optimize
                    # if is_pp_last_stage:
                    if coordinator.is_last_process():
                        pbar.set_postfix({"loss": loss.detach().cpu().item()})
                else:
                    data = next(train_dataloader_iter)
                    data = move_to_cuda(data)
                    outputs = model(**data)
                    loss = criterion(outputs, None)
                    # Backward
                    booster.backward(loss, optimizer)
                    pbar.set_postfix({"loss": loss.detach().cpu().item()})

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                pbar.update()
                del outputs
                del loss
                gc.collect()
                accelerator.empty_cache()
                
                if store_time_per_step:
                    accelerator.synchronize()
                    t_step = time.time() - t_step
                    time_each_step.append(t_step)
                    
                yield t_step

                steps += 1
                if steps == n_steps:
                    break
            
    # t_end = time.time()
    
    # results = {
    #     "steps_ran": steps,
    #     "overall_time": t_end - t_start,
    # }
    # if store_time_per_step:
    #     results['time_per_step'] = time_each_step
    # return results