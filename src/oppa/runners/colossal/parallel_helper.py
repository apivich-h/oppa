import argparse
from contextlib import nullcontext
from typing import Callable, List, Union

import evaluate
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, GPT2ForSequenceClassification, get_linear_schedule_with_warmup

from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.nn.optimizer import HybridAdam

from oppa.strategies.parallel_strategy import ParallelisationStrategy


def set_up_parallelism(
    para_strategy: ParallelisationStrategy,
    precision="fp16",
    use_fp8_comm=False, 
    # use_fp16_mixed_precision=True,
    use_flash_attention=True,
    max_norm=0,
    **plugin_args,
):
    
    pp_style = 'interleaved' if (para_strategy.num_model_chunks > 1) else '1f1b'
    
    if para_strategy.zero_stage < 3:
    
        # TODO: make the BO work on the variables that are currently fixed
        plugin = HybridParallelPlugin(
            tp_size=para_strategy.tp_size,
            pp_size=para_strategy.pp_size,
            sp_size=para_strategy.sp_size,  # TODO: allow support for sequence parallelism?
            num_microbatches=para_strategy.num_microbatches,
            num_model_chunks=para_strategy.num_model_chunks,
            pp_style=pp_style,
            enable_sequence_parallelism=(para_strategy.sp_size > 1),
            sequence_parallelism_mode=(
                'split_gather' 
                if (para_strategy.sp_size > 1) else None
            ),
            inner_ring_size=None,
            enable_fused_normalization=True,
            enable_flash_attention=use_flash_attention,   # TODO: find out why this breaks PP when set to True
            enable_jit_fused=True,
            zero_stage=para_strategy.zero_stage,
            zero_bucket_size_in_m=para_strategy.zero_bucket_size_mb,
            ddp_bucket_cap_mb=para_strategy.dp_bucket_size_mb,
            cpu_offload=False,
            overlap_communication=True,
            overlap_p2p=True,
            overlap_allgather=False,
            precision=precision,
            fp8_communication=use_fp8_comm,
            max_norm=max_norm,
            **plugin_args,
        )
        
    else:
        raise ValueError('ZERO-3 cannot.')
    
    # use mixed precision
    booster_kwargs = dict()
    # if use_fp16_mixed_precision:
    #     booster_kwargs["mixed_precision"] = 'fp16_naive'
        
    booster = Booster(plugin=plugin, **booster_kwargs)
    return booster