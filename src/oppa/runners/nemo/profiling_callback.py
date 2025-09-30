import time
from typing import Any

import torch
import lightning as L
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch.distributed


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

    @rank_zero_only
    def get_training_stats(self):
        return {
            'time': self.training_step_time, 
            'mem': self.max_mem_all_machines,
        }