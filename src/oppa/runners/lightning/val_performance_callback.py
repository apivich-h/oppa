# import time
# from typing import Any

# import torch
# import lightning as L
# from lightning.pytorch import Callback, Trainer
# from lightning.pytorch.utilities import rank_zero_only
# from lightning.pytorch.utilities.types import STEP_OUTPUT
# import torch.distributed


# # check against https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#hooks if unsure


# class ValidationCallback(Callback):

#     def __init__(self):
#         super().__init__()
#         self.start_time = None
#         self.validation_start_time = None
#         self.total_time_on_all = 0.
#         self.total_time_on_validation = 0.
#         self.train_steps = 0
#         self.validation_results = []

#     def on_train_start(self, trainer, pl_module):
#         self.start_time = time.time()
        
#     def on_train_end(self, trainer, pl_module):
#         dt = time.time() - self.start_time
#         self.total_time_on_all += dt
#         self.start_time = None
    
#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         self.train_steps += 1
        
#     def on_validation_start(self, trainer, pl_module):
#         self.validation_start_time = time.time()

#     def on_validation_end(self, trainer, pl_module):
#         # timing
#         dt = time.time() - self.validation_start_time
#         self.total_time_on_validation += dt
#         self.validation_start_time = None
#         # losses
#         val_loss = trainer.callback_metrics.get("val_loss")
#         if val_loss is not None:
#             self.validation_results.append({
#                 'train_steps': self.train_steps,
#                 'train_time': (time.time() - self.start_time) - self.total_time_on_validation,
#                 'val_loss': val_loss.item(),
#             })

#     @rank_zero_only
#     def get_validation_stats(self):
#         return self.validation_results
