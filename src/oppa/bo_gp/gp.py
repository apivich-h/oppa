from typing import Callable, List, Union, Dict
import time

import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch
from torch.optim import SGD, Adam

from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch

from .custom_components.parallelism_prior import ParallelCommCostMeanSingleHost, ParallelCommCostMeanMultiHost, MaxMemoryMean


def construct_kernel(input_dim, gp_kernel, gp_kernel_args):
    
    if gp_kernel == 'rbf':
        kernel = ScaleKernel(RBFKernel(ard_num_dims=input_dim))
        
    elif gp_kernel == 'matern':
        nu = gp_kernel_args.get('nu', 5/2)
        kernel = ScaleKernel(MaternKernel(nu=nu, ard_num_dims=input_dim))
        
    else:
        raise ValueError(f'Invalid {gp_kernel=}')
    
    return kernel


def construct_mean(input_dim, gp_mean, gp_mean_args):
    
    if gp_mean == 'constant':
        mean_module = ConstantMean()
    
    elif gp_mean == 'throughput-sh':
        mean_module = ParallelCommCostMeanSingleHost(
            x_upper_bound=gp_mean_args['x_upper_bound'],
            consider_comm=gp_mean_args.get('consider_comm', True),
            step=gp_mean_args.get('step', 0),
        )
        
    elif gp_mean == 'throughput-mh':
        mean_module = ParallelCommCostMeanMultiHost(
            x_upper_bound=gp_mean_args['x_upper_bound'],
            consider_comm=gp_mean_args.get('consider_comm', True),
            step=gp_mean_args.get('step', 0),
        )
        
    elif gp_mean == 'memory':
        mean_module = MaxMemoryMean(
            x_upper_bound=gp_mean_args['x_upper_bound'],
            max_mem_GB=gp_mean_args['max_mem_GB'],
        )
        
    else:
        raise ValueError(f'Invalid {gp_mean=}')
    
    return mean_module
    

        
def fit_gp(
    train_x: torch.Tensor, train_y: torch.Tensor, train_y_var: torch.Tensor = None, 
    gp_kernel: str = 'rbf', gp_kernel_args: Dict = dict(),
    gp_mean: str = 'constant', gp_mean_args: Dict = dict(),
    gp_fit_timeout_sec: float = 60., restore_gp: SingleTaskGP = None,
    separately_fit_mean: bool = False, verbose=True,
):
    
    kernel = construct_kernel(
        input_dim=train_x.shape[1], 
        gp_kernel=gp_kernel,
        gp_kernel_args=gp_kernel_args
    )
    if verbose:
        print(f'Kernel is {kernel}')
    
    mean = construct_mean(
        input_dim=train_x.shape[1], 
        gp_mean=gp_mean,
        gp_mean_args=gp_mean_args,
    )
    if verbose:
        print(f'Mean function is {mean}')
        
    model = SingleTaskGP(
        train_X=train_x, 
        train_Y=train_y,
        train_Yvar=train_y_var,
        mean_module=mean,
        covar_module=kernel,
        input_transform=None,
        outcome_transform=None,
    )
    if train_y_var is not None:
        model.likelihood = FixedNoiseGaussianLikelihood(
            noise=model.likelihood.noise,
            learn_additional_noise=True,
            batch_shape=model._aug_batch_shape,
        )
    mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
    # mll = LeaveOneOutPseudoLikelihood(likelihood=model.likelihood, model=model)
    mll.to(train_x)
    
    if restore_gp is not None:
        model.load_state_dict(restore_gp.state_dict())
        
    if separately_fit_mean:
        if verbose:
            print('Fitting mean function separately')
        optimizer = Adam([{'params': model.mean_module.parameters()}], lr=0.01)
        model.mean_module.train()
        t = time.time()
        loss = float('inf')
        i = 0
        while (time.time() - t < gp_fit_timeout_sec) and (i < 1000) and (loss > 1e-6):
            optimizer.zero_grad()
            output = model.mean_module(train_x)
            loss = ((output.flatten() - train_y.flatten())**2).mean()
            loss.backward()
            optimizer.step()
            i += 1
        if verbose:
            print(f'Fitting mean function done in {i} steps and {time.time() - t}s, loss={loss.item()}')
        model.mean_module.requires_grad_(False)
    
    res = fit_gpytorch_mll_torch(
        mll, 
        optimizer=(lambda p: torch.optim.Adam(p, lr=1e-2)),
        timeout_sec=gp_fit_timeout_sec,
    )
    # res = fit_gpytorch_mll_scipy(
    #     mll, 
    #     timeout_sec=gp_fit_timeout_sec,
    # )
    if verbose:
        print(f'GP fitting done with results {res}')
    return model, res
