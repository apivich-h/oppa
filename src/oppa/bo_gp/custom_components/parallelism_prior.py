from typing import Callable, List, Union, Dict
import yaml
import os
import random

import logging
logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn as nn
from gpytorch.means import Mean

from torch.optim import SGD, Adam
from torch.quasirandom import SobolEngine
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean, Mean
import botorch

botorch.settings.debug(True)


OOM_ADD_FACTOR = 0.2


class ParallelCommCostMeanSingleHost(Mean):
    r"""
    A custom mean function modeling the communication cost of:
        1) Data-Parallel (DP) AllReduce
        2) Tensor-Parallel (TP) AllReduce
        3) Pipeline-Parallel (PP) communication
    using the formula:

        Cost_AllReduce(p, n) = 2 * (log p) * alpha   startup overhead
                             + 2 * ((p - 1) / p) * n * beta    per-byte transmission cost
                             +     ((p - 1) / p) * n * gamma   per-byte reduction cost

    We store separate learnable parameters for DP and TP:
        beta_dp, gamma_dp   and   beta_tp, gamma_tp.

    We multiply each cost by a learnable scale factor:
        dp_factor, tp_factor, pp_factor.

    The input x is tensors of 
        {num_gpus,
        num_hosts,
        dp_size,
        tp_size,
        pp_size,
        sp_size,
        dp_bucket_size_mb,
        zero_stage,
        zero_bucket_size_mb,
        num_microbatches,
        num_model_chunks,
        (0.3 if use_grad_checkpoint else 0.7)}

    The final cost is the sum of computation and both DP and TP costs across the batch:

        cost = cost_dp + cost_tp + cost_pp + comp
    """

    def __init__(self,
                 learn_scales: bool = True,
                 init_comp: float = 1.0,
                 init_alpha: float = 1.0,
                 init_beta: float = 1.0,
                 init_gamma: float = 1.0,
                 x_upper_bound: torch.Tensor = None,
                 consider_comm: bool = True):
        """
        :param learn_scales: If True, dp_factor and tp_factor are learnable.
        :param init_alpha, init_beta, init_gamma: initial values for alpha, beta, gamma.
        """
        super().__init__()
        self.x_upper_bound = x_upper_bound
        self.consider_comm = consider_comm

        self.comp = torch.nn.Parameter(torch.tensor(init_comp).log())
        # self.c = torch.nn.Parameter(torch.tensor(0.))
        self.offset = torch.nn.Parameter(torch.tensor(0.))
        
        if self.consider_comm:
        
            # self.alpha = torch.nn.Parameter(torch.tensor(init_alpha).log())

            # --- DP parameters ---
            self.alpha_dp = torch.nn.Parameter(torch.tensor(init_alpha).log())
            # self.beta_dp  = torch.nn.Parameter(torch.tensor(init_beta).log())
            self.gamma_dp = torch.nn.Parameter(torch.tensor(init_gamma).log())

            # --- TP parameters ---
            self.alpha_tp = torch.nn.Parameter(torch.tensor(init_alpha).log())
            # self.beta_tp  = torch.nn.Parameter(torch.tensor(init_beta).log())
            self.gamma_tp = torch.nn.Parameter(torch.tensor(init_gamma).log())

            # --- PP parameters ---
            self.alpha_pp = torch.nn.Parameter(torch.tensor(init_alpha).log())
            self.beta_pp  = torch.nn.Parameter(torch.tensor(init_beta).log())

            # # Optionally learn a multiplier (amplitude) for DP and TP costs
            # if learn_scales:
            #     self.dp_factor = torch.nn.Parameter(torch.tensor(1.0).log())
            #     self.tp_factor = torch.nn.Parameter(torch.tensor(1.0).log())
            #     self.pp_factor = torch.nn.Parameter(torch.tensor(1.0).log())
            # else:
            #     self.register_buffer("dp_factor", torch.tensor(1.0).log())
            #     self.register_buffer("tp_factor", torch.tensor(1.0).log())
            #     self.register_buffer("pp_factor", torch.tensor(1.0).log())
            
        # else:    
        
        #     self.C_tp = torch.nn.Parameter(torch.tensor(1.).log())
        #     self.C_pp = torch.nn.Parameter(torch.tensor(1.).log())
        #     self.C_dp = torch.nn.Parameter(torch.tensor(1.).log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape (batch_size, 12)

        :return cost / throughput: shape (batch_size,)
        """

        # Extract n_gpus, dp_size, tp_size, pp_size, microbatches, n_model_chunks
        n_gpus = (x[..., 0] * self.x_upper_bound[0]).exp2()
        p_dp = (x[..., 2] * self.x_upper_bound[2]).exp2()
        p_tp = (x[..., 3] * self.x_upper_bound[3]).exp2()
        p_pp = (x[..., 4] * self.x_upper_bound[4]).exp2()
        mb = (x[..., 9] * self.x_upper_bound[9]).exp2()
        mc = (x[..., 10] * self.x_upper_bound[10]).exp2()

        if self.consider_comm:
            
            # === DP AllReduce cost ===
            # formula: 2 log(p) alpha + 2((p - 1)/p) n beta + ((p - 1)/p) n gamma
            cost_dp = (torch.log2(p_dp) * self.alpha_dp.exp()
                    # + 2.0 * self.beta_dp.exp() * (p_dp - 1.0) / p_dp
                    +       self.gamma_dp.exp() * (p_dp - 1.0) / p_dp)
            # cost_dp = cost_dp * self.dp_factor.exp()

            # === TP AllReduce cost ===
            # cost_tp = mb * (2.0 * torch.log(p_tp) * self.alpha.exp()
            #         + 2.0 * (p_tp - 1.0) / (p_tp * mb) * self.beta_tp.exp()
            #         +       (p_tp - 1.0) / (p_tp * mb) * self.gamma_tp.exp())
            cost_tp = mb * (torch.log2(p_tp) * self.alpha_tp.exp()
                    # + 2.0 * self.beta_tp.exp() * (p_tp - 1.0) / (p_tp * mb)
                    +      self.gamma_tp.exp() * (p_tp - 1.0) / (p_tp * mb))
            # cost_tp = cost_tp * self.tp_factor.exp()

            # === PP Point-to-Point communication cost ===
            # cost_pp = mb * mc * (self.alpha.exp() + (self.beta_pp.exp() / (mb * mc)))
            cost_pp = mb * mc * (self.alpha_pp.exp() + (self.beta_pp.exp() / (mb * mc)))
            # cost_pp = cost_pp * self.pp_factor.exp()

            # Sum DP + TP cost
            comm_t = cost_dp + cost_tp + cost_pp

            # Add computation cost
            comp_t = (self.comp.exp() / n_gpus) * (mb + ((p_pp - 1) / mc))
            
            cost_total = comm_t + comp_t + self.offset.exp()
        
        else:
            
            # cost_tp = self.C_tp * (mb / (p_pp * mc)) * (p_tp - 1.) / p_tp
            # cost_pp = self.C_pp * mb * mc
            # cost_dp = self.C_dp * (p_dp - 1.)
            # cost_total = cost_dp + cost_tp + cost_pp

            # Add computation cost
            comp_t = (self.comp.exp() / (p_dp * p_pp)) * (mb + (p_pp - 1) / mc)
            
            # cost_total = cost_total + comp_t + 1e-3  #+ self.offset.exp()
            cost_total = comp_t + self.offset.exp()

        return 1. / cost_total


class ParallelCommCostMeanMultiHost(Mean):
    r"""
    A custom mean function modeling communication cost for distributed training.

    Models:
        1) Data-Parallel (DP) AllReduce (for gradients)
        2) Tensor-Parallel (TP) AllReduce (for frequent activation communication)
        3) Pipeline-Parallel (PP) communication (P2P between stages)

    AllReduce cost (DP, TP) uses Ring AllReduce model:
        Cost_Ring(p, n, alpha, beta, gamma, delta) = 
            2*(p-1)*alpha + ((p-1)/p)*n*(2*beta + gamma + 3*delta)
    
    Pipeline P2P cost (PP):
        Cost_P2P_Pipe(p_pp, mb, M_act_pp, alpha, beta_pp) =
            (p_pp - 1) * 2 * mb * (alpha + M_act_pp * beta_pp)

    Data sizes 'n':
    - DP (n_dp): Derived from learnable total model size M, tp_size, and zero_stage.
    - TP (M_act_tp): Learnable characteristic data size for one TP activation communication.
    - PP (M_act_pp): Learnable characteristic data size for one P2P activation/gradient transfer between PP stages.
    - beta, gamma, delta parameters are per-unit-of-respective-data-size costs.

    Input x (batch_size, 12):
        {0:num_gpus, 1:num_hosts, 2:dp_size, 3:tp_size, 4:pp_size (p_pp), 5:sp_size,
         6:dp_bucket_size_mb, 7:zero_stage, 8:zero_bucket_size_mb,
         9:num_microbatches (mb), 10:num_model_chunks (mc_model_chunks for intra-stage interleaving),
         11:grad_checkpoint_factor}
    
    Output: throughput (1 / total_cost).
    """

    def __init__(self,
                 learn_scales: bool = True,
                 init_M: float = 1.0, # For DP: Initial guess for total model size (e.g., in MB)
                 init_M_act_tp: float = 0.1, # For TP: Initial guess for activation communication data size (e.g., in MB)
                 init_M_act_pp: float = 0.1, # For PP: Initial guess for activation/gradient transfer data size (e.g., in MB)
                 init_tp_ops_per_microbatch: float = 40.0, # Initial guess for num TP ops per microbatch
                 init_comp: float = 1.0,
                 # Parameters for INTRA-node costs
                 init_alpha: float = 1.0,
                 init_beta: float = 1.0, # Per-unit-of-data cost (e.g., per MB)
                 init_gamma: float = 1.0,
                 init_delta: float = 1.0,
                 x_upper_bound: torch.Tensor = None,
                 # Parameters for INTER-node costs
                 init_alpha_inter: float = None, 
                 init_beta_inter: float = None,  
                 init_gamma_inter: float = None,
                 init_delta_inter: float = None,
                 consider_comm: bool = True,
                 step: int = 0,
                 ):
        super().__init__()
        
        self.x_upper_bound = x_upper_bound
        self.consider_comm = consider_comm

        self.prior_scale = torch.cos(torch.tensor(step) / 40 * np.pi) 
        self.prior_scale = torch.clamp(self.prior_scale, min=0.0)

        self.constant = nn.Parameter(torch.tensor(1.0).log())

        self.M = nn.Parameter(torch.tensor(init_M).log()) 
        self.comp = nn.Parameter(torch.tensor(init_comp).log())
        
        if self.consider_comm:
            self.M_act_tp_log = nn.Parameter(torch.tensor(init_M_act_tp).log())
            self.tp_ops_per_microbatch_log = nn.Parameter(torch.tensor(init_tp_ops_per_microbatch).log())
            self.M_act_pp_log = nn.Parameter(torch.tensor(init_M_act_pp).log())

            self.alpha_intra = nn.Parameter(torch.tensor(init_alpha).log())
            self.beta_dp_intra  = nn.Parameter(torch.tensor(init_beta).log()) 
            self.gamma_dp_intra = nn.Parameter(torch.tensor(init_gamma).log())
            self.delta_dp_intra = nn.Parameter(torch.tensor(init_delta).log())
            self.beta_tp_intra  = nn.Parameter(torch.tensor(init_beta).log()) # Scales with M_act_tp
            self.gamma_tp_intra = nn.Parameter(torch.tensor(init_gamma).log()) # Scales with M_act_tp
            self.delta_tp_intra = nn.Parameter(torch.tensor(init_delta).log()) # Scales with M_act_tp
            self.beta_pp_intra  = nn.Parameter(torch.tensor(init_beta).log()) # Scales with M_act_pp

            alpha_inter_val = init_alpha_inter if init_alpha_inter is not None else init_alpha * 10.0
            beta_inter_val  = init_beta_inter if init_beta_inter is not None else init_beta * 5.0
            gamma_inter_val = init_gamma_inter if init_gamma_inter is not None else init_gamma * 5.0
            delta_inter_val = init_delta_inter if init_delta_inter is not None else init_delta * 5.0

            self.alpha_inter = nn.Parameter(torch.tensor(alpha_inter_val).log())
            self.beta_dp_inter  = nn.Parameter(torch.tensor(beta_inter_val).log())
            self.gamma_dp_inter = nn.Parameter(torch.tensor(gamma_inter_val).log())
            self.delta_dp_inter = nn.Parameter(torch.tensor(delta_inter_val).log())
            self.beta_tp_inter  = nn.Parameter(torch.tensor(beta_inter_val).log())
            self.gamma_tp_inter = nn.Parameter(torch.tensor(gamma_inter_val).log())
            self.delta_tp_inter = nn.Parameter(torch.tensor(delta_inter_val).log())
            self.beta_pp_inter  = nn.Parameter(torch.tensor(beta_inter_val).log())

            if learn_scales:
                self.dp_factor_intra = nn.Parameter(torch.tensor(1.0).log())
                self.tp_factor_intra = nn.Parameter(torch.tensor(1.0).log())
                self.pp_factor_intra = nn.Parameter(torch.tensor(1.0).log())
                self.dp_factor_inter = nn.Parameter(torch.tensor(1.0).log())
                self.tp_factor_inter = nn.Parameter(torch.tensor(1.0).log())
                self.pp_factor_inter = nn.Parameter(torch.tensor(1.0).log())
            else:
                for name_prefix in ["dp", "tp", "pp"]:
                    for node_level in ["intra", "inter"]:
                        param_name = f"{name_prefix}_factor_{node_level}"
                        self.register_buffer(param_name, torch.tensor(0.0)) 
        
        if hasattr(super(), 'initialize') and callable(super().initialize):
            try: super().initialize() 
            except TypeError:
                try: super().initialize(batch_shape=torch.Size(), feature_dim=1) 
                except Exception as e: logger.warning(f"Failed to call super().initialize(): {e}.")

    def _calculate_allreduce_cost(self, p, n, alpha_exp, beta_exp, gamma_exp, delta_exp):
        p_float = p.float()
        n_float = torch.max(n.float(), torch.tensor(1e-6, device=n.device, dtype=n.dtype))
        cost = torch.zeros_like(p_float, device=p.device, dtype=p.dtype)
        active_mask = p_float > 1.0001 
        if torch.any(active_mask):
            p_active = p_float[active_mask]
            n_active = n_float[active_mask]
            term_alpha = 2.0 * (p_active - 1.0) * alpha_exp
            safe_p_active = torch.max(p_active, torch.tensor(1.0 + 1e-6, device=p_active.device, dtype=p_active.dtype))
            common_factor_data_coeff = (p_active - 1.0) / safe_p_active
            data_dependent_sum = (2.0 * beta_exp + gamma_exp + 3.0 * delta_exp)
            term_data = common_factor_data_coeff * n_active * data_dependent_sum
            cost[active_mask] = (term_alpha + term_data).to(cost.dtype)
        return cost

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.x_upper_bound is not None:
            self.x_upper_bound = self.x_upper_bound.to(x.device)

        n_gpus = (x[..., 0] * self.x_upper_bound[0]).exp2()
        num_hosts = (x[..., 1] * self.x_upper_bound[1]).exp2()
        p_dp = (x[..., 2] * self.x_upper_bound[2]).exp2()
        p_tp = (x[..., 3] * self.x_upper_bound[3]).exp2()
        p_pp = (x[..., 4] * self.x_upper_bound[4]).exp2() # Number of pipeline stages
        
        zero_stage_input = x[..., 7]
        zero_stage_val = torch.round(zero_stage_input * self.x_upper_bound[7] if self.x_upper_bound is not None and self.x_upper_bound.numel() > 7 else zero_stage_input)

        mb_microbatches = (x[..., 9] * self.x_upper_bound[9]).exp2() 
        mc_model_chunks = (x[..., 10] * self.x_upper_bound[10]).exp2() # Chunks for intra-stage interleaving

        M_exp = self.M.exp().expand_as(n_gpus)

        p_tp_clamped = torch.max(p_tp, torch.tensor(1.0, device=x.device, dtype=x.dtype))

        zero_divisor = torch.where(zero_stage_val < 1.999, 
                                   torch.tensor(1.0, device=x.device, dtype=x.dtype), 
                                   torch.tensor(2.0, device=x.device, dtype=x.dtype))
        n_dp = (M_exp / p_tp_clamped) / zero_divisor
        n_dp = torch.max(n_dp, torch.tensor(1e-6, device=x.device, dtype=x.dtype))

        num_hosts_clamped = torch.max(num_hosts, torch.tensor(1.0, device=x.device, dtype=x.dtype))
        gpus_per_host = torch.max(n_gpus / num_hosts_clamped, torch.tensor(1.0, device=x.device, dtype=x.dtype))
        is_multi_host = num_hosts > 1.0001 

        cost_total_batch = torch.zeros_like(n_gpus) 

        if self.consider_comm:
            M_act_tp_exp = self.M_act_tp_log.exp().expand_as(n_gpus) # Data size for a single TP transfer
            tp_ops_per_microbatch_exp = self.tp_ops_per_microbatch_log.exp()
            M_act_pp_exp = self.M_act_pp_log.exp().expand_as(n_gpus) # Data size for a single PP transfer
            
            alpha_intra_exp = self.alpha_intra.exp()
            beta_dp_intra_exp, gamma_dp_intra_exp, delta_dp_intra_exp = self.beta_dp_intra.exp(), self.gamma_dp_intra.exp(), self.delta_dp_intra.exp()
            beta_tp_intra_exp, gamma_tp_intra_exp, delta_tp_intra_exp = self.beta_tp_intra.exp(), self.gamma_tp_intra.exp(), self.delta_tp_intra.exp()
            beta_pp_intra_exp_val = self.beta_pp_intra.exp() # Per-unit-of-M_act_pp cost

            alpha_inter_exp = self.alpha_inter.exp()
            beta_dp_inter_exp, gamma_dp_inter_exp, delta_dp_inter_exp = self.beta_dp_inter.exp(), self.gamma_dp_inter.exp(), self.delta_dp_inter.exp()
            beta_tp_inter_exp, gamma_tp_inter_exp, delta_tp_inter_exp = self.beta_tp_inter.exp(), self.gamma_tp_inter.exp(), self.delta_tp_inter.exp()
            beta_pp_inter_exp_val = self.beta_pp_inter.exp() # Per-unit-of-M_act_pp cost

            dp_factor_intra_exp, tp_factor_intra_exp, pp_factor_intra_exp = self.dp_factor_intra.exp(), self.tp_factor_intra.exp(), self.pp_factor_intra.exp()
            dp_factor_inter_exp, tp_factor_inter_exp, pp_factor_inter_exp = self.dp_factor_inter.exp(), self.tp_factor_inter.exp(), self.pp_factor_inter.exp()

            # === DP AllReduce Cost (Gradients) ===
            cost_dp = torch.zeros_like(n_gpus)
            dp_active_mask = p_dp > 1.0001
            if torch.any(dp_active_mask):
                _is_multi_host_dp = is_multi_host 
                _p_dp_active = p_dp 
                _gpus_per_host_dp = gpus_per_host
                _n_dp_active = n_dp 

                is_hierarchical_dp = _is_multi_host_dp & (_p_dp_active > _gpus_per_host_dp + 0.0001)
                is_flat_inter_node_dp = _is_multi_host_dp & (~is_hierarchical_dp)
                is_flat_intra_node_dp = ~_is_multi_host_dp

                num_inter_participants_dp = torch.ceil(_p_dp_active / torch.max(_gpus_per_host_dp, torch.tensor(1.0, device=x.device, dtype=x.dtype)))
                
                cost_dp_h_intra_val = self._calculate_allreduce_cost(
                    _gpus_per_host_dp, _n_dp_active, alpha_intra_exp, beta_dp_intra_exp, gamma_dp_intra_exp, delta_dp_intra_exp
                ) * dp_factor_intra_exp
                cost_dp_h_inter_val = self._calculate_allreduce_cost(
                    num_inter_participants_dp, _n_dp_active, alpha_inter_exp, beta_dp_inter_exp, gamma_dp_inter_exp, delta_dp_inter_exp
                ) * dp_factor_inter_exp
                hierarchical_total_cost_val = cost_dp_h_intra_val + cost_dp_h_inter_val

                flat_inter_node_cost_val = self._calculate_allreduce_cost(
                    _p_dp_active, _n_dp_active, alpha_inter_exp, beta_dp_inter_exp, gamma_dp_inter_exp, delta_dp_inter_exp
                ) * dp_factor_inter_exp
                flat_intra_node_cost_val = self._calculate_allreduce_cost(
                    _p_dp_active, _n_dp_active, alpha_intra_exp, beta_dp_intra_exp, gamma_dp_intra_exp, delta_dp_intra_exp
                ) * dp_factor_intra_exp
                
                current_cost_dp = torch.zeros_like(_p_dp_active)
                current_cost_dp = torch.where(is_hierarchical_dp, hierarchical_total_cost_val, current_cost_dp)
                current_cost_dp = torch.where(is_flat_inter_node_dp, flat_inter_node_cost_val, current_cost_dp)
                current_cost_dp = torch.where(is_flat_intra_node_dp, flat_intra_node_cost_val, current_cost_dp)
                cost_dp = torch.where(dp_active_mask, current_cost_dp, torch.zeros_like(cost_dp))


            # === TP AllReduce Cost (Activations) ===
            cost_tp = torch.zeros_like(n_gpus)
            tp_active_mask = p_tp > 1.0001
            if torch.any(tp_active_mask):
                _p_tp_active = p_tp
                tp_is_inter_node = is_multi_host & (_p_tp_active > gpus_per_host + 0.0001)
                
                cost_tp_one_op_intra = self._calculate_allreduce_cost(
                    _p_tp_active, M_act_tp_exp, 
                    alpha_intra_exp, beta_tp_intra_exp, gamma_tp_intra_exp, delta_tp_intra_exp
                )
                cost_tp_one_op_inter = self._calculate_allreduce_cost(
                    _p_tp_active, M_act_tp_exp,
                    alpha_inter_exp, beta_tp_inter_exp, gamma_tp_inter_exp, delta_tp_inter_exp
                )

                cost_tp_one_op = torch.where(tp_is_inter_node,
                                             cost_tp_one_op_inter * tp_factor_inter_exp,
                                             cost_tp_one_op_intra * tp_factor_intra_exp)
                
                total_tp_comms_factor = tp_ops_per_microbatch_exp * mb_microbatches
                cost_tp_val = total_tp_comms_factor * cost_tp_one_op
                cost_tp = torch.where(tp_active_mask, cost_tp_val, torch.zeros_like(cost_tp))


            # === PP Point-to-Point Communication Cost ===
            cost_pp = torch.zeros_like(n_gpus)
            pp_active_mask = p_pp > 1.0001 # Only active if more than one stage
            if torch.any(pp_active_mask):
                # Number of communication boundaries
                # p_pp is a tensor, ensure ops are tensor ops
                num_pp_boundaries = p_pp - 1.0 
                
                # Total number of P2P transfers (send activation, send gradient for each microbatch across each boundary)
                # This applies where p_pp > 1. If p_pp = 1, num_pp_boundaries = 0, so cost is 0.
                num_total_pp_transfers = num_pp_boundaries * 2.0 * mb_microbatches

                # Cost of a single P2P transfer (latency + data_size * cost_per_unit_data)
                cost_per_single_transfer_intra = alpha_intra_exp + M_act_pp_exp * beta_pp_intra_exp_val
                cost_per_single_transfer_inter = alpha_inter_exp + M_act_pp_exp * beta_pp_inter_exp_val

                total_pp_intra = num_total_pp_transfers * cost_per_single_transfer_intra * pp_factor_intra_exp
                total_pp_inter = num_total_pp_transfers * cost_per_single_transfer_inter * pp_factor_inter_exp
                
                pp_is_inter_node = is_multi_host # Simplified: if multi-host, assume PP crosses nodes
                current_cost_pp = torch.where(pp_is_inter_node, total_pp_inter, total_pp_intra)
                
                # Ensure cost is applied only where pp is active (p_pp > 1)
                cost_pp = torch.where(pp_active_mask, current_cost_pp, torch.zeros_like(cost_pp))


            cost_total_batch = cost_dp + cost_tp + cost_pp
            cost_total_batch = (1 - self.prior_scale) * self.constant.exp() + self.prior_scale * cost_total_batch
            
            # === Computation Cost ===
            safe_p_pp = torch.max(p_pp, torch.tensor(1.0, device=x.device, dtype=x.dtype))
            safe_mc_model_chunks = torch.max(mc_model_chunks, torch.tensor(1.0, device=x.device, dtype=x.dtype))
            p_dp_clamped = torch.max(p_dp, torch.tensor(1.0, device=x.device, dtype=x.dtype))
            
            # self.comp.exp() is comp time for one microbatch on one TP-sharded segment of a pipeline stage
            time_steady_state_per_stage = self.comp.exp() / p_tp_clamped 
            
            # Pipeline bubble slots, reduced by intra-stage chunking (mc_model_chunks)
            # Bubble is (P-1) T_stage_slots. With M chunks, effectively (P-1)/M T_stage_slots.
            pipeline_bubble_slots = (safe_p_pp - 1.0) / safe_mc_model_chunks
            # Ensure bubble is not negative if p_pp=1
            pipeline_bubble_slots = torch.max(pipeline_bubble_slots, torch.tensor(0.0, device=x.device, dtype=x.dtype))

            comp_t_one_dp_rank = time_steady_state_per_stage * (mb_microbatches + pipeline_bubble_slots)
            comp_t = comp_t_one_dp_rank / p_dp_clamped
            
            cost_total_batch = cost_total_batch + comp_t
        else: 
            # Simplified Computation Cost (if not considering communication)
            safe_p_pp = torch.max(p_pp, torch.tensor(1.0, device=x.device, dtype=x.dtype))
            safe_mc_model_chunks = torch.max(mc_model_chunks, torch.tensor(1.0, device=x.device, dtype=x.dtype))
            p_dp_clamped = torch.max(p_dp, torch.tensor(1.0, device=x.device, dtype=x.dtype))
            
            # No TP sharding of compute if comm is off, so p_tp_clamped is effectively 1 here.
            comp_per_stage_per_microbatch_no_comm = self.comp.exp() 
            
            pipeline_bubble_slots_no_comm = (safe_p_pp - 1.0) / safe_mc_model_chunks
            pipeline_bubble_slots_no_comm = torch.max(pipeline_bubble_slots_no_comm, torch.tensor(0.0, device=x.device, dtype=x.dtype))

            comp_t_one_dp_rank_no_comm = comp_per_stage_per_microbatch_no_comm * (mb_microbatches + pipeline_bubble_slots_no_comm)
            comp_t = comp_t_one_dp_rank_no_comm / p_dp_clamped
            cost_total_batch = comp_t

        cost_total_batch = torch.max(cost_total_batch, torch.tensor(1e-7, device=x.device, dtype=x.dtype))
        return 1.0 / cost_total_batch
    
    
class MaxMemoryMean(Mean):

    def __init__(self, x_upper_bound: torch.Tensor, max_mem_GB: float):
        super().__init__()
        self.x_upper_bound = x_upper_bound
        self.max_mem_GB = max_mem_GB
        
        self.m1 = torch.nn.Parameter(torch.tensor(1.).log())
        self.m2 = torch.nn.Parameter(torch.tensor(1.).log())
        self.m3 = torch.nn.Parameter(torch.tensor(1.).log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Extract n_gpus, dp_size, tp_size, pp_size, microbatches, n_model_chunks
        n_gpus = (x[..., 0] * self.x_upper_bound[0]).exp2()
        p_dp = (x[..., 2] * self.x_upper_bound[2]).exp2()
        p_tp = (x[..., 3] * self.x_upper_bound[3]).exp2()
        p_pp = (x[..., 4] * self.x_upper_bound[4]).exp2()
        mb = (x[..., 9] * self.x_upper_bound[9]).exp2()
        mc = (x[..., 10] * self.x_upper_bound[10]).exp2()

        # shape: (batch_size,)
        # Approx memory usage on ONE GPU:
        #    a) "Param memory" if the model is sharded by p_tp * p_pp
        #    b) "Activation memory" for local microbatch = global batch / (p_dp * p_pp * p_tp * mb)
        #    c) base_mem for overhead, etc.
        f1 = self.m1.exp() / (p_pp * p_tp)
        f2 = self.m2.exp() / (n_gpus * mb)
        f3 = self.m3.exp()
        max_mem = f1 + f2 + f3
        max_mem_transformed = (max_mem - self.max_mem_GB) / self.max_mem_GB
        # return max_mem_transformed
        # return torch.clip(max_mem_transformed, max=OOM_ADD_FACTOR)
        return OOM_ADD_FACTOR - torch.nn.functional.softplus(
            OOM_ADD_FACTOR - max_mem_transformed,
            beta=20.,
            threshold=0.5,
        )