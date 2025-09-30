from typing import Dict
import torch
from botorch.optim import optimize_acqf_discrete
from botorch.models import SingleTaskGP, ModelListGP
from botorch.acquisition import UpperConfidenceBound, LogExpectedImprovement, qLogExpectedImprovement, qUpperConfidenceBound
from botorch.optim import optimize_acqf_discrete, optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.objective import GenericMCObjective, ConstrainedMCObjective


def choose_next_point(
    model_objective: SingleTaskGP, model_constraint: SingleTaskGP = None,
    acq_fn: str = 'ucb', acq_fn_args: Dict = dict(),
    acq_candidates: torch.Tensor = None, acq_opt_args: Dict = dict(),
):
    
    if model_constraint is not None:
    
        if acq_fn == 'ei':
            f = qLogExpectedImprovement(
                model=ModelListGP(model_objective, model_constraint), 
                best_f=acq_fn_args['best_f'], 
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([acq_fn_args.get('sampler_num', 4096)])),
                objective=GenericMCObjective(lambda Z, X: Z[..., 0]),
                constraints=[lambda Z: Z[..., 1]]
            )
            
        elif acq_fn == 'ucb':
            f = qUpperConfidenceBound(
                model=ModelListGP(model_objective, model_constraint), 
                beta=acq_fn_args.get('beta', 1.), 
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([acq_fn_args.get('sampler_num', 4096)])),
                objective=ConstrainedMCObjective(
                    objective=lambda Z, X: Z[..., 0],
                    constraints=[lambda Z: Z[..., 1]],
                    infeasible_cost=0.,
                )
            )
        else:
            raise ValueError(f'Invalid {acq_fn=}')
        
    else:
        
        if acq_fn == 'ei':
            f = LogExpectedImprovement(
                model=model_objective, 
                best_f=acq_fn_args['best_f'], 
                maximize=True,
            )
            
        elif acq_fn == 'ucb':
            f = UpperConfidenceBound(
                model=model_objective, 
                beta=acq_fn_args.get('beta', 1.),
                maximize=True,
            )
            
        else:
            raise ValueError(f'Invalid {self.acq_fn=}')
        
    if acq_candidates is not None:
        x_next, acq_score = optimize_acqf_discrete(
            acq_function=f,
            q=1,
            choices=acq_candidates,
        )
        x_next = x_next[0]
        
    else:
        x_next, acq_score = optimize_acqf(
            acq_function=f,
            bounds=acq_opt_args['bounds'],
            q=1,
            return_best_only=True,
            equality_constraints=acq_opt_args.get('equality_constraints', None),
            inequality_constraints=acq_opt_args.get('inequality_constraints', None),
            num_restarts=acq_opt_args.get('num_restarts', 8),
            raw_samples=acq_opt_args.get('raw_samples', 128),
        )
        x_next = x_next[0]
        
    return x_next, acq_score.tolist()
