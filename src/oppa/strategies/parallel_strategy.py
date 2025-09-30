from dataclasses import dataclass, field
from typing import Dict, List
import math


CAN_LOG_TRANSFORM = [
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    False,
    False,
    False,
    False,
]

STRATEGY_EMBEDDING_DIM = len(CAN_LOG_TRANSFORM)
STRATEGY_PARALLELISM_DIM_SIZE_INDEXS = slice(0, 6)


@dataclass(order=True)
class ParallelisationStrategy:

    sort_index: int = field(init=False, repr=False)
    num_gpus: int
    num_hosts: int
    dp_size: int
    tp_size: int
    pp_size: int
    sp_size: int
    dp_bucket_size_mb: int
    zero_stage: int  # one of {0, 1, 2, 3}
    zero_bucket_size_mb: int  # need to multiply by 1024**2 to get the correct value to feed into ZERO
    num_microbatches: int
    num_model_chunks: int  # if larger than 1, then do interleaved, otherwise do normal 1f1b
    use_grad_checkpoint: bool
    overlap_communication_for_zero: bool
    overlap_allgather_for_zero: bool
    overlap_p2p_for_pp: bool

    def __post_init__(self):

        assert self.num_gpus > 0
        assert self.num_hosts > 0
        assert self.num_gpus % self.num_hosts == 0, (self.num_gpus, self.num_hosts)

        assert self.dp_size > 0
        assert self.tp_size > 0
        assert self.pp_size > 0
        assert self.sp_size > 0
        assert self.num_gpus == self.dp_size * self.tp_size * self.pp_size * self.sp_size, (self.num_gpus, self.dp_size, self.tp_size, self.pp_size, self.sp_size)
        # assert self.tp_size <= (self.num_gpus // self.num_hosts)  # don't do TP across hosts

        # assert not ((self.sp_size > 1) and (self.tp_size == 1))
        # assert not ((self.sp_size > 1) and (self.pp_size > 1))

        # restrictions for ZeRO optimiser
        # assert not (self.dp_size == 1 and self.dp_bucket_size_mb > 1), (self.dp_size, self.dp_bucket_size_mb)
        # assert not (self.dp_size > 1 and self.dp_bucket_size_mb < 1), (self.dp_size, self.dp_bucket_size_mb)
        assert self.zero_stage in {0, 1, 2, 3}, self.zero_stage
        assert not (self.pp_size > 1 and self.zero_stage == 3), (self.pp_size, self.zero_stage)
        # assert not (self.dp_size == 1 and self.zero_stage > 0), (self.pp_size, self.zero_stage)
        # assert not (self.zero_stage == 0 and self.zero_bucket_size_mb > 1), (self.zero_bucket_size_mb, self.zero_stage)

        # restrictions for pipeline parallelism related hyperparams
        assert self.num_microbatches >= self.pp_size
        assert self.num_model_chunks > 0
        # assert not ((self.num_microbatches > 1) and (self.pp_size == 1))
        # assert not ((self.num_model_chunks > 1) and (self.pp_size == 1))
        
        # just use default value if those parallelism dont apply
        if self.dp_size == 1:
            self.dp_bucket_size_mb = 1
            self.zero_stage = 0
        if self.zero_stage == 0:
            self.overlap_communication_for_zero = False
            self.overlap_allgather_for_zero = False
            self.zero_bucket_size_mb = 1
        if self.pp_size == 1:
            self.overlap_p2p_for_pp = False
            self.num_microbatches = 1
            self.num_model_chunks = 1

        super().__setattr__('sort_index', tuple(self.to_list()))

    def __repr__(self):
        return f'ParaStrategy[{self.to_string()}]'
    
    def __hash__(self):
        return hash(tuple(self.to_list()))

    @classmethod
    def from_dict(cls, data: Dict[str, int]):
        return cls(
            num_gpus=data["num_gpus"],
            num_hosts=data["num_hosts"],
            dp_size=data["dp_size"],
            tp_size=data["tp_size"],
            pp_size=data["pp_size"],
            sp_size=data["sp_size"],
            dp_bucket_size_mb=data["dp_bucket_size_mb"],
            zero_stage=data["zero_stage"],
            zero_bucket_size_mb=data["zero_bucket_size_mb"],
            num_microbatches=data["num_microbatches"],
            num_model_chunks=data["num_model_chunks"],
            use_grad_checkpoint=bool(data["use_grad_checkpoint"]),
            overlap_communication_for_zero=bool(data["overlap_communication_for_zero"]),
            overlap_allgather_for_zero=bool(data["overlap_allgather_for_zero"]),
            overlap_p2p_for_pp=bool(data["overlap_p2p_for_pp"]),
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "num_gpus": self.num_gpus,
            "num_hosts": self.num_hosts,
            "dp_size": self.dp_size,
            "tp_size": self.tp_size,
            "pp_size": self.pp_size,
            "sp_size": self.sp_size,
            "dp_bucket_size_mb": self.dp_bucket_size_mb,
            "zero_stage": self.zero_stage,
            "zero_bucket_size_mb": self.zero_bucket_size_mb,
            "num_microbatches": self.num_microbatches,
            "num_model_chunks": self.num_model_chunks,
            "use_grad_checkpoint": int(self.use_grad_checkpoint),
            "overlap_communication_for_zero": int(self.overlap_communication_for_zero),
            "overlap_allgather_for_zero": int(self.overlap_allgather_for_zero),
            "overlap_p2p_for_pp": int(self.overlap_p2p_for_pp),
        }
    
    @classmethod
    def from_args(cls, args):
        d = {
            "num_gpus": args.num_gpus,
            "num_hosts": args.num_hosts,
            "dp_size": args.dp_size,
            "tp_size": args.tp_size,
            "pp_size": args.pp_size,
            "sp_size": args.sp_size,
            "dp_bucket_size_mb": args.dp_bucket_size_mb,
            "zero_stage": args.zero_stage,
            "zero_bucket_size_mb": args.zero_bucket_size_mb,
            "num_microbatches": args.num_microbatches,
            "num_model_chunks": args.num_model_chunks,
            "use_grad_checkpoint": bool(int(args.use_grad_checkpoint)),
            "overlap_communication_for_zero": bool(int(args.overlap_communication_for_zero)),
            "overlap_allgather_for_zero": bool(int(args.overlap_allgather_for_zero)),
            "overlap_p2p_for_pp": bool(int(args.overlap_p2p_for_pp)),
        }
        return cls.from_dict(d)

    def to_args(self):
        return [
            "--num-gpus", str(self.num_gpus),
            "--num-hosts", str(self.num_hosts),
            "--dp-size", str(self.dp_size),
            "--tp-size", str(self.tp_size),
            "--pp-size", str(self.pp_size),
            "--sp-size", str(self.sp_size),
            "--dp-bucket-size-mb", str(self.dp_bucket_size_mb),
            "--zero-stage", str(self.zero_stage),
            "--zero-bucket-size-mb", str(self.zero_bucket_size_mb),
            "--num-microbatches", str(self.num_microbatches),
            "--num-model-chunks", str(self.num_model_chunks),
            "--use-grad-checkpoint", str(int(self.use_grad_checkpoint)),
            "--overlap-communication-for-zero", str(int(self.overlap_communication_for_zero)),
            "--overlap-allgather-for-zero", str(int(self.overlap_allgather_for_zero)),
            "--overlap-p2p-for-pp", str(int(self.overlap_p2p_for_pp)),
        ]
    
    @classmethod
    def from_string(cls, string: str):
        """Creates an instance of ParallelisationStrategy from a string."""
        try:
            parts = string.split("_")
            data = {
                "num_gpus": int(parts[0][1:]),
                "num_hosts": int(parts[1][1:]),
                "dp_size": int(parts[2][2:]),
                "tp_size": int(parts[3][2:]),
                "pp_size": int(parts[4][2:]),
                "sp_size": int(parts[5][2:]),
                "dp_bucket_size_mb": int(parts[6][2:]),
                "zero_stage": int(parts[7][2:]),
                "zero_bucket_size_mb": int(parts[8][2:]),
                "num_microbatches": int(parts[9][2:]),
                "num_model_chunks": int(parts[10][2:]),
                "use_grad_checkpoint": int(parts[11][2:]),
                "overlap_communication_for_zero": int(parts[12][3:]),
                "overlap_allgather_for_zero": int(parts[13][3:]),
                "overlap_p2p_for_pp": int(parts[14][3:]),
            }
            return cls.from_dict(data)
        except (IndexError, ValueError):
            raise ValueError(f"Invalid string format: {string}")

    def to_string(self) -> str:
        return (
            f"g{self.num_gpus}_h{self.num_hosts}_"
            f"dp{self.dp_size}_tp{self.tp_size}_pp{self.pp_size}_sp{self.sp_size}_"
            f"db{self.dp_bucket_size_mb}_zs{self.zero_stage}_zb{self.zero_bucket_size_mb}_"
            f"mb{self.num_microbatches}_mc{self.num_model_chunks}_gc{int(self.use_grad_checkpoint)}_"
            f"ocz{int(self.overlap_communication_for_zero)}_oaz{int(self.overlap_allgather_for_zero)}_opp{int(self.overlap_p2p_for_pp)}"
        )
    
    @classmethod
    def from_list(cls, data: List[float]):
        num_gpus, num_hosts, dp_size, tp_size, pp_size, sp_size, dpbsz, zs, zbsz, num_mb, num_mc, use_gc, ocz, oaz, opp = data
        # to reduce the dimensions of input by a bit, instead infer dp_size from other args
        # num_gpus, num_hosts, tp_size, pp_size, sp_size, dpbsz, zs, zbsz, num_mb, num_mc, use_gc = data
        # dp_size = num_gpus / (tp_size * pp_size * sp_size)
        has_dp = int(int(round(dp_size)) > 1)
        has_zero = int(int(round(zs) - has_dp) > 0)
        has_pp = int(int(round(pp_size)) > 1)
        d = {
            "num_gpus": int(round(num_gpus)),
            "num_hosts": int(round(num_hosts)),
            "dp_size": int(round(dp_size)),
            "tp_size": int(round(tp_size)),
            "pp_size": int(round(pp_size)),
            "sp_size": int(round(sp_size)),
            "dp_bucket_size_mb": int(round(dpbsz)),
            "zero_stage": int(round(zs)),
            "zero_bucket_size_mb": int(round(zbsz)),
            "num_microbatches": int(round(num_mb)),
            "num_model_chunks": int(round(num_mc)),
            "use_grad_checkpoint": bool(int(round(use_gc))),
            "overlap_communication_for_zero": bool(int(round(ocz))),
            "overlap_allgather_for_zero": bool(int(round(oaz))),
            "overlap_p2p_for_pp": bool(int(round(opp))),
        }
        return cls.from_dict(d)
    
    def to_list(self):
        has_dp = int(self.dp_size > 1)
        has_zero = int(self.zero_stage > 0)
        has_pp = int(self.pp_size > 1)
        x = [
            self.num_gpus,
            self.num_hosts,
            self.dp_size,
            self.tp_size,
            self.pp_size,
            self.sp_size,
            self.dp_bucket_size_mb,
            self.zero_stage,
            self.zero_bucket_size_mb,
            self.num_microbatches,
            self.num_model_chunks,
            (1. if self.use_grad_checkpoint else 0.),  # do this so that the trust region can enclose this dim sometimes
            (1. if self.overlap_communication_for_zero else 0.),
            (1. if self.overlap_allgather_for_zero else 0.),
            (1. if self.overlap_p2p_for_pp else 0.),
        ]
        return [float(i) for i in x]
    
    def to_log_list(self):
        return [(math.log2(x) if y else x) for (x, y) in zip(self.to_list(), CAN_LOG_TRANSFORM)]
    
    @classmethod
    def get_list_log_transform_upper_bound(cls, num_hosts, num_gpus_per_host, batch_size, max_model_chunks, max_dp_bucket_size_mb=2**12):
        # align this with the self.to_list() method
        max_num_gpus = num_hosts * num_gpus_per_host
        bound = [
            max_num_gpus,
            num_hosts,
            max_num_gpus,
            max_num_gpus,
            max_num_gpus,
            max_num_gpus,
            max_dp_bucket_size_mb,
            3,
            max_dp_bucket_size_mb,
            batch_size,
            max_model_chunks,
            1,
            1,
            1,
            1,
        ]
        assert len(CAN_LOG_TRANSFORM) == len(bound)
        transformed_bound = [(math.log2(x) if t else x) for (x, t) in zip(bound, CAN_LOG_TRANSFORM)]
        transformed_bound = [max(1, x) for x in transformed_bound]
        return transformed_bound
    
    @classmethod
    def log_transform(cls, x):
        x1 = x.clone()
        x1[..., CAN_LOG_TRANSFORM] = x1[..., CAN_LOG_TRANSFORM].log2()
        return x1
    
    @classmethod
    def undo_log_transform(cls, x):
        x1 = x.clone()
        x1[..., CAN_LOG_TRANSFORM] = x1[..., CAN_LOG_TRANSFORM].exp2()
        return x1


def check_implementation_specific_strategy(p: ParallelisationStrategy, implementation: str):
    
    if implementation == 'colossal':
        assert p.zero_stage in {0, 1, 2}
        assert p.sp_size == 1
        assert not (p.pp_size > 1 and p.zero_stage in {2, 3})
        
    elif implementation == 'nemo':
        # to restrict irrelevant parameters
        # TODO: incorporate these params in the runner strategy
        assert not p.use_grad_checkpoint
        assert not p.overlap_communication_for_zero
        assert not p.overlap_allgather_for_zero
        assert not p.overlap_p2p_for_pp
        assert p.zero_bucket_size_mb == 1
        assert p.dp_bucket_size_mb == 1
        if (p.zero_stage > 0) and (p.dp_size * p.tp_size == p.num_gpus):
            # to restrict values when using DeepSpeed
            pass
        else:
            # to restrict values when using Megatron
            assert p.zero_stage == 0
            assert not p.overlap_communication_for_zero
            
    elif implementation == 'lightning':
        assert p.pp_size == 1
        assert p.sp_size == 1
        assert not p.use_grad_checkpoint
        assert not p.overlap_communication_for_zero
        assert not p.overlap_allgather_for_zero
        assert not p.overlap_p2p_for_pp
        assert p.dp_bucket_size_mb == 1
        assert p.zero_bucket_size_mb == 1
        
    elif implementation == 'hf':
        assert p.num_hosts == 1
        assert p.dp_size == p.num_gpus
        assert not p.use_grad_checkpoint
        assert not p.overlap_p2p_for_pp
        assert p.dp_bucket_size_mb == 1
        assert p.zero_bucket_size_mb == 1
    
    else:
        raise ValueError(f'Invalid {implementation=}')
