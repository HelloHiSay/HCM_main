# sparse_embedding.py
from typing import Union
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, ParamsT


class SparseEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 batch_size: int,
                 init_std: float = 0.02,
                 cast_to: torch.dtype = torch.bfloat16):
        super().__init__()
        self.cast_to = cast_to

        # 主权重（持久化）
        weight = torch.empty(num_embeddings, embedding_dim)
            # 统一 truncated normal（与 ZERO2LLM 一致）
        nn.init.trunc_normal_(weight, mean=0.0, std=init_std, a=-2*init_std, b=2*init_std)
        self.register_buffer("weight", weight, persistent=True)

        # 训练时局部缓存（非持久化，带梯度）
        self.register_buffer("local_weight",
                             torch.zeros(batch_size, embedding_dim,
                                         requires_grad=True),
                             persistent=False)
        self.register_buffer("local_ids",
                             torch.zeros(batch_size, dtype=torch.int32),
                             persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.training:
            # 推理阶段直接查表
            return F.embedding(input_ids, self.weight.to(self.cast_to))

        # 训练阶段：将权重拷贝到局部 buffer 以触发梯度
        with torch.no_grad():
            self.local_weight.copy_(self.weight[input_ids])
            self.local_ids.copy_(input_ids)
        return self.local_weight.to(self.cast_to)


class SparseEmbeddingSignSGD(Optimizer):
    """
    SignSGD 优化器，专为 HCMSparseEmbedding 设计
    支持分布式 all‑gather，接口风格与 transformers AdamW 对齐
    """
    def __init__(self,
                 params: ParamsT,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-2,
                 world_size: int = 1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, weight_decay=weight_decay, world_size=world_size)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            world_size = group["world_size"]
            lr = group["lr"]
            wd = group["weight_decay"]

            # 解析 param_group 中 3 个 buffer
            local_grad = None
            local_ids = None
            master_weight = None
            for p in group["params"]:
                if p.requires_grad:                 # 带梯度的即为 local_weight
                    local_grad = p.grad
                elif p.ndim == 1:                   # local_ids
                    local_ids = p
                elif p.ndim == 2:                   # master weight
                    master_weight = p
                else:
                    raise RuntimeError("Unexpected param shape in HCMSparseEmbeddingSignSGD")

            if local_grad is None or local_ids is None or master_weight is None:
                continue

            # 分布式 all‑gather
            if world_size > 1:
                N, D = local_grad.shape
                all_grad = torch.empty(world_size * N, D,
                                       dtype=local_grad.dtype,
                                       device=local_grad.device)
                all_ids = torch.empty(world_size * N,
                                      dtype=local_ids.dtype,
                                      device=local_ids.device)
                dist.all_gather_into_tensor(all_grad, local_grad)
                dist.all_gather_into_tensor(all_ids, local_ids)
            else:
                all_grad, all_ids = local_grad, local_ids

            # 按 ID 聚合梯度
            uniq_ids, inv_idx = all_ids.unique(return_inverse=True)
            agg_grad = torch.zeros(uniq_ids.size(0), all_grad.size(1),
                                   dtype=all_grad.dtype, device=all_grad.device)
            agg_grad.scatter_add_(0,
                                  inv_idx.unsqueeze(1).expand_as(all_grad),
                                  all_grad)

            # SignSGD + 解耦权重衰减
            p = master_weight[uniq_ids]
            p.mul_(1.0 - lr * wd).add_(torch.sign(agg_grad), alpha=-lr)
            master_weight[uniq_ids] = p