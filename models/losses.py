# losses.py
from typing import Any, Tuple, Dict, Sequence, Optional
import torch
import torch.nn.functional as F
from torch import nn

IGNORE_LABEL_ID = -100


def _stable_sigmoid(x: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    return torch.where(x < 0,
                       1 / (1 - x + eps),
                       x + 1)


def _log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    s_x = _stable_sigmoid(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits: torch.Tensor,
                            labels: torch.Tensor,
                            ignore_index: int = -100) -> torch.Tensor:
    logprobs = _log_stablemax(logits.to(torch.float32), dim=-1)
    mask = labels != ignore_index
    labels_clamp = torch.where(mask, labels, 0)
    token_loss = -torch.gather(logprobs, dim=-1,
                               index=labels_clamp.unsqueeze(-1)).squeeze(-1)
    return torch.where(mask, token_loss, 0.0)


def softmax_cross_entropy(logits: torch.Tensor,
                          labels: torch.Tensor,
                          ignore_index: int = -100) -> torch.Tensor:
    """保持接口与 transformers 一致，支持 float16/bfloat16 输入"""
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.size(-1)),
                           labels.to(torch.long).view(-1),
                           ignore_index=ignore_index,
                           reduction="none").view(labels.shape)


class HCMLossHead(nn.Module):
    """HCM 统一损失头，命名与 ZERO2LLM 对齐"""
    model: nn.Module
    loss_fn: Any

    def __init__(self, model: nn.Module, loss_type: str = "softmax_cross_entropy"):
        super().__init__()
        self.model = model
        # 支持通过字符串指定损失函数
        self.loss_fn = globals()[loss_type]

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor],
               Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        """
        返回值：
        new_carry, total_loss, metrics, detached_outputs, all_halted
        """
        new_carry, outputs = self.model(**model_kwargs)
        labels: torch.Tensor = new_carry.current_data["labels"]

        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1, keepdim=True).clamp_min(1)

            pred_ids = torch.argmax(outputs["logits"], dim=-1)
            is_correct = mask & (pred_ids == labels)
            seq_is_correct = is_correct.sum(-1) == mask.sum(-1)

            valid = new_carry.halted & mask.sum(-1).bool()
            metrics: Dict[str, torch.Tensor] = {
                "count": valid.sum(),

                "accuracy":
                    torch.where(valid,
                                (is_correct.float().sum(-1) / loss_counts.squeeze(-1)).sum(),
                                0).sum(),

                "exact_accuracy":
                    (valid & seq_is_correct).sum(),

                "q_halt_accuracy":
                    (valid &
                     ((outputs["q_halt_logits"] >= 0).squeeze(-1) == seq_is_correct)).sum(),

                "steps":
                    torch.where(valid, new_carry.steps, 0).sum(),
            }

        # --- 损失计算 ---
        lm_loss = (self.loss_fn(outputs["logits"], labels,
                                ignore_index=IGNORE_LABEL_ID).sum(-1, keepdim=True)
                   / loss_counts).sum()

        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"].squeeze(-1),
            seq_is_correct.float(),
            reduction="sum"
        )

        total_loss = lm_loss + 0.5 * q_halt_loss

        # --- 可选 Q-continue 损失 ---
        if "target_q_continue" in outputs:
            q_cont_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="sum"
            )
            total_loss = total_loss + 0.5 * q_cont_loss
            metrics["q_continue_loss"] = q_cont_loss.detach()

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # --- 按需返回输出 ---
        detached_outputs = {k: outputs[k].detach()
                            for k in return_keys if k in outputs}

        return (new_carry,
                total_loss,
                metrics,
                detached_outputs,
                new_carry.halted.all())