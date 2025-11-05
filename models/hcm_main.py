# hcm.py
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn

from models.common import trunc_normal_init_
from models.layers import rms_norm, HCMSwiGLU, HCMAttention, HCMRotaryEmbedding, CosSin
from models.sparse_embedding import HCMSparseEmbedding


@dataclass
class HCM_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HCM_ACTV1Carry:
    inner_carry: HCM_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class HCM_ACTV1Config:
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        num_puzzle_identifiers: int,
        vocab_size: int,
        H_cycles: int,
        L_cycles: int,
        H_layers: int,
        L_layers: int,
        hidden_size: int,
        expansion: float = 8 / 3,
        num_heads: int = 8,
        num_key_value_heads: Optional[int] = None,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10_000.0,
        halt_max_steps: int = 8,
        halt_exploration_prob: float = 0.1,
        forward_dtype: str = "bfloat16",
        puzzle_emb_ndim: int = 0,
        pos_encodings: str = "rope",        # "rope" | "learned"
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_puzzle_identifiers = num_puzzle_identifiers
        self.vocab_size = vocab_size
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.H_layers = H_layers
        self.L_layers = L_layers
        self.hidden_size = hidden_size
        self.expansion = expansion
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob
        self.forward_dtype = getattr(torch, forward_dtype)
        self.puzzle_emb_ndim = puzzle_emb_ndim
        self.pos_encodings = pos_encodings


class HCM_ACTV1Block(nn.Module):
    def __init__(self, config: HCM_ACTV1Config):
        super().__init__()
        self.self_attn = HCMAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=config.hidden_size // config.num_heads,
            num_key_value_heads=config.num_key_value_heads,
            causal=False,
        )
        self.mlp = HCMSwiGLU(config.hidden_size, config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor, cos_sin: Optional[CosSin] = None) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(hidden_states, cos_sin=cos_sin)
        hidden_states = rms_norm(hidden_states, self.norm_eps)
        hidden_states = hidden_states + self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states, self.norm_eps)
        return hidden_states


class HCM_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HCM_ACTV1Block]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, cos_sin: Optional[CosSin] = None) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos_sin=cos_sin)
        return hidden_states


class HCMACTV1Inner(nn.Module):
    def __init__(self, config: HCM_ACTV1Config):
        super().__init__()
        self.config = config

        # ---- 词嵌入 ----
        embed_scale = math.sqrt(config.hidden_size)
        embed_std = 1.0 / embed_scale
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.trunc_normal_(self.embed_tokens.weight, std=embed_std)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = nn.Linear(config.hidden_size, 2, bias=True)

        # ---- 谜题嵌入 ----
        self.puzzle_emb_len = -(config.puzzle_emb_ndim // -config.hidden_size)
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = HCMSparseEmbedding(
                num_embeddings=config.num_puzzle_identifiers,
                embedding_dim=config.puzzle_emb_ndim,
                batch_size=config.batch_size,
                init_std=0.0,
                cast_to=config.forward_dtype,
            )

        # ---- 位置编码 ----
        if config.pos_encodings == "rope":
            self.rotary_emb = HCMRotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.seq_len + self.puzzle_emb_len,
                base=config.rope_theta,
            )
        elif config.pos_encodings == "learned":
            self.pos_emb = nn.Embedding(config.seq_len + self.puzzle_emb_len, config.hidden_size)
            nn.init.trunc_normal_(self.pos_emb.weight, std=embed_std)
        else:
            raise ValueError(f"Unknown pos_encodings: {config.pos_encodings}")

        # ---- 推理层 ----
        self.H_level = HCM_ACTV1ReasoningModule(
            [HCM_ACTV1Block(config) for _ in range(config.H_layers)]
        )
        self.L_level = HCM_ACTV1ReasoningModule(
            [HCM_ACTV1Block(config) for _ in range(config.L_layers)]
        )

        # ---- 初始状态 ----
        self.register_buffer("H_init", trunc_normal_init_(torch.empty(config.hidden_size, dtype=config.forward_dtype), std=1.0), persistent=True)
        self.register_buffer("L_init", trunc_normal_init_(torch.empty(config.hidden_size, dtype=config.forward_dtype), std=1.0), persistent=True)

        # ---- Q-head 零初始化 ----
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)

    def _input_embeddings(self, input_ids: torch.Tensor, puzzle_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)

        if self.config.puzzle_emb_ndim > 0:
            puzzle_emb = self.puzzle_emb(puzzle_ids)
            pad = self.puzzle_emb_len * self.config.hidden_size - puzzle_emb.size(-1)
            if pad > 0:
                puzzle_emb = F.pad(puzzle_emb, (0, pad))
            puzzle_emb = puzzle_emb.view(-1, self.puzzle_emb_len, self.config.hidden_size)
            emb = torch.cat([puzzle_emb, emb], dim=1)

        if self.config.pos_encodings == "learned":
            pos = torch.arange(emb.size(1), device=emb.device, dtype=torch.long)
            emb = emb + self.pos_emb(pos) * 0.707106781  # 1/sqrt(2)

        return emb

    def empty_carry(self, batch_size: int) -> HCM_ACTV1InnerCarry:
        return HCM_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.config.forward_dtype, device=self.H_init.device),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.config.forward_dtype, device=self.L_init.device),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: HCM_ACTV1InnerCarry) -> HCM_ACTV1InnerCarry:
        return HCM_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: HCM_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HCM_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None
        emb = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # —— 无梯度推理循环 ——
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L
            for h_step in range(self.config.H_cycles):
                for l_step in range(self.config.L_cycles):
                    if not (h_step == self.config.H_cycles - 1 and l_step == self.config.L_cycles - 1):
                        z_L = self.L_level(z_L, z_H + emb, cos_sin=cos_sin)
                if h_step != self.config.H_cycles - 1:
                    z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        # —— 1-step 梯度路径 ——
        z_L = self.L_level(z_L, z_H + emb, cos_sin=cos_sin)
        z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        new_carry = HCM_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        logits = self.lm_head(z_H[:, self.puzzle_emb_len:])
        q_logits = self.q_head(z_H[:, 0]).float()
        return new_carry, logits, (q_logits[..., 0], q_logits[..., 1])


class HCMACTV1(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HCM_ACTV1Config(**config_dict)
        self.inner = HCMACTV1Inner(self.config)

    @property
    def puzzle_emb(self) -> Optional[HCMSparseEmbedding]:
        return getattr(self.inner, "puzzle_emb", None)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> HCM_ACTV1Carry:
        bsz = batch["inputs"].size(0)
        return HCM_ACTV1Carry(
            inner_carry=self.inner.empty_carry(bsz),
            steps=torch.zeros(bsz, dtype=torch.int32, device=batch["inputs"].device),
            halted=torch.ones(bsz, dtype=torch.bool, device=batch["inputs"].device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(self, carry: HCM_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HCM_ACTV1Carry, Dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view(-1, *(1,) * (v.ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        new_inner_carry, logits, (q_halt, q_cont) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt,
            "q_continue_logits": q_cont,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last = new_steps >= self.config.halt_max_steps
            halted = is_last

            if self.training and self.config.halt_max_steps > 1:
                halted = halted | (q_halt > q_cont)
                min_halt = (torch.rand_like(q_halt) < self.config.halt_exploration_prob) * \
                           torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt)

                _, (_, next_q_halt, next_q_cont) = self.inner(new_inner_carry, new_current_data)
                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(is_last, next_q_halt, torch.maximum(next_q_halt, next_q_cont))
                )

        new_carry = HCM_ACTV1Carry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            current_data=new_current_data,
        )
        return new_carry, outputs