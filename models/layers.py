# layers.py 
from typing import Tuple, Optional
import math
import torch
from torch import nn
import torch.nn.functional as F

try:
    from flash_attn_interface import flash_attn_func  # type: ignore[import]
except ImportError:
    # Fallback to FlashAttention 2
    from flash_attn import flash_attn_func  # type: ignore[import]

def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    with torch.no_grad():
        tensor.normal_(0, std)
        tensor.clamp_(a_min=-2*std, a_max=2*std)
    return tensor


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _make_divisible(x: int, div: int = 256) -> int:
    
    return div * ((x + div - 1) // div)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor,
                         unsqueeze_dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 1e6, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self) -> CosSin:
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 head_dim: int,
                 num_key_value_heads: int,
                 causal: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.n_rep = self.num_heads // self.num_key_value_heads
        self.causal = causal
        self.flash = flash_attn_func is not None

        qkv_total = self.num_heads + 2 * self.num_key_value_heads
        self.qkv_proj = nn.Linear(self.hidden_size, qkv_total * self.head_dim, bias=False)
        self.o_proj   = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        nn.init.kaiming_uniform_(self.qkv_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.o_proj.weight, a=math.sqrt(5))

    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        B, S, H, D = x.shape
        x = x[:, :, :, None, :].expand(B, S, H, self.n_rep, D).reshape(B, S, H * self.n_rep, D)
        return x

    def forward(self,
                hidden_states: torch.Tensor,
                cos_sin: Optional[CosSin] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(B, S, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query = qkv[:, :, :self.num_heads]
        key   = qkv[:, :, self.num_heads:self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos[:S], sin[:S])

        if self.flash and S > 1:
            # FlashAttention 需要 (B, S, H, D)
            out = flash_attn_func(query, key, value, causal=self.causal, dropout_p=0.0)
            if isinstance(out, tuple):
                out = out[0]
        else:
            query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            key, value = self.repeat_kv(key), self.repeat_kv(value)
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal_mask = torch.triu(torch.full((S, S), float("-inf"), device=scores.device), diagonal=1)
            scores += causal_mask
            if attention_mask is not None:
                scores += attention_mask
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, value).transpose(1, 2)

        out = out.reshape(B, S, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float = 8/3):
        super().__init__()
        inter = _make_divisible(round(expansion * hidden_size * 2 / 3))
        self.gate_up_proj = nn.Linear(hidden_size, inter * 2, bias=False)
        self.down_proj    = nn.Linear(inter, hidden_size, bias=False)
        nn.init.kaiming_uniform_(self.gate_up_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)