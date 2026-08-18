import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath
from .init import init_trunc_normal
from .norm import RMSNorm
from .rope import RotaryEmbedding


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        return args


# swiglufused
class FeedForward(nn.Module):
    def __init__(self, n_dim, dropout=0.0):
        super().__init__()
        self.proj_up = nn.Linear(n_dim, 8 * int(n_dim / 3), bias=False)
        self.proj_back = nn.Linear(4 * int(n_dim / 3), n_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.init_weights()

    def init_weights(self):
        init_trunc_normal(self.proj_up)
        init_trunc_normal(self.proj_back)

    def forward(self, x: torch.Tensor):
        gate, data = self.proj_up(x).chunk(chunks=2, dim=-1)
        gate = F.silu(gate)
        return self.dropout(self.proj_back(gate * data))


# Attention
# OK
class SelfAttention(nn.Module):
    def __init__(
        self,
        n_dim: int,
        n_head: int,
        dropout: float,
        rope: bool,
        causal: bool,
        hidden_dim: bool = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim else n_dim
        assert hidden_dim % n_head == 0
        self.rope = rope
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.head_dim = hidden_dim // n_head
        self.causal = causal
        self.q_proj = nn.Linear(n_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(n_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(n_dim, hidden_dim, bias=False)
        self.rope_embedding_layer = (
            RotaryEmbedding(dim=self.head_dim) if self.rope else Identity()
        )
        self.proj = nn.Linear(hidden_dim, n_dim)
        self.init_weights()

    def init_weights(self):
        init_trunc_normal(self.q_proj)
        init_trunc_normal(self.k_proj)
        init_trunc_normal(self.v_proj)
        init_trunc_normal(self.proj)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        True element in mask will take part in attention
        x    B T C
        mask B T T
        """
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "B T (H D) -> B H T D", H=self.n_head)
        k = rearrange(k, "B T (H D) -> B H T D", H=self.n_head)
        v = rearrange(v, "B T (H D) -> B H T D", H=self.n_head)
        if self.rope:
            q, k = self.rope_embedding_layer(q, k)

        # add head_dim
        if mask is not None and mask.ndim == 3:
            mask = mask.unsqueeze(1)

        output = (
            F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
            )
            .transpose(1, 2)
            .contiguous()
            .view(B, T, -1)
        )

        return self.proj(output)


# OK
class SelfAttnBlock(nn.Module):
    def __init__(
        self,
        n_dim: int,
        n_head: int,
        dropout: float,
        drop_path: float,
        rope: bool,
        causal: bool,
    ):
        super().__init__()
        self.pre_attn_norm = RMSNorm(n_dim)
        self.attn = SelfAttention(
            n_dim=n_dim,
            n_head=n_head,
            dropout=dropout,
            rope=rope,
            causal=causal,
        )
        self.pre_ff_norm = RMSNorm(n_dim)
        self.ff = FeedForward(n_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        True element in mask will take part in attention
        x B T C
        mask B T T
        """
        x = x + self.drop_path(self.attn(self.pre_attn_norm(x), mask=mask))
        x = x + self.drop_path(self.ff(self.pre_ff_norm(x)))
        return x
