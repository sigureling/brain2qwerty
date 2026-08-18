import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, init_seq_len=320, base=10000):
        super().__init__()
        assert dim % 2 == 0
        self.register_buffer(
            "freqs",
            1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
            persistent=False,
        )
        self._set_rotate_cache(init_seq_len)

    def _set_rotate_cache(self, seq_len):
        self.max_seq_len_cache = seq_len
        t = torch.arange(seq_len, device=self.freqs.device, dtype=self.freqs.dtype)
        rotate = torch.outer(t, self.freqs).float()
        self.register_buffer(
            "rotate", torch.polar(torch.ones_like(rotate), rotate), persistent=False
        )

    def reshape_for_broadcast(self, x: torch.Tensor):
        """
        x      Batch n_head seq d_head
        rotate seq dim
        """
        _, H, T, D = x.shape
        if T > self.max_seq_len_cache:
            self._set_rotate_cache(T)
        rotate = self.rotate[:T, :]
        assert D == rotate.shape[1]
        return rotate.view(1, 1, T, D)

    def forward(self, q, k):
        assert len(q.shape) == len(k.shape) == 4
        q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        rotate = self.reshape_for_broadcast(q_)
        q_out = torch.view_as_real(q_ * rotate).flatten(-2)
        k_out = torch.view_as_real(k_ * rotate).flatten(-2)
        return q_out.type_as(q), k_out.type_as(k)
