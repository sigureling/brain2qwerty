import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.RMSNorm):
    def __init__(self, n_dim, elementwise_affine=True, eps=1e-6):
        super().__init__(n_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        weight = self.weight
        if weight is not None:
            if weight.dtype != x.dtype:
                weight = weight.to(x.dtype)
        return F.rms_norm(x, self.normalized_shape, weight, self.eps)


class LayerNorm(nn.LayerNorm):
    def __init__(self, n_dim, elementwise_affine=True, eps=1e-6):
        super().__init__(n_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        weight, bias = self.weight, self.bias
        if weight is not None:
            if weight.dtype != x.dtype:
                weight = weight.to(x.dtype)
        if bias is not None:
            if bias.dtype != x.dtype:
                bias = bias.to(x.dtype)
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
