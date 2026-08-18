import torch.nn as nn


def init_trunc_normal(module: nn.Module, std: float = 0.02):
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=std)
