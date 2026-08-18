import torch
import torch.nn as nn
from einops import rearrange
from typing import Iterable, List
from .init import init_trunc_normal
from .norm import LayerNorm
from .attn import SelfAttnBlock, FeedForward


class SensorEncoder(nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.pos_embedding_layer = nn.Sequential(
            nn.Linear(6, n_dim // 2, bias=False),
            LayerNorm(n_dim // 2),
            nn.GELU(),
            nn.Linear(n_dim // 2, n_dim, bias=False),
        )
        self.sensor_embedding_layer = nn.Embedding(5, n_dim)
        self.aggregate_mlp = FeedForward(n_dim, dropout=0.0)
        self.norm = LayerNorm(n_dim)
        self.init_weights()

    def init_weights(self):
        init_trunc_normal(self.pos_embedding_layer[0])
        init_trunc_normal(self.pos_embedding_layer[3])
        init_trunc_normal(self.sensor_embedding_layer)

    def preprocess(self, pos: torch.Tensor, sensor_type: torch.Tensor) -> torch.Tensor:
        """Center and scale positions independently for EEG, MEG, and iEEG.

        For each sensor group, the xyz coordinates are centered by their
        channel-wise mean and divided by the group's RMS xyz radius. The
        direction vectors in the last three values are left unchanged.
        """

        def normalize_group(position: torch.Tensor, mask: torch.Tensor):
            mask = mask.unsqueeze(-1)
            count = mask.sum(dim=-2, keepdim=True).clamp_min(1)
            mean = (position * mask).sum(dim=-2, keepdim=True) / count
            centered = position - mean

            sensor_mask = mask.squeeze(-1)
            radius = centered.square().sum(dim=-1)
            radius = radius.masked_fill(~sensor_mask, 0)
            count = sensor_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            radius = (radius.sum(dim=-1, keepdim=True) / count + 1e-12).sqrt()
            normalized = centered / radius.unsqueeze(-1)
            return torch.where(mask, normalized, position)

        position = pos[..., :3]
        for mask in (
            sensor_type == 0,  # EEG
            (sensor_type == 1) | (sensor_type == 2),  # MEG
            (sensor_type == 3) | (sensor_type == 4),  # iEEG
        ):
            if mask.any():
                position = normalize_group(position, mask)

        return torch.cat((position, pos[..., 3:]), dim=-1)

    def forward(self, pos: torch.Tensor, sensor_type: torch.Tensor):
        """
        pos,dir         B C 6
        sensor_type     B C
        SENSOR_TYPE_DICT = {"EEG": 0, "MAG": 1, "GRAD": 2, "ECOG": 3, "SEEG": 4}
        """
        pos = self.preprocess(pos, sensor_type)
        x = self.pos_embedding_layer(pos) + self.sensor_embedding_layer(
            sensor_type
        ).type_as(pos)
        x = x + self.aggregate_mlp(x)
        return self.norm(x)


class BrainEncoder(nn.Module):
    def __init__(
        self,
        n_dim: int,
        head_dim: int,
        num_queries: int,
        dropout: float,
        drop_path: float,
        qformer_layers: int,
        attn_layers: int,
        use_cls: bool,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.use_cls = use_cls
        self.queries = nn.Parameter(
            data=torch.empty(size=(num_queries, n_dim)),
            requires_grad=True,
        )
        self.qformer_blocks: List[SelfAttnBlock] = nn.ModuleList(
            [
                SelfAttnBlock(
                    n_dim=n_dim,
                    n_head=n_dim // head_dim,
                    dropout=dropout,
                    drop_path=drop_path,
                    rope=False,
                    causal=False,
                )
                for i in range(qformer_layers)
            ]
        )

        if use_cls:
            self.cls_token = nn.Parameter(
                data=torch.empty(size=(1, num_queries, 1, n_dim))
            )

        self.attn_blocks: List[SelfAttnBlock] = nn.ModuleList(
            [
                SelfAttnBlock(
                    n_dim=n_dim,
                    n_head=n_dim // head_dim,
                    dropout=dropout,
                    drop_path=drop_path,
                    rope=True,
                    causal=False,
                )
                for _ in range(attn_layers)
            ]
        )
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.queries, std=0.02)
        if self.use_cls:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def spatial_forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x B C W D
        mask B C W
        """
        B, C, W, D = x.shape
        x = rearrange(x, "B C W D -> (B W) C D")
        queries = self.queries.type_as(x).unsqueeze(0).expand(B * W, -1, -1)
        x = torch.cat([queries, x], dim=1)  # (B W) (Q+C) D

        if mask != None:
            mask = rearrange(mask, "B C W -> (B W) C")
            mask = mask.unsqueeze(1).expand(-1, self.num_queries + C, -1)  # BW Q+C C
            mask = torch.cat(
                [
                    torch.ones(
                        size=(B * W, self.num_queries + C, self.num_queries),
                        dtype=torch.bool,
                        device=x.device,
                    ),
                    mask,
                ],
                dim=-1,
            )  # BN Q+C Q+C
        else:
            mask = None

        for i, block in enumerate(self.qformer_blocks):
            x = block(x, mask)

        x = x[:, : self.num_queries]  # BW Q D

        return rearrange(x, "(B W) Q D -> B Q W D", B=B)

    def temporal_forward(self, x: torch.Tensor):
        B = x.shape[0]
        if self.use_cls:
            cls_token = self.cls_token.type_as(x).expand(B, -1, -1, -1)
            x = torch.cat([cls_token, x], dim=2)

        x = rearrange(x, "B Q W D -> (B Q) W D")
        for block in self.attn_blocks:
            x = block(x)
        x = rearrange(x, "(B Q) W D -> B Q W D", B=B)
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        x = self.spatial_forward(x, mask=mask)
        return self.temporal_forward(x)


class BrainDecoder(nn.Module):
    def __init__(
        self,
        n_dim,
        head_dim,
        dropout,
        drop_path,
        qformer_layers,
        attn_layers,
    ):
        super().__init__()
        assert qformer_layers > 1
        self.attn_blocks = nn.ModuleList(
            [
                SelfAttnBlock(
                    n_dim=n_dim,
                    n_head=n_dim // head_dim,
                    dropout=dropout,
                    drop_path=drop_path,
                    rope=True,
                    causal=False,
                )
                for _ in range(attn_layers)
            ]
        )
        self.qformer_blocks: Iterable[SelfAttnBlock] = nn.ModuleList(
            [
                SelfAttnBlock(
                    n_dim=n_dim,
                    n_head=n_dim // head_dim,
                    dropout=dropout,
                    drop_path=drop_path,
                    rope=False,
                    causal=False,
                )
                for i in range(qformer_layers)
            ]
        )

    def forward(self, x: torch.Tensor, spatial_queries: torch.Tensor):
        """
        x B Q W D
        spatail_queries B C W D
        """
        B, C, W, D = spatial_queries.shape
        x = rearrange(x, "B Q W D -> (B Q) W D")
        for block in self.attn_blocks:
            x = block(x)
        x = rearrange(x, "(B Q) W D -> B Q W D", B=B)
        B, Q, W, D = x.shape
        x = torch.cat([x, spatial_queries], dim=1)  # B Q+C W D
        x = rearrange(x, "B C W D -> (B W) C D")
        mask = torch.ones(size=(B * W, Q + C, Q + C), dtype=torch.bool, device=x.device)
        mask[:, :, Q:] = False
        mask[:, Q:, Q:] = mask[:, Q:, Q:] | torch.eye(
            C, dtype=torch.bool, device=x.device
        ).unsqueeze(0)
        for i, block in enumerate(self.qformer_blocks):
            x = block(x, mask)
        x = rearrange(x[:, Q:], "(B W) C D -> B C W D", B=B)
        return x


class OffsetInfoDecomposer(nn.Module):
    def __init__(self, num_queries, n_dim):
        super().__init__()
        self.centers = nn.Parameter(
            data=torch.zeros(size=(1, num_queries, 1, n_dim)), requires_grad=True
        )

    def forward(self, x: torch.Tensor):
        mu = self.centers.to(x.dtype)
        return x - mu, mu
