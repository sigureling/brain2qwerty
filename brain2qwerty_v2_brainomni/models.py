import typing as tp
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from neuraltrain.models.base import BaseModelConfig

from .model_utils.init import init_trunc_normal
from .model_utils.module import BrainEncoder, SensorEncoder
from .model_utils.norm import RMSNorm


class BrainOmniBackbone(nn.Module):
    """BrainOmni backbone with a fixed 16 Hz overlapping patch encoder."""

    def __init__(
        self,
        *,
        patch_size: int,
        n_dim: int,
        head_dim: int,
        num_queries: int,
        qformer_layers: int,
        attn_layers: int,
        dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_size // 2
        self.num_queries = num_queries
        self.n_dim = n_dim
        self.patch_encoder = nn.Sequential(
            nn.ZeroPad1d(padding=(0, patch_size // 8)),
            nn.Conv1d(
                in_channels=1,
                out_channels=n_dim,
                kernel_size=patch_size + patch_size // 8,
                stride=self.patch_stride,
                bias=False,
            ),
        )
        self.sensor_encoder = SensorEncoder(n_dim)
        self.brain_encoder = BrainEncoder(
            n_dim=n_dim,
            head_dim=head_dim,
            num_queries=num_queries,
            dropout=dropout,
            drop_path=drop_path,
            qformer_layers=qformer_layers,
            attn_layers=attn_layers,
            use_cls=True,
        )
        init_trunc_normal(self.patch_encoder[1])

    def preprocess_padding(self, x: torch.Tensor) -> torch.Tensor:
        remainder = x.shape[-1] % self.patch_size
        if remainder == 0:
            return x
        return F.pad(x, (0, self.patch_size - remainder))

    def spatial_forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        subject_embedding: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, channels, _ = x.shape
        x = self.preprocess_padding(x)
        x = rearrange(
            self.patch_encoder(rearrange(x, "b c t -> (b c) 1 t")),
            "(b c) d w -> b c w d",
            b=batch_size,
            c=channels,
        )
        sensor_embedding = self.sensor_encoder(pos, sensor_type).unsqueeze(2)
        x = x + sensor_embedding + subject_embedding[:, None, None, :]
        return self.brain_encoder.spatial_forward(x, mask=mask)

    def temporal_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.brain_encoder.temporal_forward(x)


class RMSLinearCTCHead(nn.Module):
    def __init__(self, feature_dim: int, n_outputs: int, dropout: float) -> None:
        super().__init__()
        self.norm = RMSNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(feature_dim, n_outputs)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b q w d -> b w (q d)")
        return self.dropout(self.norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self._features(x))


class RMSConvCTCHead(RMSLinearCTCHead):
    def __init__(self, feature_dim: int, n_outputs: int, dropout: float) -> None:
        super().__init__(feature_dim, n_outputs, dropout)
        self.temporal_conv = nn.Conv1d(
            feature_dim,
            feature_dim,
            kernel_size=3,
            padding=1,
            groups=feature_dim,
        )
        init_trunc_normal(self.temporal_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._features(x)
        residual = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        return self.output(x + residual)


class BrainOmniCTCModel(nn.Module):
    """BrainOmni plus subject conditioning and auxiliary/final CTC heads."""

    def __init__(
        self,
        backbone: BrainOmniBackbone,
        *,
        n_outputs: int,
        n_subjects: int,
        classifier_head: tp.Literal["rms_linear", "rms_conv"],
        classifier_dropout: float,
        aux_prediction: bool,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.aux_prediction = aux_prediction
        self.frozen = False
        self.subject_embedding = nn.Embedding(n_subjects, backbone.n_dim)
        init_trunc_normal(self.subject_embedding)

        feature_dim = backbone.num_queries * backbone.n_dim
        head_cls = {
            "rms_linear": RMSLinearCTCHead,
            "rms_conv": RMSConvCTCHead,
        }[classifier_head]
        self.ctc_head = head_cls(feature_dim, n_outputs, classifier_dropout)
        if aux_prediction:
            self.aux_projection = nn.Linear(n_outputs, feature_dim)
            nn.init.xavier_uniform_(self.aux_projection.weight)
            nn.init.zeros_(self.aux_projection.bias)

    def freeze_backbone(self) -> None:
        self.frozen = True
        self.backbone.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        self.frozen = False
        self.backbone.requires_grad_(True)

    def train(self, mode: bool = True) -> "BrainOmniCTCModel":
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def compute_output_lens(self, neuro_sizes: torch.Tensor) -> torch.Tensor:
        n_patches = torch.div(
            neuro_sizes + self.backbone.patch_size - 1,
            self.backbone.patch_size,
            rounding_mode="floor",
        )
        steps_per_patch = self.backbone.patch_size // self.backbone.patch_stride
        return (n_patches - 1) * steps_per_patch + 1

    def forward(
        self,
        x: torch.Tensor,
        day_idx: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = x.transpose(1, 2)
        subject_embedding = self.subject_embedding(day_idx.reshape(-1))
        spatial = self.backbone.spatial_forward(
            x,
            pos,
            sensor_type,
            subject_embedding,
            mask=mask,
        )

        output: dict[str, torch.Tensor] = {}
        if self.aux_prediction:
            z_aux = self.ctc_head(spatial)
            feedback = self.aux_projection(torch.softmax(z_aux, dim=-1))
            spatial = spatial + rearrange(
                feedback,
                "b w (q d) -> b q w d",
                q=self.backbone.num_queries,
            )
            output["z_aux"] = z_aux

        temporal = self.backbone.temporal_forward(spatial)
        temporal = temporal[:, :, 1:, :]
        output["c_out"] = self.ctc_head(temporal)
        return output


class BrainOmniCTC(BaseModelConfig):
    patch_size: int
    n_dim: int
    head_dim: int
    num_queries: int
    qformer_layers: int
    attn_layers: int
    dropout: float
    drop_path: float
    classifier_head: tp.Literal["rms_linear", "rms_conv"] = "rms_linear"
    classifier_dropout: float = 0.0
    aux_prediction: bool = True
    pretrained_checkpoint: str | None = None

    def build(
        self,
        n_in_channels: int,
        n_outputs: int,
        n_subjects: int,
    ) -> BrainOmniCTCModel:
        del n_in_channels
        backbone = BrainOmniBackbone(
            patch_size=self.patch_size,
            n_dim=self.n_dim,
            head_dim=self.head_dim,
            num_queries=self.num_queries,
            qformer_layers=self.qformer_layers,
            attn_layers=self.attn_layers,
            dropout=self.dropout,
            drop_path=self.drop_path,
        )
        if self.pretrained_checkpoint is not None:
            checkpoint = torch.load(
                Path(self.pretrained_checkpoint), map_location="cpu", weights_only=True
            )
            backbone.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
        return BrainOmniCTCModel(
            backbone,
            n_outputs=n_outputs,
            n_subjects=n_subjects,
            classifier_head=self.classifier_head,
            classifier_dropout=self.classifier_dropout,
            aux_prediction=self.aux_prediction,
        )
