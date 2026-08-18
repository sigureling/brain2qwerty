from pathlib import Path


PRETRAINED_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "resources" / "brainomni_model_state.pt"
)


ENCODER = {
    "name": "BrainOmniCTC",
    "patch_size": 32,
    "n_dim": 512,
    "head_dim": 64,
    "num_queries": 32,
    "qformer_layers": 4,
    "attn_layers": 12,
    "dropout": 0.0,
    "drop_path": 0.1,
    "classifier_head": "rms_linear",
    "classifier_dropout": 0.0,
    "aux_prediction": True,
    "pretrained_checkpoint": str(PRETRAINED_CHECKPOINT),
}
