# Brain2Qwerty V2 + BrainOmni

This package keeps the sentence-level SpanishBCBL CTC pipeline from
`brain2qwerty_v2` and replaces its ConvConformer encoder with the pretrained
BrainOmni backbone.

## Model

- Input MEG is sampled at 256 Hz and encoded by BrainOmni with `patch_size=32`
  and fixed `stride=16`, producing approximately 16 feature frames per second.
- Sensor input contains the six BrainOmni position/direction values and the
  `MAG=1`, `GRAD=2` sensor code.
- A learned subject embedding is added before the spatial qformer.
- `classifier_head` selects `rms_linear` or `rms_conv`.
- `aux_prediction=True` enables the spatial-qformer CTC head, its feedback into
  the temporal encoder, and the blended auxiliary/final CTC objective. Setting
  it to `False` leaves only the final CTC path.

The pretrained file is expected at
`resources/brainomni_model_state.pt`. It is intentionally ignored by Git. The
training script copies it from the local `standalone_brainomni` reference when
needed.

## Data

The event transform, sentence windows, typed targets, and text-based split are
identical to `brain2qwerty_v2`. Waveforms use a 0.5–85 Hz band-pass, `[50, 60]`
notch configuration, 256 Hz resampling, and sample-level per-channel z-score.
The default cache is `.cache/spanishbcbl_meg_v2_brainomni`.

The reference SpanishBCBL event contract is 5,147 sentences from 19 subjects
and 128 unique texts, split into 4,145 train, 562 validation, and 440 test
samples. CTC feasibility uses the strict minimum `target length + adjacent
repeats`; no sample is filtered or silently assigned zero loss.

## Running

```bash
bash script/brain2qwerty_v2_brainomni/preprocess.sh --debug
bash script/brain2qwerty_v2_brainomni/preprocess.sh

python -m brain2qwerty_v2_brainomni.main debug
python -m brain2qwerty_v2_brainomni.main train
python -m brain2qwerty_v2_brainomni.main eval \
  --ckpt "$BRAIN2QWERTY_RESULTS/best_ctc.ckpt"

python -m brain2qwerty_v2_brainomni.scripts.extract_predictions \
  --input "$BRAIN2QWERTY_RESULTS/predictions_test.json" --split test
```

Training uses 20 frozen-backbone epochs followed by 275 joint epochs. Stage-one
checkpoints and TensorBoard logs are written below `stage1/`; final artifacts
retain the V2 names in the main result directory. Passing `--resume` resumes
stage two. Set `stage1_epochs=0` to skip the frozen stage.
