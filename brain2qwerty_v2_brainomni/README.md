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
# 单次实验也可以直接覆盖四个调参项
python -m brain2qwerty_v2_brainomni.main train \
  --lr 5e-4 \
  --weight-decay 1e-3 \
  --no-aux-prediction \
  --classifier-head rms_conv
python -m brain2qwerty_v2_brainomni.main eval \
  --ckpt "$BRAIN2QWERTY_RESULTS/best_ctc.ckpt"

python -m brain2qwerty_v2_brainomni.scripts.extract_predictions \
  --input "$BRAIN2QWERTY_RESULTS/predictions_test.json" --split test
```

Training uses 20 frozen-backbone epochs followed by 275 joint epochs. Stage-one
checkpoints and TensorBoard logs are written below `stage1/`; final artifacts
retain the V2 names in the main result directory. Passing `--resume` resumes
stage two. Set `stage1_epochs=0` to skip the frozen stage.

## Dual-cluster automatic sweep

The launcher runs a two-stage sweep and uses only the shared prediction files
for completion detection:

```bash
python script/brain2qwerty_v2_brainomni/rjob_train.py --auto
```

Stage A submits four head/auxiliary-CTC combinations to both `brainllm` and
`speechllm` (eight jobs). It waits locally for
`$BRAIN2QWERTY_CACHE/results/<EXPERIMENT_NAME>/predictions_test.json`; the JSON
must have non-empty `rows` and a finite `across_subject_cer`. Once all four are
valid, the lowest-CER head/auxiliary pair is selected. Stage B submits its
three learning rates crossed with four weight decays (24 jobs across both
clusters) and exits without monitoring or selecting a final LR/WD pair.

Every run writes a `workflow.json` manifest under the cache. If the launcher is
interrupted, continue stage-A local waiting or retry only stage-B submissions
that were not recorded as successful:

```bash
python script/brain2qwerty_v2_brainomni/rjob_train.py \
  --resume /path/to/workflow.json
```

Use `--dry-run` to print the eight stage-A and 24 stage-B `rjob submit`
commands without submitting jobs or waiting. Its stage-B preview uses the
baseline head/aux pair because no stage-A metrics exist during a dry run.
