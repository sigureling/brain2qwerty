#!/bin/bash
# Brain2Qwerty V2 + BrainOmni 训练：手工修改下面变量后直接执行
# 种子可从 rjob 用 -e SEED=xxx 传入，未传时默认 123

source ~/miniconda3/bin/activate brainomni
eval $(curl -s http://deploy.i.h.pjlab.org.cn/infra/scripts/nccl_auto_config.py | python3 - --shell-export)


cd /mnt/shared-storage-user/xiaoqinfan/brain2qwerty

export PYTHONPATH=./
export BRAIN2QWERTY_STUDIES="${BRAIN2QWERTY_STUDIES:-/mnt/shared-storage-user/xiaoqinfan/SpanishBCBL}"
export BRAIN2QWERTY_CACHE="${BRAIN2QWERTY_CACHE:-/mnt/shared-storage-user/xiaoqinfan/brain2qwerty/.cache/spanishbcbl_meg_v2_brainomni}"
SEED="${SEED:-123}"
: "${EXPERIMENT_NAME:?EXPERIMENT_NAME must be provided by rjob_train.py}"

WEIGHT=brain2qwerty_v2_brainomni/resources/brainomni_model_state.pt
SOURCE_WEIGHT=standalone_brainomni/v2_results/brainomni/eeg_meg_ieeg/ps_32_nd_512_nq_32_ql_4_al_12/eeg_meg_ieeg_ps_32_nd_512_nq_32_ql_4_cs_4096_nc_1/checkpoint/best/model_state.pt
if [ ! -f "$WEIGHT" ]; then
    cp "$SOURCE_WEIGHT" "$WEIGHT"
fi

RESUME=

export BRAIN2QWERTY_RESULTS="${BRAIN2QWERTY_RESULTS:-${BRAIN2QWERTY_CACHE}/results/${EXPERIMENT_NAME}}"

# 防止误覆盖已有训练结果：results 目录已存在时直接退出（不进入 main.py）
if [ -d "${BRAIN2QWERTY_RESULTS}" ]; then
    echo "[brain2qwerty_v2_brainomni] results directory '${BRAIN2QWERTY_RESULTS}' already exists; exiting."
    exit 1
fi

TRAIN_ARGS=(train --resume="$RESUME" --seed="$SEED")
if [ "${AUX_PREDICTION:-}" = "true" ]; then
    TRAIN_ARGS+=(--aux-prediction)
elif [ "${AUX_PREDICTION:-}" = "false" ]; then
    TRAIN_ARGS+=(--no-aux-prediction)
fi
[ -n "${CLASSIFIER_HEAD:-}" ] && TRAIN_ARGS+=(--classifier-head "$CLASSIFIER_HEAD")
[ -n "${LR:-}" ] && TRAIN_ARGS+=(--lr "$LR")
[ -n "${WEIGHT_DECAY:-}" ] && TRAIN_ARGS+=(--weight-decay "$WEIGHT_DECAY")

python -m brain2qwerty_v2_brainomni.main "${TRAIN_ARGS[@]}"
