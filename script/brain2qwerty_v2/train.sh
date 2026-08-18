#!/bin/bash
# Brain2Qwerty V2 训练：手工修改下面变量后直接执行
# 种子可从 rjob 用 -e SEED=xxx 传入，未传时默认 123

source ~/miniconda3/bin/activate brainomni
eval $(curl -s http://deploy.i.h.pjlab.org.cn/infra/scripts/nccl_auto_config.py | python3 - --shell-export)


cd /mnt/shared-storage-user/xiaoqinfan/brain2qwerty

export PYTHONPATH=./
export BRAIN2QWERTY_STUDIES=/mnt/shared-storage-user/xiaoqinfan/SpanishBCBL
export BRAIN2QWERTY_CACHE=/mnt/shared-storage-user/xiaoqinfan/brain2qwerty/.cache/spanishbcbl_meg_v2

RESUME=

export BRAIN2QWERTY_RESULTS=${BRAIN2QWERTY_CACHE}/results/seed_${SEED}

# 防止误覆盖已有训练结果：results 目录已存在时直接退出（不进入 main.py）
if [ -d "${BRAIN2QWERTY_RESULTS}" ]; then
    echo "[brain2qwerty_v2] results directory '${BRAIN2QWERTY_RESULTS}' already exists; exiting."
    exit 1
fi

python -m brain2qwerty_v2.main train --resume=$RESUME --seed=$SEED
