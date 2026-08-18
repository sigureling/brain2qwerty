#!/usr/bin/env bash
# Brain2Qwerty V2 + BrainOmni 数据预处理（预加热特征缓存）
#
# 用法：
#   ./preprocess.sh           # 全量预处理（84 条 timeline，CPU 密集，耗时较长）
#   ./preprocess.sh --debug   # 只处理 timeline 0，快速验证
#
# 可覆盖的环境变量（默认值见下）：
#   BRAIN2QWERTY_STUDIES  原始数据集根目录
#   BRAIN2QWERTY_CACHE    预处理特征缓存目录
#   BRAIN2QWERTY_RESULTS  训练/评估产物目录

set -euo pipefail

cd /mnt/shared-storage-user/xiaoqinfan/brain2qwerty

# 默认路径
export BRAIN2QWERTY_STUDIES="${BRAIN2QWERTY_STUDIES:-/mnt/shared-storage-user/xiaoqinfan/SpanishBCBL}"
export BRAIN2QWERTY_CACHE="${BRAIN2QWERTY_CACHE:-/mnt/shared-storage-user/xiaoqinfan/brain2qwerty/.cache/spanishbcbl_meg_v2_brainomni}"
export BRAIN2QWERTY_RESULTS="${BRAIN2QWERTY_RESULTS:-${BRAIN2QWERTY_CACHE}/results}"

# 检查数据集是否存在
if [[ ! -d "$BRAIN2QWERTY_STUDIES/MEG/FIF" ]]; then
    echo "[preprocess] 错误：未找到数据集 $BRAIN2QWERTY_STUDIES/MEG/FIF" >&2
    exit 1
fi

mkdir -p "$BRAIN2QWERTY_CACHE" "$BRAIN2QWERTY_RESULTS"

LOG_FILE="${BRAIN2QWERTY_RESULTS}/preprocess_$(date +%Y%m%d_%H%M%S).log"
echo "[preprocess] 数据集 : $BRAIN2QWERTY_STUDIES"
echo "[preprocess] 缓存目录: $BRAIN2QWERTY_CACHE"
echo "[preprocess] 输出目录: $BRAIN2QWERTY_RESULTS (训练/评估产物)"
echo "[preprocess] 日志    : $LOG_FILE"

MODE_ARGS=()
if [[ "${1:-}" == "--debug" ]]; then
    MODE_ARGS+=(--debug)
fi

echo "[preprocess] 开始预处理: python -m brain2qwerty_v2_brainomni.main cache ${MODE_ARGS[*]:-}"
python -m brain2qwerty_v2_brainomni.main cache "${MODE_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo "[preprocess] 完成。缓存已写入 $BRAIN2QWERTY_CACHE"
