# 将 SpanishBCBL 接入 Brain2Qwerty V2

本分支默认使用 SpanishBCBL（study 名称 `Pinet2024Meg`）运行 V2 的完整
CTC 数据处理和训练流程。实现原则是只增加一个事件适配层；V2 的数值预处理、
句子级切分、数据增强、模型、损失和优化配置均保持不变。

## 配置与运行

`BRAIN2QWERTY_STUDIES` 指向解压后的 SpanishBCBL 数据根目录；缓存和输出建议
放在独立目录中：

```bash
export BRAIN2QWERTY_STUDIES=/path/to/SpanishBCBL
export BRAIN2QWERTY_CACHE=/path/to/cache
export BRAIN2QWERTY_RESULTS=/path/to/results

python -m brain2qwerty_v2.main cache --debug
python -m brain2qwerty_v2.main debug
python -m brain2qwerty_v2.main train
python -m brain2qwerty_v2.main eval --ckpt /path/to/checkpoint.ckpt

# 查看训练/评估曲线（日志位于结果目录下的 logs/version_N/）
tensorboard --logdir "$BRAIN2QWERTY_RESULTS/logs"
```

默认配置位于 `config/xp_config.py`，使用 `Pinet2024Meg`、
`SpanishBCBLV2Preprocessing` 和 `Brain2QwertyV2Splitter`。

## SpanishBCBL 专用处理

`SpanishBCBLV2Preprocessing` 只负责把 V1 数据事件转换为 V2 所要求的句子级
输入格式，具体处理如下：

1. 将原始 `Button` / `DetectedButton` 统一改名为 `Keystroke`，删除每个 block
   的练习 trial 0 和 1。
2. 复用 V1 的受试者规则：删除无键盘对照组和被排除的 S23，并合并属于同一人的
   重复 recording ID。受试者仍保留字符串 ID，由 V2 的 `LabelEncoder` 编码。
3. 删除已知异常句子
   `65.0_Pinet2024Meg_subject-S1_session-1_task-block1`。
4. 只保留正式输入阶段（`is_percep == False`）的 Sentence 和 Keystroke，以及
   MEG 事件；感知阶段和 Word 事件不进入当前 CTC-only 流程。
5. 每个 `sentence_UID` 必须有且只有一个非空的源 Sentence。参考文本转为小写并
   去除首尾空白，以稳定切分键和评估文本。
6. CTC 标签表示参与者实际按下的有效按键，按时间排序。`<space>` 转为 `&`；
   只保留 V2 原有的 `a-z` 加空格字表。`<special>`、`<number>`、重音/符号等
   不在 V2 字表内的按键被删除。删除后没有有效按键的句子也被删除。
7. 保留原始 Sentence 窗口；若有效按键位于窗口外，只向前或向后扩展边界，绝不
   缩短窗口。这样 V2 的连续 MEG 片段覆盖全部 CTC 目标按键，同时尽量保持源时序。
8. 为每个按键生成稳定的 `button_UID`。`SentenceKeySeq` 的缓存项 ID 同时包含
   参考文本和 `typed_label`，避免相同参考句被不同参与者输错成不同序列时复用错误
   标签缓存；该缓存版本已从 v5 更新到 v6。

## 保持不变的 V2 流程

- 切分：按规范化后的唯一 Sentence 参考文本做确定性哈希，比例为
  train/val/test = 80%/10%/10%。相同文本的所有重复记录进入同一 split，但同一
  受试者仍可能出现在多个 split。该规则不同于 V1 的 TF-IDF 近义句聚类切分。
- MEG 数值预处理：仅选 MEG 通道，重采样至 100 Hz，0.5–45 Hz 带通，50 Hz
  陷波，`RobustScaler`，数值截断到 ±5，不做 baseline correction。
- 窗口：Sentence 开始前取 0.4 秒，并在句末随机增加 0.4–0.5 秒；训练集保留
  V2 的随机 onset jitter。
- 训练增强：每通道常量偏移标准差 0.3、时间遮挡、频率/通道遮挡和 0.8–1.2
  time stretch；白噪声标准差为 0。
- 模型与训练：Conv + Conformer、辅助/最终双 CTC 头，`loss_alpha=0.7`，
  AdamW（学习率 `8e-4`、weight decay `1e-3`），500-step warmup + cosine
  scheduler，275 epochs，bf16 mixed precision，梯度累积 2。

## 标签与评估含义

训练目标 `typed_label` 是参与者的实际有效按键序列，不是标准句子。测试输出中的
`typed_text` 由 `typed_label` 还原得到，`CTC_CER` 以该 typed label 为目标；源 Sentence
的参考文本不写入 `predictions_test.json`。

## 验证

快速单元测试覆盖练习 trial、受试者合并/排除、感知事件、无效按键、边界扩展、
实际标签缓存和确定性切分。真实数据可运行：

```bash
PYTHONPATH=. python tests/live_spanish_v2_check.py /path/to/SpanishBCBL
```

该检查加载一条真实时间线，并验证适配后每个有效 Keystroke 都位于对应 Sentence
窗口内且每个 Sentence 都有非空 CTC 标签。
