# 任务
参考代码：
- brain2qwerty_v2
- standalone_brainomni
在 brain2qwerty_v2_brainomni 中完全模仿 brain2qwerty_v2 的文件结构，但是数据预处理需要与brainomni模型的预处理对齐，并且需要将原模型替换为brainomni

# 要求
- 数据预处理部分
    - 直接修改config里面的滤波陷波重采样参数，与standalone_brainomni/factory里面的数值对齐即可，做最小修改
    - 去掉robust scaler和clamp操作。替换为sample-level的逐通道zscore
    - 加入standalone_brainomni/factory/preprocess_utils.py中pos，sensortype的提取逻辑
    - 要求预处理结果的样本条数，每条样本的句子内容，时间，最终划分情况与brain2qwerty_v2一致，与.cache/spanishbcbl_meg_v2一致，确保可比
    - 预处理的数据保存在 .cache/spanishbcbl_meg_v2_brainomni 路径
- 模型接入部分
  - brainomni最开始的conv1d固定使用overlap stride，在256hz输入下输出16hz特征
  - brainomni的ctc head支持rms+linear和rms+temporal-conv两种结构，通过classifier_head切换
  - brainomni同样需要aux以及final ctc，aux位置加入在spatial qformer计算完成之后，time attention之前；通过aux_prediction总开关控制aux、feedback和aux loss
  - 思考给brainomni加入subject embedding的做法
- 训练pipeline
  - 需要参考classification/tester.py，通过最小的改动，允许先冻住brainomni backbone，先微调对齐ctc head，之后再放开一起调整

# 约束
- 仅允许修改 brain2qwerty_v2_brainomni 以及 script/brain2qwerty_v2_brainomni
- 代码不从standalone_brainomni进行import
- 保持代码简洁，干净，只做最小改动
- 不做过度包装，不做容错判断处理
