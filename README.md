## 引用 (Citation)

如果您在研究中使用了本代码或相关思想，请引用我们的论文。我们的论文目前正在 **IEEE IoTJ** 经历第二轮评审，其预印本已发布在 arXiv 上：

If you use this code or related ideas in your research, please cite our paper. Our paper is currently under its second review at the **IEEE Internet of Things Journal** and is available as a preprint on arXiv:

[1] Shuai L, Tan S, Zhang N, et al. PoisonCatcher: Revealing and Identifying LDP Poisoning Attacks in IIoT[J]. arXiv preprint arXiv:2412.15704, 2024.

或者使用 BibTeX 格式引用：

```bibtex
@article{shuai2024poisoncatcher,
  title={PoisonCatcher: Revealing and Identifying LDP Poisoning Attacks in IIoT},
  author={Shuai, Lisha and Tan, Shaofeng and Zhang, Nan and Zhang, Jiamin and Zhang, Min and Yang, Xiaolong},
  journal={arXiv preprint arXiv:2412.15704},
  year={2024}
}
```

一旦论文正式发表，我们将在此更新最终的引用信息（包括 DOI、卷、期、页码等）。

Once the paper is officially published, we will update the final citation information (including DOI, volume, issue, pages, etc.) here.

## 实验设计与验证概述

本研究的核心目标是通过严谨的实验设计，全面评估并验证我们提出的 **PoisonCatcher** 方法在识别针对本地差分隐私 (LDP) 环境下的数据毒化攻击方面的有效性。

实验验证选用了具有代表性的真实世界数据集：**全球天气存储库** (Kaggle: [https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository))。该数据集包含超过40项气象特征，其复杂性和多样性有助于充分模拟现实应用场景，从而验证 PoisonCatcher 在实际条件下的鲁棒性和普适性。数据集预处理遵循标准流程：对连续属性进行标准化，对离散属性进行独热编码，确保数据质量并消除量纲影响。

为了全面探究 PoisonCatcher 的防御能力，我们模拟了论文中详述的 **三种典型的 LDP 数据毒化攻击**。这些攻击策略涵盖了不同的毒化模式，通过评估 PoisonCatcher 在这些攻击下的表现，我们不仅能验证其有效性，还能更深入地理解 LDP 协议在面临蓄意攻击时的潜在脆弱点。

我们将 **F2 分数** 作为主要的性能评估指标。F2 分数能够综合考量模型的精确率和召回率，尤其侧重于高召回率（即能否尽可能多地找出毒化数据项），这对于数据清洗和防御至关重要。

实验对比了 PoisonCatcher 与多种现有或常用的基线方法，包括：**基于 FP-Growth 的虚假用户检测 (DETECT)[20]**、**基于双向采集方法LDPGuard方法**[20]，以及 **由8位权威机构信息安全专家组成的专家评估小组（包括3个基于人工智能的评估模型）进行的人工评估**。通过与这些基线方法的定量对比，旨在有力地论证 PoisonCatcher 在毒化程度估计方面的显著优势。

所有实验均严格控制随机种子，以保证结果的完全可复现性。本项目的全部实验代码、数据处理脚本及合成数据生成工具均已开源，旨在提升研究的透明度并便于社区进行独立的验证和进一步研究。

## PoisonCatcher 项目结构

- Attack_Simulation_Four.py // 攻击过程模拟
- Attacked_Dataset_Generate.py // 攻击数据生成
- Attribute_Correlation_Detector.py // 属性相关性检测器
- Part1_Temporal_Similarity_Detector.py // 时序相似性检测器
- Experiment_Result.py // 完整实验流程
- Real_Data_Process.py  // 对真实数据的预处理
- Statistical.py  // 对LDP数据进行统计查询
- Experiment_[num] // 论文中相应的实验

---

- File
  - GlobalWeatherRepository.csv  // 原始真实数据
  - Preprocessing_Data.csv  // 预处理后数据
  - Divide_data_by_time //（文件夹）按天提取的原始数据集
  - Attacked_Dataset  // （文件夹）生成的攻击数据
  - Human_Expert_Review_Scale // （文件夹）由8位权威机构信息安全专家组成的专家评估小组（包括3个基于人工智能的评估模型）数据
  - pdf // （文件夹）论文中相应的实验结果（图片）
  - Experiment_[num] // 论文中相应的实验数据
