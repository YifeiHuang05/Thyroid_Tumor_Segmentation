# ThyroidXL 医学影像算法 Summer Camp 参赛须知

## 一、活动目标

本 Summer Camp 围绕 ThyroidXL 甲状腺结节超声官方数据集开展算法研究。各组以病灶分割和疾病相关预测为核心，自主完成数据分析、模型设计、训练、测试和结果展示。

鼓励探索的技术路线包括但不限于：

- 深度学习；
- 机器学习；
- 影像组学；
- 分割与分类联合建模；
- 多任务学习；
- 迁移学习；
- 自监督学习；
- 数据增强与合成数据；
- 多模型集成；
- 多图像或多帧信息融合。

本活动仅用于内部教学、科研训练和算法交流，不构成临床诊断或临床决策依据。

## 二、组队要求

- 学生自由组队；
- 每组 4-5 位同学；
- 每位同学仅可加入一个队伍；
- 每组指定一名组长负责沟通与材料汇总；
- 各组可自行安排数据处理、模型开发、实验评估和展示等分工。

## 三、官方数据集与数据划分

各组统一以 ThyroidXL 官方数据集及其官方训练/测试划分为基础开展实验。

- 训练阶段使用官方训练集；
- 最终结果使用官方测试集评估；
- 如需从训练集中自行划分验证集，建议按患者或病灶进行划分；
- 不建议将同一患者或同一病灶的不同图像同时分配至训练集和验证集；
- 官方测试集不应用于训练、人工标注、伪标签生成或反复调参。

## 四、任务设置

### 任务 A：甲状腺结节分割

输入为甲状腺超声图像，输出为结节区域的二值分割 mask。

在分割结果基础上，可进一步计算：病灶面积、周长、长轴、短轴、外接框、纵横比、位置和边界相关特征。

### 任务 B：FNAC 分组预测

预测甲状腺结节 FNAC 分组，标签包括：

- Bethesda II；
- Bethesda III-IV；
- Bethesda V-VI。

### 任务 C：PTC 预测

预测结节是否为甲状腺乳头状癌（PTC），标签包括：

- `ptc`；
- `non_ptc_pathology`。

### 任务 D：良恶性预测

预测结节良恶性，标签包括：

- `benign`；
- `malignant`。

### 任务 E：TI-RADS 分级预测

预测甲状腺结节 TI-RADS 分级，标签为 1、2、3、4、5。

## 五、外部数据与公开资源

允许各组自行使用合法公开的数据集、开源代码、公开预训练权重和基础模型。

外部数据不要求与 ThyroidXL 标签完全一致。对于外部数据存在的标签类别不同、标签粒度不同、病理定义不同、数据格式不同或缺失部分标签等情况，各组可自行决定：

- 是否纳入训练；
- 是否仅用于预训练或自监督学习；
- 是否仅用于分割任务；
- 是否仅用于良恶性、TI-RADS 或其他单项预测任务；
- 是否进行标签映射；
- 是否剔除部分不匹配样本。

使用外部数据时，应在展示材料中说明：

- 数据集名称和来源；
- 使用的任务和样本数量；
- 标签映射方式；
- 未使用或剔除样本的原因；
- 外部数据是否直接参与监督训练。

## 六、统一评估指标

各组应在 ThyroidXL 官方测试集上报告以下核心指标。

| 任务 | 必须报告的核心指标 |
|---|---|
| 病灶分割 | Dice、IoU、HD95 |
| FNAC 分组预测 | Macro F1、Balanced Accuracy、Accuracy |
| PTC 预测 | Macro F1、Balanced Accuracy、Accuracy |
| 良恶性预测 | Macro F1、Balanced Accuracy、Accuracy |
| TI-RADS 分级预测 | Macro F1、Balanced Accuracy、Accuracy |

允许额外报告 AUROC、Average Precision、Sensitivity、Specificity、混淆矩阵、各类别 F1、推理时间、可解释性结果等，但不作为统一必需指标。

## 七、指标定义与计算公式

设：

- $P$：预测的病灶区域；
- $G$：真实病灶区域；
- $TP$：真正例；
- $TN$：真负例；
- $FP$：假正例；
- $FN$：假负例；
- $K$：分类类别数。

### 1. Dice

$$
\operatorname{Dice}(P,G)=\frac{2\lvert P\cap G\rvert}{\lvert P\rvert+\lvert G\rvert}
$$

Dice 衡量预测病灶区域与真实病灶区域的重叠程度，取值范围为 0 到 1，越高越好。

### 2. IoU

$$
\operatorname{IoU}(P,G)=\frac{\lvert P\cap G\rvert}{\lvert P\cup G\rvert}
$$

IoU 即交并比，取值范围为 0 到 1，越高越好。

### 3. HD95

HD95 用于衡量预测边界与真实边界之间的距离：

$$
HD_{95}(P,G)=\max\left\{Q_{0.95}\bigl(D(\partial P,\partial G)\bigr),\;Q_{0.95}\bigl(D(\partial G,\partial P)\bigr)\right\}
$$

其中，$\partial P$ 与 $\partial G$ 分别为预测边界和真实边界；$D(A,B)$ 表示边界集合 $A$ 中每个点到 $B$ 的最短距离集合；$Q_{0.95}$ 表示第 95 百分位数。HD95 越低，说明预测边界越接近真实边界。

### 4. Accuracy

$$
\operatorname{Accuracy}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}\left(\hat{y}_i=y_i\right)
$$

对于二分类任务，也可写为：

$$
\operatorname{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$

Accuracy 反映总体正确率，但在类别不平衡时不能单独作为主要判断依据。

### 5. Balanced Accuracy

对于第 $k$ 个类别：

$$
\operatorname{Recall}_k=\frac{TP_k}{TP_k+FN_k}
$$

多分类任务的 Balanced Accuracy 定义为各类别 Recall 的平均值：

$$
\operatorname{Balanced\ Accuracy}=\frac{1}{K}\sum_{k=1}^{K}\operatorname{Recall}_k
$$

对于二分类任务：

$$
\operatorname{Balanced\ Accuracy}=\frac{\operatorname{Sensitivity}+\operatorname{Specificity}}{2}
$$

Balanced Accuracy 能降低多数类别对结果的主导作用，适用于类别不均衡的医学分类任务。

### 6. Macro F1

对于第 $k$ 个类别：

$$
\operatorname{Precision}_k=\frac{TP_k}{TP_k+FP_k}
$$

$$
F1_k=\frac{2\operatorname{Precision}_k\operatorname{Recall}_k}{\operatorname{Precision}_k+\operatorname{Recall}_k}=\frac{2TP_k}{2TP_k+FP_k+FN_k}
$$

Macro F1 为所有类别 F1 的直接平均：

$$
\operatorname{Macro\ F1}=\frac{1}{K}\sum_{k=1}^{K}F1_k
$$

Macro F1 对每个类别赋予相同权重，因此适合评估 FNAC 和 TI-RADS 等类别分布不均衡的任务。
