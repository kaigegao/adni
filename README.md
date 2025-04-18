# 多模态动态神经网络用于ADNI数据集分析


## 数据集说明

本项目使用ADNI数据集中的两种数据类型：

1. **生物样本数据**：包含基因型信息、蛋白质标记物等生物学指标
   - APOERES：载脂蛋白E基因型数据
   - UPENNBIOMK_ROCHE_ELECSYS：生物标志物数据

2. **临床数据**：包含患者的临床信息、医学史等
   - MEDHIST：病史数据
   - NEUROEXM：神经学检查数据
   - RECCMEDS：用药记录
   - PTDEMOG：人口统计学数据
   - VITALS：生命体征数据


### 使用模态补全

```bash
python main.py --use_imputation
```

### 自定义参数

```bash
python main.py --biospecimen_dir biospecimen --clinical_dir clinical --target_col GENOTYPE --batch_size 64 --hidden_dims 512 256 128 --fusion_dim 64 --lr 0.0005 --epochs 200
```

## 主要参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| biospecimen_dir | 生物样本数据目录 | biospecimen |
| clinical_dir | 临床数据目录 | clinical |
| target_col | 目标列名 | GENOTYPE |
| batch_size | 批量大小 | 32 |
| modality_drop_rate | 训练时模态丢弃率 | 0.3 |
| hidden_dims | 隐藏层维度 | [256, 128, 64] |
| fusion_dim | 融合特征维度 | 32 |
| dropout | Dropout率 | 0.3 |
| use_imputation | 是否使用模态补全 | False |
| lr | 学习率 | 0.001 |
| weight_decay | 权重衰减 | 1e-5 |
| epochs | 训练轮数 | 100 |
| early_stopping | 早停耐心 | 10 |
| save_dir | 模型保存目录 | checkpoints |
| seed | 随机种子 | 42 |
| gpu | GPU ID，-1表示使用CPU | 0 |

## 模型架构

### 基础动态多模态网络

该模型由以下部分组成：

1. **模态编码器**：为每种模态（生物样本和临床数据）提供单独的特征提取网络
2. **注意力融合**：根据模态可用性和重要性自适应融合特征
3. **分类器**：对融合特征进行分类预测

### 带模态补全的动态网络

在基础网络上增加了模态补全功能：

1. **模态间映射**：学习不同模态间的相互映射关系
2. **补全模块**：使用可用模态补全缺失模态的特征
3. **特征融合**：将原始特征和补全特征进行加权融合

