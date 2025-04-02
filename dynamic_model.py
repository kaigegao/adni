import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    """
    单一模态的编码器
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.3):
        super(ModalityEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建多层感知机
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 添加输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)


class AttentionFusion(nn.Module):
    """
    使用注意力机制融合多模态特征
    """
    def __init__(self, input_dim, num_modalities=2):
        super(AttentionFusion, self).__init__()
        
        # 为每个模态创建注意力权重计算模块
        self.attention_weights = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(num_modalities)
        ])
    
    def forward(self, modality_features, mask):
        """
        根据可用模态融合特征
        
        参数:
            modality_features: 模态特征列表 [batch_size, num_modalities, feature_dim]
            mask: 模态可用性掩码 [batch_size, num_modalities]
            
        返回:
            融合后的特征 [batch_size, feature_dim]
        """
        batch_size, num_modalities, feature_dim = modality_features.size()
        
        # 计算每个模态的注意力分数
        attention_scores = []
        for i in range(num_modalities):
            # 对每个模态的特征计算注意力分数
            score = self.attention_weights[i](modality_features[:, i, :])  # [batch_size, 1]
            attention_scores.append(score)
        
        # 将分数拼接并转置为 [batch_size, num_modalities]
        attention_scores = torch.cat(attention_scores, dim=1)
        
        # 应用掩码，将不可用模态的注意力分数设为极小值
        masked_attention = attention_scores.clone()
        masked_attention[mask == 0] = -1e9
        
        # 应用Softmax获取注意力权重
        attention_weights = F.softmax(masked_attention, dim=1)  # [batch_size, num_modalities]
        
        # 使用注意力权重融合特征
        # 扩展attention_weights为 [batch_size, num_modalities, 1]
        attention_weights = attention_weights.unsqueeze(2)
        
        # 加权求和融合特征 [batch_size, feature_dim]
        weighted_features = torch.sum(modality_features * attention_weights, dim=1)
        
        return weighted_features


class DynamicMultiModalNet(nn.Module):
    """
    动态多模态神经网络，能处理模态缺失情况
    """
    def __init__(self, bio_input_dim, clinical_input_dim, hidden_dims, fusion_dim, num_classes, dropout_rate=0.3):
        super(DynamicMultiModalNet, self).__init__()
        
        # 创建各模态的编码器
        self.bio_encoder = ModalityEncoder(bio_input_dim, hidden_dims, fusion_dim, dropout_rate)
        self.clinical_encoder = ModalityEncoder(clinical_input_dim, hidden_dims, fusion_dim, dropout_rate)
        
        # 融合模块
        self.fusion = AttentionFusion(fusion_dim, num_modalities=2)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1], num_classes)
        )
        
    def forward(self, bio_data, clinical_data, modality_mask):
        """
        前向传播
        
        参数:
            bio_data: 生物样本数据 [batch_size, bio_input_dim]
            clinical_data: 临床数据 [batch_size, clinical_input_dim]
            modality_mask: 模态可用性掩码 [batch_size, 2]
            
        返回:
            logits: 分类logits
            modality_features: 各模态编码的特征
        """
        batch_size = bio_data.size(0)
        
        # 编码各模态数据
        bio_features = self.bio_encoder(bio_data)  # [batch_size, fusion_dim]
        clinical_features = self.clinical_encoder(clinical_data)  # [batch_size, fusion_dim]
        
        # 将特征整合为 [batch_size, num_modalities, fusion_dim]
        modality_features = torch.stack([bio_features, clinical_features], dim=1)
        
        # 使用注意力机制融合特征
        fused_features = self.fusion(modality_features, modality_mask)  # [batch_size, fusion_dim]
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits, modality_features


class ModalityImputationNet(nn.Module):
    """
    模态缺失情况下的补全网络
    """
    def __init__(self, fusion_dim):
        super(ModalityImputationNet, self).__init__()
        
        # 模态间互相补全的转换模块
        self.bio_to_clinical = nn.Linear(fusion_dim, fusion_dim)
        self.clinical_to_bio = nn.Linear(fusion_dim, fusion_dim)
    
    def forward(self, modality_features, modality_mask):
        """
        补全缺失模态
        
        参数:
            modality_features: 各模态编码的特征 [batch_size, num_modalities, fusion_dim]
            modality_mask: 模态可用性掩码 [batch_size, 2]
            
        返回:
            imputed_features: 补全后的特征 [batch_size, num_modalities, fusion_dim]
        """
        batch_size, num_modalities, fusion_dim = modality_features.size()
        
        # 复制特征，用于保存补全结果
        imputed_features = modality_features.clone()
        
        # 遍历每个样本
        for i in range(batch_size):
            mask = modality_mask[i]
            
            # 如果生物样本模态缺失，但临床模态可用
            if mask[0] == 0 and mask[1] == 1:
                # 使用临床模态特征补全生物样本模态
                imputed_features[i, 0] = self.clinical_to_bio(modality_features[i, 1])
            
            # 如果临床模态缺失，但生物样本模态可用
            elif mask[0] == 1 and mask[1] == 0:
                # 使用生物样本模态特征补全临床模态
                imputed_features[i, 1] = self.bio_to_clinical(modality_features[i, 0])
        
        return imputed_features


class DynamicMultiModalNetWithImputation(nn.Module):
    """
    结合模态补全的动态多模态神经网络
    """
    def __init__(self, bio_input_dim, clinical_input_dim, hidden_dims, fusion_dim, num_classes, dropout_rate=0.3):
        super(DynamicMultiModalNetWithImputation, self).__init__()
        
        # 基础多模态网络
        self.base_model = DynamicMultiModalNet(
            bio_input_dim, clinical_input_dim, hidden_dims, fusion_dim, num_classes, dropout_rate
        )
        
        # 模态补全网络
        self.imputation_net = ModalityImputationNet(fusion_dim)
        
        # 用于融合原始和补全特征的参数
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, bio_data, clinical_data, modality_mask):
        """
        前向传播
        
        参数:
            bio_data: 生物样本数据 [batch_size, bio_input_dim]
            clinical_data: 临床数据 [batch_size, clinical_input_dim]
            modality_mask: 模态可用性掩码 [batch_size, 2]
            
        返回:
            logits: 分类logits
            原始模态特征
            补全后的模态特征
        """
        # 获取原始模态特征和初步预测
        logits_orig, modality_features = self.base_model(bio_data, clinical_data, modality_mask)
        
        # 补全缺失模态
        imputed_features = self.imputation_net(modality_features, modality_mask)
        
        # 融合原始特征和补全特征
        alpha = torch.sigmoid(self.alpha)  # 确保alpha在0-1之间
        fused_features = alpha * modality_features + (1 - alpha) * imputed_features
        
        # 使用融合特征再次进行预测
        batch_size, num_modalities, fusion_dim = fused_features.size()
        
        # 将特征重组后送入分类器
        fused_features_flat = fused_features.view(batch_size * num_modalities, fusion_dim)
        
        # 使用全模态掩码(表示所有模态都可用)进行融合和分类
        all_modalities_mask = torch.ones(batch_size, num_modalities, device=modality_mask.device)
        fused_feature_vectors = self.base_model.fusion(fused_features, all_modalities_mask)
        
        # 最终分类
        logits_final = self.base_model.classifier(fused_feature_vectors)
        
        return logits_final, modality_features, imputed_features 