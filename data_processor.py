import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

class ADNIDataProcessor:
    """ADNI数据集处理器：用于加载和预处理生物样本和临床数据"""
    
    def __init__(self, biospecimen_dir='biospecimen', clinical_dir='clinical'):
        """
        初始化ADNI数据处理器
        
        参数:
            biospecimen_dir: 生物样本数据目录
            clinical_dir: 临床数据目录
        """
        self.biospecimen_dir = biospecimen_dir
        self.clinical_dir = clinical_dir
        self.bio_data = None
        self.clinical_data = None
        self.merged_data = None
        self.feature_scalers = {}
        self.label_encoder = None
        
    def load_data(self):
        """加载生物样本和临床数据"""
        # 加载生物样本数据
        bio_files = [f for f in os.listdir(self.biospecimen_dir) if f.endswith('.csv')]
        bio_dfs = []
        
        for file in bio_files:
            file_path = os.path.join(self.biospecimen_dir, file)
            df = pd.read_csv(file_path)
            bio_dfs.append(df)
            
        if bio_dfs:
            self.bio_data = pd.concat(bio_dfs, axis=0)
            
        # 加载临床数据
        clinical_files = [f for f in os.listdir(self.clinical_dir) if f.endswith('.csv')]
        clinical_dfs = []
        
        for file in clinical_files:
            file_path = os.path.join(self.clinical_dir, file)
            df = pd.read_csv(file_path)
            clinical_dfs.append(df)
            
        if clinical_dfs:
            self.clinical_data = pd.concat(clinical_dfs, axis=0)
            
        print(f"加载了 {len(bio_dfs)} 个生物样本文件和 {len(clinical_dfs)} 个临床数据文件")
        
        # 合并数据集
        self._merge_datasets()
        
        return self.bio_data, self.clinical_data
    
    def _merge_datasets(self):
        """合并生物样本和临床数据集"""
        if self.bio_data is not None and self.clinical_data is not None:
            # 使用RID（受试者ID）和VISCODE（访问代码）进行合并
            self.merged_data = pd.merge(
                self.bio_data, 
                self.clinical_data,
                on=['RID', 'VISCODE'], 
                how='inner'
            )
            print(f"合并后的数据集包含 {self.merged_data.shape[0]} 行和 {self.merged_data.shape[1]} 列")
        else:
            print("无法合并数据集：生物样本或临床数据缺失")
    
    def preprocess_data(self, target_col='GENOTYPE'):
        """
        预处理数据：处理缺失值、编码分类变量、标准化数值特征
        
        参数:
            target_col: 目标列名
        """
        if self.merged_data is None:
            print("请先加载和合并数据")
            return None
        
        # 复制数据以避免修改原始数据
        data = self.merged_data.copy()
        
        # 处理缺失值
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
        
        # 查找分类列
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != target_col:  # 不处理目标列
        # 用最常见的值填充缺失值
                mode_values = data[col].mode()
                if not mode_values.empty:
                    data[col] = data[col].fillna(mode_values[0])
                else:
            # 如果没有众数，可以用其他方法填充，例如用一个特殊值
                    data[col] = data[col].fillna("UNKNOWN")
        # 进行One-Hot编码
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                data = pd.concat([data, dummies], axis=1)
                
        # 删除原始分类列
        data = data.drop(columns=[col for col in categorical_cols if col != target_col])
        
        # 标准化数值特征
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if not data[col].isna().all():  # 确保列不是全部NaN
                scaler = StandardScaler()
                data[col] = scaler.fit_transform(data[[col]])
                self.feature_scalers[col] = scaler
            else:
        # 如果列全是NaN，填充为0
                data[col] = 0
        
        # 编码目标变量
        if target_col in data.columns:
            self.label_encoder = LabelEncoder()
            data[target_col] = self.label_encoder.fit_transform(data[target_col])
        
        return data
    
    def split_data(self, data, target_col='GENOTYPE', test_size=0.2, val_size=0.1, random_state=42):
        """
        将数据集分割为训练集、验证集和测试集
        
        参数:
            data: 预处理后的数据
            target_col: 目标列名
            test_size: 测试集比例
            val_size: 验证集比例
            random_state: 随机种子
        
        返回:
            训练特征、验证特征、测试特征、训练标签、验证标签、测试标签、生物样本特征列名、临床特征列名
        """
        if data is None:
            print("请先预处理数据")
            return None
        
        # 区分生物样本特征和临床特征
        bio_features = [col for col in data.columns if col in self.bio_data.columns and col != target_col]
        clinical_features = [col for col in data.columns if col in self.clinical_data.columns and col != target_col]
        
        # 分离特征和目标变量
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # 首先分离测试集
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 从剩余数据中分离验证集
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=random_state, stratify=y_train_val
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test, bio_features, clinical_features


class ADNIMultiModalDataset(Dataset):
    """ADNI多模态数据集类"""
    
    def __init__(self, X, y, bio_features, clinical_features, modality_mask=None):
        """
        初始化多模态数据集
        
        参数:
            X: 特征数据
            y: 标签数据
            bio_features: 生物样本特征列名
            clinical_features: 临床特征列名
            modality_mask: 模态缺失掩码 (None表示不使用掩码，默认所有模态可用)
        """
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.bio_features = bio_features
        self.clinical_features = clinical_features
        self.modality_mask = modality_mask
        
        # 创建特征列的索引映射
        self.feature_indices = {
            'bio': [i for i, col in enumerate(X.columns) if col in bio_features],
            'clinical': [i for i, col in enumerate(X.columns) if col in clinical_features]
        }
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        如果指定了modality_mask，按照掩码进行模态缺失
        否则所有模态都可用
        """
        x = self.X.iloc[idx].values.astype(np.float32)
        y = self.y.iloc[idx]
        
        # 分离不同模态的数据
        bio_data = x[self.feature_indices['bio']]
        clinical_data = x[self.feature_indices['clinical']]
        
        # 创建可用模态掩码
        if self.modality_mask is not None:
            # 使用提供的掩码
            available_modalities = self.modality_mask[idx % len(self.modality_mask)]
            
            # 根据掩码将不可用模态数据清零
            if available_modalities[0] == 0:  # 生物样本模态不可用
                bio_data = np.zeros_like(bio_data)
            
            if available_modalities[1] == 0:  # 临床模态不可用
                clinical_data = np.zeros_like(clinical_data)
                
            modality_mask = torch.tensor(available_modalities, dtype=torch.float32)
        else:
            # 默认所有模态可用
            modality_mask = torch.ones(2, dtype=torch.float32)
        
        return {
            'bio': torch.tensor(bio_data, dtype=torch.float32),
            'clinical': torch.tensor(clinical_data, dtype=torch.float32),
            'mask': modality_mask,
            'target': torch.tensor(y, dtype=torch.long)
        }


def create_dataloaders(processor, target_col='GENOTYPE', batch_size=32, modality_drop_rate=0.3):
    """
    创建用于训练的DataLoader
    
    参数:
        processor: 数据处理器实例
        target_col: 目标列名
        batch_size: 批量大小
        modality_drop_rate: 训练时模态丢弃率
    
    返回:
        训练、验证和测试数据加载器
    """
    # 加载数据
    processor.load_data()
    
    # 预处理数据
    processed_data = processor.preprocess_data(target_col=target_col)
    
    # 分割数据
    X_train, X_val, X_test, y_train, y_val, y_test, bio_features, clinical_features = processor.split_data(
        processed_data, target_col=target_col
    )
    
    # 创建训练时的模态缺失掩码
    # 每个样本可能缺失生物样本数据、临床数据，或者都不缺失
    # [[1, 1], [1, 0], [0, 1], [0, 0]] 分别表示：所有模态可用、只有生物样本可用、只有临床数据可用、所有模态缺失
    possible_masks = [[1, 1], [1, 0], [0, 1]]  # 排除所有模态缺失的情况
    
    # 随机生成训练中可能的模态组合
    train_masks = []
    for _ in range(len(X_train)):
        if np.random.random() < modality_drop_rate:
            # 随机选择一种缺失模式(排除第一种"所有模态可用"的情况)
            mask_idx = np.random.randint(1, len(possible_masks))
            train_masks.append(possible_masks[mask_idx])
        else:
            # 所有模态都可用
            train_masks.append(possible_masks[0])
    
    # 创建数据集
    train_dataset = ADNIMultiModalDataset(X_train, y_train, bio_features, clinical_features, train_masks)
    val_dataset = ADNIMultiModalDataset(X_val, y_val, bio_features, clinical_features)
    test_dataset = ADNIMultiModalDataset(X_test, y_test, bio_features, clinical_features)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, bio_features, clinical_features 