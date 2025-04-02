import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import os
from utils import set_matplotlib_chinese_font


class MultiModalTrainer:
    """多模态网络训练器"""
    
    def __init__(self, model, device, learning_rate=0.001, weight_decay=1e-5, save_dir='checkpoints'):
        """
        初始化训练器
        
        参数:
            model: 要训练的模型
            device: 训练设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            save_dir: 模型保存目录
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        # 定义优化器
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 定义学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        
        # 记录训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader):
        """
        训练一个epoch
        
        参数:
            train_loader: 训练数据加载器
            
        返回:
            epoch_loss: 平均损失
            epoch_acc: 准确率
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in tqdm(train_loader, desc="训练"):
            # 提取批次数据
            bio_data = batch['bio'].to(self.device)
            clinical_data = batch['clinical'].to(self.device)
            mask = batch['mask'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if isinstance(self.model, torch.nn.DataParallel):
                model = self.model.module
            else:
                model = self.model
                
            if hasattr(model, 'imputation_net'):
                # 如果是带有模态补全的模型
                logits, _, _ = self.model(bio_data, clinical_data, mask)
            else:
                # 基础模型
                logits, _ = self.model(bio_data, clinical_data, mask)
            
            # 计算损失
            loss = self.criterion(logits, targets)
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 记录损失和预测
            total_loss += loss.item()
            
            # 获取预测
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        # 计算平均损失和准确率
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = accuracy_score(all_targets, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """
        在验证集上评估模型
        
        参数:
            val_loader: 验证数据加载器
            
        返回:
            val_loss: 验证集损失
            val_acc: 验证集准确率
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证"):
                # 提取批次数据
                bio_data = batch['bio'].to(self.device)
                clinical_data = batch['clinical'].to(self.device)
                mask = batch['mask'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # 前向传播
                if isinstance(self.model, torch.nn.DataParallel):
                    model = self.model.module
                else:
                    model = self.model
                    
                if hasattr(model, 'imputation_net'):
                    # 如果是带有模态补全的模型
                    logits, _, _ = self.model(bio_data, clinical_data, mask)
                else:
                    # 基础模型
                    logits, _ = self.model(bio_data, clinical_data, mask)
                
                # 计算损失
                loss = self.criterion(logits, targets)
                
                # 记录损失和预测
                total_loss += loss.item()
                
                # 获取预测
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 计算平均损失和准确率
        val_loss = total_loss / len(val_loader)
        val_acc = accuracy_score(all_targets, all_preds)
        
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader, num_epochs=100, early_stopping_patience=10):
        """
        训练模型
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            early_stopping_patience: 早停耐心
            
        返回:
            history: 训练历史
        """
        best_val_loss = float('inf')
        early_stopping_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 在验证集上评估
            val_loss, val_acc = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 打印当前epoch的结果
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pth')
                early_stopping_counter = 0
                print("保存了最佳模型！")
            else:
                early_stopping_counter += 1
                print(f"验证损失未改善. 早停计数器: {early_stopping_counter}/{early_stopping_patience}")
                
                if early_stopping_counter >= early_stopping_patience:
                    print("触发早停！")
                    break
                
            # 每10个epoch保存一次checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
        
        # 加载最佳模型
        self.load_model('best_model.pth')
        
        return self.history
    
    def test(self, test_loader, label_names=None):
        """
        在测试集上评估模型
        
        参数:
            test_loader: 测试数据加载器
            label_names: 标签名称，用于分类报告
            
        返回:
            test_acc: 测试集准确率
            f1: F1分数
            report: 分类报告
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="测试"):
                # 提取批次数据
                bio_data = batch['bio'].to(self.device)
                clinical_data = batch['clinical'].to(self.device)
                mask = batch['mask'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # 前向传播
                if isinstance(self.model, torch.nn.DataParallel):
                    model = self.model.module
                else:
                    model = self.model
                    
                if hasattr(model, 'imputation_net'):
                    # 如果是带有模态补全的模型
                    logits, _, _ = self.model(bio_data, clinical_data, mask)
                else:
                    # 基础模型
                    logits, _ = self.model(bio_data, clinical_data, mask)
                
                # 获取预测
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 计算各种指标
        test_acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        # 生成分类报告
        if label_names:
            report = classification_report(all_targets, all_preds, target_names=label_names)
        else:
            report = classification_report(all_targets, all_preds)
        
        # 生成混淆矩阵
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        print(f"\n测试准确率: {test_acc:.4f}")
        print(f"F1分数: {f1:.4f}")
        print("\n分类报告:")
        print(report)
        print("\n混淆矩阵:")
        print(conf_matrix)
        
        return test_acc, f1, report
    
    def test_with_specific_modality(self, test_loader, modality_type='all'):
        """
        使用特定模态在测试集上评估模型
        
        参数:
            test_loader: 测试数据加载器
            modality_type: 模态类型，可以是'all'、'bio'、'clinical'
            
        返回:
            acc: 准确率
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        
        # 设置模态掩码
        if modality_type == 'all':
            modality_mask = torch.tensor([1, 1], dtype=torch.float32)
        elif modality_type == 'bio':
            modality_mask = torch.tensor([1, 0], dtype=torch.float32)
        elif modality_type == 'clinical':
            modality_mask = torch.tensor([0, 1], dtype=torch.float32)
        else:
            raise ValueError("模态类型必须是'all'、'bio'或'clinical'之一")
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"测试({modality_type})"):
                # 提取批次数据
                bio_data = batch['bio'].to(self.device)
                clinical_data = batch['clinical'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # 为每个样本应用相同的掩码
                batch_size = bio_data.size(0)
                masks = modality_mask.unsqueeze(0).repeat(batch_size, 1).to(self.device)
                
                # 前向传播
                if isinstance(self.model, torch.nn.DataParallel):
                    model = self.model.module
                else:
                    model = self.model
                    
                if hasattr(model, 'imputation_net'):
                    # 如果是带有模态补全的模型
                    logits, _, _ = self.model(bio_data, clinical_data, masks)
                else:
                    # 基础模型
                    logits, _ = self.model(bio_data, clinical_data, masks)
                
                # 获取预测
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 计算准确率
        acc = accuracy_score(all_targets, all_preds)
        print(f"\n使用{modality_type}模态的准确率: {acc:.4f}")
        
        return acc
    
    def save_model(self, filename):
        """保存模型"""
        filepath = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filename):
        """加载模型"""
        filepath = os.path.join(self.save_dir, filename)
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
    
    def plot_training_history(self):
        """绘制训练历史"""
        # 确保matplotlib支持中文
        set_matplotlib_chinese_font()
        
        plt.figure(figsize=(12, 4))
        
        # 绘制损失
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='训练损失')
        plt.plot(self.history['val_loss'], label='验证损失')
        plt.title('训练和验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        
        # 绘制准确率
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='训练准确率')
        plt.plot(self.history['val_acc'], label='验证准确率')
        plt.title('训练和验证准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        plt.show() 