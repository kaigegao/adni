import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import platform
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


def set_matplotlib_chinese_font():
    """配置matplotlib支持中文显示"""
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import platform
    
    system = platform.system()
    if system == 'Windows':
        # Windows系统
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    elif system == 'Darwin':
        # macOS系统
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
    else:
        # Linux系统
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    print("已配置matplotlib支持中文显示")


def set_seed(seed):
    """
    设置随机种子以确保结果可重现
    
    参数:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"已设置随机种子为 {seed}")


def get_device():
    """
    获取可用的设备（CPU或GPU）
    
    返回:
        device: torch设备
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("使用CPU")
    return device


def save_config(config, save_dir):
    """
    保存实验配置
    
    参数:
        config: 配置字典
        save_dir: 保存目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"配置已保存到 {config_path}")


def create_tensorboard_writer(log_dir):
    """
    创建TensorBoard写入器
    
    参数:
        log_dir: 日志目录
        
    返回:
        writer: TensorBoard SummaryWriter
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return SummaryWriter(log_dir)


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    绘制混淆矩阵
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        save_path: 保存路径
    """
    # 确保matplotlib支持中文
    set_matplotlib_chinese_font()
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    # 使用 seaborn 绘制混淆矩阵热图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def plot_roc_curve(y_true, y_score, save_path=None):
    """
    绘制ROC曲线（仅适用于二分类）
    
    参数:
        y_true: 真实标签
        y_score: 预测概率
        save_path: 保存路径
    """
    # 确保matplotlib支持中文
    set_matplotlib_chinese_font()
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率')
    plt.ylabel('真正例率')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    return roc_auc


def plot_precision_recall_curve(y_true, y_score, save_path=None):
    """
    绘制精确率-召回率曲线（仅适用于二分类）
    
    参数:
        y_true: 真实标签
        y_score: 预测概率
        save_path: 保存路径
    """
    # 确保matplotlib支持中文
    set_matplotlib_chinese_font()
    
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率曲线')
    plt.legend(loc="lower left")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
    
    return pr_auc


def analyze_modality_performance(trainer, test_loader):
    """
    分析不同模态组合的性能
    
    参数:
        trainer: 多模态训练器
        test_loader: 测试数据加载器
        
    返回:
        results: 性能结果字典
    """
    # 确保matplotlib支持中文
    set_matplotlib_chinese_font()
    
    modality_types = ['all', 'bio', 'clinical']
    results = {}
    
    for modality_type in modality_types:
        acc = trainer.test_with_specific_modality(test_loader, modality_type)
        results[modality_type] = acc
    
    # 绘制性能对比图
    plt.figure(figsize=(8, 6))
    bars = plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange'])
    
    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.ylim(0, 1.0)
    plt.title('不同模态组合的性能对比')
    plt.ylabel('准确率')
    plt.xlabel('模态组合')
    
    plt.savefig('modality_performance_comparison.png')
    plt.show()
    
    return results


def export_results_to_csv(results, filename='results.csv'):
    """
    将实验结果导出为CSV文件
    
    参数:
        results: 结果字典
        filename: 导出文件名
    """
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"结果已导出到 {filename}")

