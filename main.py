import os
import argparse
import torch
from data_processor import ADNIDataProcessor, create_dataloaders
from dynamic_model import DynamicMultiModalNet, DynamicMultiModalNetWithImputation
from trainer import MultiModalTrainer
from utils import set_seed, get_device, save_config, analyze_modality_performance, set_matplotlib_chinese_font


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='动态多模态神经网络训练')
    
    # 数据相关参数
    parser.add_argument('--biospecimen_dir', type=str, default='biospecimen', help='生物样本数据目录')
    parser.add_argument('--clinical_dir', type=str, default='clinical', help='临床数据目录')
    parser.add_argument('--target_col', type=str, default='GENOTYPE', help='目标列名')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--modality_drop_rate', type=float, default=0.3, help='训练时模态丢弃率')
    
    # 模型相关参数
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128, 64], help='隐藏层维度')
    parser.add_argument('--fusion_dim', type=int, default=32, help='融合特征维度')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout率')
    parser.add_argument('--use_imputation', action='store_true', help='是否使用模态补全')
    
    # 训练相关参数
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--early_stopping', type=int, default=10, help='早停耐心')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID，-1表示使用CPU')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置matplotlib支持中文
    set_matplotlib_chinese_font()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    print(f"使用设备: {device}")
    
    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 保存配置
    config = vars(args)
    save_config(config, args.save_dir)
    
    # 加载和处理数据
    processor = ADNIDataProcessor(
        biospecimen_dir=args.biospecimen_dir,
        clinical_dir=args.clinical_dir
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, bio_features, clinical_features = create_dataloaders(
        processor,
        target_col=args.target_col,
        batch_size=args.batch_size,
        modality_drop_rate=args.modality_drop_rate
    )
    
    print(f"生物样本特征数量: {len(bio_features)}")
    print(f"临床特征数量: {len(clinical_features)}")
    
    # 获取类别数量
    num_classes = len(processor.label_encoder.classes_)
    print(f"类别数量: {num_classes}")
    
    # 创建模型
    if args.use_imputation:
        model = DynamicMultiModalNetWithImputation(
            bio_input_dim=len(bio_features),
            clinical_input_dim=len(clinical_features),
            hidden_dims=args.hidden_dims,
            fusion_dim=args.fusion_dim,
            num_classes=num_classes,
            dropout_rate=args.dropout
        )
        print("使用带模态补全的动态多模态网络")
    else:
        model = DynamicMultiModalNet(
            bio_input_dim=len(bio_features),
            clinical_input_dim=len(clinical_features),
            hidden_dims=args.hidden_dims,
            fusion_dim=args.fusion_dim,
            num_classes=num_classes,
            dropout_rate=args.dropout
        )
        print("使用基础动态多模态网络")
    
    # 如果有多个GPU可用，使用数据并行
    if torch.cuda.device_count() > 1 and args.gpu >= 0:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = torch.nn.DataParallel(model)
    
    # 创建训练器
    trainer = MultiModalTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir
    )
    
    # 训练模型
    print("开始训练...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping
    )
    
    # 绘制训练历史
    trainer.plot_training_history()
    
    # 在测试集上评估模型
    print("\n在测试集上进行评估...")
    label_names = [f"Class_{i}" for i in range(num_classes)]
    test_acc, f1, report = trainer.test(test_loader, label_names)
    
    # 分析不同模态组合的性能
    print("\n分析不同模态组合的性能...")
    modality_performance = analyze_modality_performance(trainer, test_loader)
    
    print("\n训练完成！")


if __name__ == "__main__":
    main()

