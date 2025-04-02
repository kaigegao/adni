import os
import subprocess
from pathlib import Path

# 确定工作目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 确保安装了必要的包
try:
    import matplotlib
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    subprocess.run(['pip', 'install', 'matplotlib', 'numpy', 'pillow'])
    import matplotlib
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

print(f"当前工作目录: {os.getcwd()}")

# 创建流程图
def create_process_flow_diagram():
    # 创建空白图像
    width, height = 1200, 1600
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # 尝试加载字体，如果不可用则使用默认字体
    try:
        font_title = ImageFont.truetype('simhei.ttf', 28)
        font_normal = ImageFont.truetype('simhei.ttf', 20)
        font_small = ImageFont.truetype('simhei.ttf', 16)
    except IOError:
        font_title = ImageFont.load_default()
        font_normal = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # 添加标题
    title = "ADNI动态神经网络训练流程图"
    draw.text((width//2-200, 30), title, fill="black", font=font_title)
    
    # 绘制主要部分框架
    draw.rectangle([(50, 90), (width-50, 250)], outline="black", width=2)
    draw.text((60, 100), "数据处理阶段", fill="black", font=font_title)
    
    draw.rectangle([(50, 280), (width-50, 700)], outline="black", width=2)
    draw.text((60, 290), "模型架构", fill="black", font=font_title)
    
    draw.rectangle([(50, 730), (width-50, 900)], outline="black", width=2)
    draw.text((60, 740), "训练流程", fill="black", font=font_title)
    
    draw.rectangle([(50, 930), (width-50, 1100)], outline="black", width=2)
    draw.text((60, 940), "评估阶段", fill="black", font=font_title)
    
    # 数据处理阶段详情
    data_process_items = [
        "1. 数据加载: 读取生物样本数据(biospecimen)和临床数据(clinical)",
        "2. 数据预处理: 处理缺失值、分类变量编码、特征标准化",
        "3. 数据集划分: 训练集(70%)、验证集(10%)、测试集(20%)",
        "4. 模态缺失模拟: 随机丢弃部分模态以模拟真实场景中的缺失情况"
    ]
    
    y_pos = 140
    for item in data_process_items:
        draw.text((70, y_pos), item, fill="black", font=font_normal)
        y_pos += 30
    
    # 模型架构详情
    # 基础模型
    draw.text((70, 330), "基础动态多模态网络:", fill="black", font=font_title)
    
    model_diagram1_y = 380
    model_components1 = [
        "生物样本数据 → 生物样本编码器(MLP) →",
        "                                      ↘",
        "                                        注意力融合机制 → 分类器(MLP) → 预测结果",
        "                                      ↗",
        "临床数据 → 临床数据编码器(MLP) →"
    ]
    
    for i, component in enumerate(model_components1):
        draw.text((70, model_diagram1_y + i*30), component, fill="black", font=font_normal)
    
    # 带补全的模型
    draw.text((70, 540), "带模态补全的动态多模态网络:", fill="black", font=font_title)
    
    model_diagram2_y = 590
    model_components2 = [
        "生物样本数据 → 生物样本编码器 →",
        "                                ↘",
        "                                  注意力融合 → 模态补全模块 → 加权融合 → 分类器 → 预测结果",
        "                                ↗",
        "临床数据 → 临床数据编码器 →"
    ]
    
    for i, component in enumerate(model_components2):
        draw.text((70, model_diagram2_y + i*30), component, fill="black", font=font_normal)
    
    # 训练流程
    training_steps = [
        "1. 优化器: Adam (学习率=0.001, 权重衰减=1e-5)",
        "2. 损失函数: 交叉熵损失 (分类任务)",
        "3. 学习率调度: ReduceLROnPlateau (验证损失不下降时减小学习率)",
        "4. 早停策略: 当验证损失连续10个epoch未改善时停止训练",
        "5. 保存最佳模型: 根据验证集性能保存最佳模型权重"
    ]
    
    y_pos = 780
    for item in training_steps:
        draw.text((70, y_pos), item, fill="black", font=font_normal)
        y_pos += 25
    
    # 评估阶段
    evaluation_steps = [
        "1. 全模态评估: 使用所有可用模态进行测试",
        "2. 单模态评估: 分别只使用生物样本数据或临床数据进行测试",
        "3. 模态缺失评估: 评估在不同模态缺失情况下的性能",
        "4. 性能指标: 准确率、F1分数、混淆矩阵、分类报告",
        "5. 结果可视化: 不同模态组合的性能对比图表"
    ]
    
    y_pos = 980
    for item in evaluation_steps:
        draw.text((70, y_pos), item, fill="black", font=font_normal)
        y_pos += 25
    
    # 底部说明
    notes = [
        "注: 1. 动态神经网络能够根据输入的可用模态自适应调整处理流程",
        "    2. 模态补全功能通过模态间的相互映射来填补缺失的模态信息",
        "    3. 注意力融合机制根据不同模态的重要性进行加权融合",
        "    4. 早停机制可有效防止模型过拟合训练数据"
    ]
    
    y_pos = 1150
    for note in notes:
        draw.text((70, y_pos), note, fill="black", font=font_small)
        y_pos += 25
    
    # 保存图像
    output_path = 'adni_dynamic_network_flow.png'
    image.save(output_path)
    print(f"已创建流程图: {output_path}")

# 创建基础动态多模态网络结构图
def create_basic_model_diagram():
    # 创建空白图像
    width, height = 1200, 1000
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # 尝试加载字体
    try:
        font_title = ImageFont.truetype('simhei.ttf', 32)
        font_subtitle = ImageFont.truetype('simhei.ttf', 24)
        font_normal = ImageFont.truetype('simhei.ttf', 20)
        font_small = ImageFont.truetype('simhei.ttf', 16)
    except IOError:
        font_title = ImageFont.load_default()
        font_subtitle = ImageFont.load_default()
        font_normal = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # 添加标题
    title = "基础动态多模态网络结构图"
    draw.text((width//2-200, 30), title, fill="black", font=font_title)
    
    # 绘制输入层
    input_y = 120
    input_height = 80
    bio_color = "#D6EAF8"  # 浅蓝色
    clinical_color = "#D5F5E3"  # 浅绿色
    mask_color = "#FCF3CF"  # 浅黄色
    
    # 生物样本输入框
    draw.rectangle([(100, input_y), (350, input_y+input_height)], fill=bio_color, outline="black", width=2)
    draw.text((120, input_y+30), "生物样本数据输入", fill="black", font=font_normal)
    
    # 临床数据输入框
    draw.rectangle([(400, input_y), (650, input_y+input_height)], fill=clinical_color, outline="black", width=2)
    draw.text((430, input_y+30), "临床数据输入", fill="black", font=font_normal)
    
    # 模态掩码输入框
    draw.rectangle([(700, input_y), (950, input_y+input_height)], fill=mask_color, outline="black", width=2)
    draw.text((740, input_y+30), "模态掩码输入", fill="black", font=font_normal)
    draw.text((700, input_y+input_height+10), "[1,1]:全模态 [1,0]:仅生物 [0,1]:仅临床", fill="black", font=font_small)
    
    # 绘制编码器层
    encoder_y = 250
    encoder_height = 120
    
    draw.rectangle([(100, encoder_y), (350, encoder_y+encoder_height)], fill="#D6EAF8", outline="black", width=2)
    draw.text((130, encoder_y+10), "生物样本编码器", fill="black", font=font_subtitle)
    draw.text((110, encoder_y+50), "多层感知机 (MLP)", fill="black", font=font_normal)
    draw.text((110, encoder_y+80), "输入→隐藏层→特征表示", fill="black", font=font_small)
    
    draw.rectangle([(400, encoder_y), (650, encoder_y+encoder_height)], fill="#D5F5E3", outline="black", width=2)
    draw.text((430, encoder_y+10), "临床数据编码器", fill="black", font=font_subtitle)
    draw.text((410, encoder_y+50), "多层感知机 (MLP)", fill="black", font=font_normal)
    draw.text((410, encoder_y+80), "输入→隐藏层→特征表示", fill="black", font=font_small)
    
    # 连接线 - 从输入到编码器
    draw.line([(225, input_y+input_height), (225, encoder_y)], fill="black", width=2)
    draw.line([(525, input_y+input_height), (525, encoder_y)], fill="black", width=2)
    
    # 绘制融合层
    fusion_y = 450
    fusion_height = 140
    
    draw.rectangle([(250, fusion_y), (650, fusion_y+fusion_height)], fill="#E8DAEF", outline="black", width=2)
    draw.text((370, fusion_y+10), "注意力融合模块", fill="black", font=font_subtitle)
    
    fusion_details = [
        "1. 计算每个模态的注意力分数",
        "2. 应用模态掩码屏蔽缺失模态",
        "3. Softmax获取注意力权重",
        "4. 加权求和融合特征"
    ]
    
    for i, detail in enumerate(fusion_details):
        draw.text((270, fusion_y+50+i*25), detail, fill="black", font=font_normal)
    
    # 连接线 - 从编码器到融合
    draw.line([(225, encoder_y+encoder_height), (225, fusion_y)], fill="black", width=2)
    draw.line([(225, fusion_y), (250, fusion_y+fusion_height//2)], fill="black", width=2)
    
    draw.line([(525, encoder_y+encoder_height), (525, fusion_y)], fill="black", width=2)
    draw.line([(525, fusion_y), (650, fusion_y+fusion_height//2)], fill="black", width=2)
    
    # 连接线 - 从掩码到融合
    draw.line([(825, input_y+input_height), (825, fusion_y+fusion_height//2)], fill="black", width=2)
    draw.line([(825, fusion_y+fusion_height//2), (650, fusion_y+fusion_height//2)], fill="black", width=2)
    
    # 绘制分类器层
    classifier_y = 670
    classifier_height = 120
    
    draw.rectangle([(250, classifier_y), (650, classifier_y+classifier_height)], fill="#FADBD8", outline="black", width=2)
    draw.text((400, classifier_y+10), "分类器", fill="black", font=font_subtitle)
    draw.text((280, classifier_y+50), "多层感知机，用于最终分类预测", fill="black", font=font_normal)
    draw.text((280, classifier_y+80), "融合特征→隐藏层→输出类别概率", fill="black", font=font_normal)
    
    # 连接线 - 从融合到分类器
    draw.line([(450, fusion_y+fusion_height), (450, classifier_y)], fill="black", width=2)
    
    # 绘制输出层
    output_y = 850
    output_height = 80
    
    draw.rectangle([(250, output_y), (650, output_y+output_height)], fill="#F9E79F", outline="black", width=2)
    draw.text((400, output_y+30), "预测结果", fill="black", font=font_subtitle)
    
    # 连接线 - 从分类器到输出
    draw.line([(450, classifier_y+classifier_height), (450, output_y)], fill="black", width=2)
    
    # 动态特性标注
    dynamics_notes = [
        "动态特性1: 根据掩码自动调整处理流程",
        "动态特性2: 注意力机制动态加权各模态",
        "动态特性3: 适应任意模态组合的输入"
    ]
    
    for i, note in enumerate(dynamics_notes):
        draw.text((700, 350+i*40), note, fill="red", font=font_normal)
        # 添加指向线
        if i == 0:
            draw.line([(700, 350+i*40+10), (650, fusion_y+30)], fill="red", width=2)
        elif i == 1:
            draw.line([(700, 350+i*40+10), (650, fusion_y+70)], fill="red", width=2)
        elif i == 2:
            draw.line([(700, 350+i*40+10), (650, fusion_y+110)], fill="red", width=2)
    
    # 保存图像
    output_path = 'adni_basic_dynamic_network.png'
    image.save(output_path)
    print(f"已创建基础动态多模态网络结构图: {output_path}")

# 创建带模态补全的动态多模态网络结构图
def create_imputation_model_diagram():
    # 创建空白图像
    width, height = 1200, 1200
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # 尝试加载字体
    try:
        font_title = ImageFont.truetype('simhei.ttf', 32)
        font_subtitle = ImageFont.truetype('simhei.ttf', 24)
        font_normal = ImageFont.truetype('simhei.ttf', 20)
        font_small = ImageFont.truetype('simhei.ttf', 16)
    except IOError:
        font_title = ImageFont.load_default()
        font_subtitle = ImageFont.load_default()
        font_normal = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # 添加标题
    title = "带模态补全的动态多模态网络结构图"
    draw.text((width//2-250, 30), title, fill="black", font=font_title)
    
    # 绘制输入层
    input_y = 120
    input_height = 80
    bio_color = "#D6EAF8"  # 浅蓝色
    clinical_color = "#D5F5E3"  # 浅绿色
    mask_color = "#FCF3CF"  # 浅黄色
    
    # 生物样本输入框
    draw.rectangle([(100, input_y), (350, input_y+input_height)], fill=bio_color, outline="black", width=2)
    draw.text((120, input_y+30), "生物样本数据输入", fill="black", font=font_normal)
    
    # 临床数据输入框
    draw.rectangle([(400, input_y), (650, input_y+input_height)], fill=clinical_color, outline="black", width=2)
    draw.text((430, input_y+30), "临床数据输入", fill="black", font=font_normal)
    
    # 模态掩码输入框
    draw.rectangle([(700, input_y), (950, input_y+input_height)], fill=mask_color, outline="black", width=2)
    draw.text((740, input_y+30), "模态掩码输入", fill="black", font=font_normal)
    draw.text((700, input_y+input_height+10), "[1,1]:全模态 [1,0]:仅生物 [0,1]:仅临床", fill="black", font=font_small)
    
    # 绘制编码器层
    encoder_y = 250
    encoder_height = 120
    
    draw.rectangle([(100, encoder_y), (350, encoder_y+encoder_height)], fill="#D6EAF8", outline="black", width=2)
    draw.text((130, encoder_y+10), "生物样本编码器", fill="black", font=font_subtitle)
    draw.text((110, encoder_y+50), "多层感知机 (MLP)", fill="black", font=font_normal)
    draw.text((110, encoder_y+80), "输入→隐藏层→特征表示", fill="black", font=font_small)
    
    draw.rectangle([(400, encoder_y), (650, encoder_y+encoder_height)], fill="#D5F5E3", outline="black", width=2)
    draw.text((430, encoder_y+10), "临床数据编码器", fill="black", font=font_subtitle)
    draw.text((410, encoder_y+50), "多层感知机 (MLP)", fill="black", font=font_normal)
    draw.text((410, encoder_y+80), "输入→隐藏层→特征表示", fill="black", font=font_small)
    
    # 连接线 - 从输入到编码器
    draw.line([(225, input_y+input_height), (225, encoder_y)], fill="black", width=2)
    draw.line([(525, input_y+input_height), (525, encoder_y)], fill="black", width=2)
    
    # 绘制融合层
    fusion_y = 430
    fusion_height = 100
    
    draw.rectangle([(250, fusion_y), (650, fusion_y+fusion_height)], fill="#E8DAEF", outline="black", width=2)
    draw.text((370, fusion_y+10), "注意力融合模块", fill="black", font=font_subtitle)
    draw.text((270, fusion_y+50), "动态融合可用模态的特征", fill="black", font=font_normal)
    
    # 连接线 - 从编码器到融合
    draw.line([(225, encoder_y+encoder_height), (225, fusion_y+fusion_height//2)], fill="black", width=2)
    draw.line([(225, fusion_y+fusion_height//2), (250, fusion_y+fusion_height//2)], fill="black", width=2)
    
    draw.line([(525, encoder_y+encoder_height), (525, fusion_y+fusion_height//2)], fill="black", width=2)
    draw.line([(525, fusion_y+fusion_height//2), (650, fusion_y+fusion_height//2)], fill="black", width=2)
    
    # 连接线 - 从掩码到融合
    draw.line([(825, input_y+input_height), (825, fusion_y+fusion_height//2)], fill="black", width=2)
    draw.line([(825, fusion_y+fusion_height//2), (650, fusion_y+fusion_height//2)], fill="black", width=2)
    
    # 绘制模态补全层
    imputation_y = 590
    imputation_height = 160
    
    draw.rectangle([(150, imputation_y), (750, imputation_y+imputation_height)], fill="#FADBD8", outline="black", width=2)
    draw.text((370, imputation_y+10), "模态补全网络", fill="black", font=font_subtitle)
    
    imputation_details = [
        "1. 检查每个样本的模态可用性",
        "2. 若生物样本模态缺失: 使用临床→生物映射补全",
        "3. 若临床模态缺失: 使用生物→临床映射补全",
        "4. 通过可学习参数α平衡原始和补全特征"
    ]
    
    for i, detail in enumerate(imputation_details):
        draw.text((170, imputation_y+50+i*25), detail, fill="black", font=font_normal)
    
    # 连接线 - 从融合到补全
    draw.line([(450, fusion_y+fusion_height), (450, imputation_y)], fill="black", width=2)
    
    # 绘制特征平衡层
    balance_y = 810
    balance_height = 100
    
    draw.rectangle([(250, balance_y), (650, balance_y+balance_height)], fill="#D2B4DE", outline="black", width=2)
    draw.text((310, balance_y+10), "特征平衡与最终融合", fill="black", font=font_subtitle)
    draw.text((270, balance_y+50), "α*原始特征 + (1-α)*补全特征", fill="black", font=font_normal)
    
    # 连接线 - 从补全到平衡
    draw.line([(450, imputation_y+imputation_height), (450, balance_y)], fill="black", width=2)
    
    # 绘制分类器层
    classifier_y = 970
    classifier_height = 80
    
    draw.rectangle([(250, classifier_y), (650, classifier_y+classifier_height)], fill="#F9E79F", outline="black", width=2)
    draw.text((400, classifier_y+10), "分类器", fill="black", font=font_subtitle)
    draw.text((280, classifier_y+50), "最终预测类别概率分布", fill="black", font=font_normal)
    
    # 连接线 - 从平衡到分类器
    draw.line([(450, balance_y+balance_height), (450, classifier_y)], fill="black", width=2)
    
    # 动态特性标注
    dynamics_notes = [
        "动态特性1: 模态可用性自适应",
        "动态特性2: 按需激活模态补全",
        "动态特性3: 自动平衡原始与补全信息",
        "动态特性4: 利用模态间相互关系"
    ]
    
    for i, note in enumerate(dynamics_notes):
        draw.text((800, 590+i*40), note, fill="red", font=font_normal)
        # 添加指向线
        if i == 0:
            draw.line([(800, 590+i*40+10), (750, imputation_y+40)], fill="red", width=2)
        elif i == 1:
            draw.line([(800, 590+i*40+10), (750, imputation_y+80)], fill="red", width=2)
        elif i == 2:
            draw.line([(800, 590+i*40+10), (750, imputation_y+120)], fill="red", width=2)
        elif i == 3:
            draw.line([(800, 590+i*40+10), (650, balance_y+50)], fill="red", width=2)
    
    # 保存图像
    output_path = 'adni_imputation_dynamic_network.png'
    image.save(output_path)
    print(f"已创建带模态补全的动态多模态网络结构图: {output_path}")

# 创建图
if not os.path.exists('adni_dynamic_network_flow.png'):
    create_process_flow_diagram()

if not os.path.exists('adni_basic_dynamic_network.png'):
    create_basic_model_diagram()

if not os.path.exists('adni_imputation_dynamic_network.png'):
    create_imputation_model_diagram()
else:
    # 如果图片已存在，则重新生成所有图片
    create_process_flow_diagram()
    create_basic_model_diagram()
    create_imputation_model_diagram()

print("所有图表已创建完成！") 