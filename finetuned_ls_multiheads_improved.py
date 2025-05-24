import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import Counter

import data
from lsnet_multi import load_lsnet_multitask

# 设置设备
device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"使用设备: {device}")

# 增强数据增强策略，缓解过拟合
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # 更多尺度变化
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),  # 添加垂直翻转
    transforms.RandomRotation(30),  # 更大的旋转角度
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  # 更强的颜色扰动
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    transforms.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.1), scale=(0.8, 1.2)),  # 更多的仿射变换
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # 更大的擦除区域
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 测试集使用简单变换
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 加载数据集
train_dataset = data.HuaLiDataset_Multi(root_dir='./data/huali/train8', transform=train_transform)

# 创建独立的测试集，使用相同数据源但不同变换
test_dataset = data.HuaLiDataset_Multi(root_dir='./data/huali/test1', transform=test_transform)

# 分析类别分布情况
def analyze_dataset(dataset):
    labels = []
    logic_labels = []
    for _, label, logic in dataset:
        labels.append(label)
        logic_labels.append(logic)
    
    class_counts = Counter(labels)
    logic_counts = Counter(logic_labels)
    
    print("类别分布:")
    for class_idx, count in class_counts.items():
        print(f"类别 {class_idx} ({dataset.classes[class_idx]}): {count} 样本")
    
    print("\n逻辑标签分布:")
    for logic_val, count in logic_counts.items():
        print(f"逻辑值 {logic_val}: {count} 样本")
    
    return class_counts, logic_counts

print("训练集分析:")
train_class_counts, train_logic_counts = analyze_dataset(train_dataset)
print("\n测试集分析:")
test_class_counts, test_logic_counts = analyze_dataset(test_dataset)

# 使用加权采样来平衡类别
def create_weighted_sampler(dataset):
    labels = []
    for _, label, _ in dataset:
        labels.append(label)
    
    # 计算类别权重: 1/频率
    class_counts = np.bincount(labels)
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    
    # 为每个样本分配权重
    sample_weights = weights[labels]
    
    # 创建加权采样器
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

sampler = create_weighted_sampler(train_dataset)

# 使用加权采样器加载训练数据
train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    sampler=sampler,  # 使用加权采样器，不再使用shuffle=True
    num_workers=4,
    pin_memory=True
)

# 测试集数据加载器
test_loader = DataLoader(
    test_dataset, 
    batch_size=32, 
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# 加载LSNet多任务模型
model = load_lsnet_multitask(backbone='lsnet_t', num_classes=5, pretrained=True)

# 添加Dropout以减轻过拟合
model.classification_head = nn.Sequential(
    nn.BatchNorm1d(model.feature_dim),
    nn.Dropout(0.3),  # 增加dropout
    nn.Linear(model.feature_dim, 5)
)

model.logic_head = nn.Sequential(
    nn.BatchNorm1d(model.feature_dim),
    nn.Dropout(0.3),  # 增加dropout
    nn.Linear(model.feature_dim, 1),
    nn.Sigmoid()
)

# 重新初始化模型头部
model._init_weights()

# 将模型移动到设备
model.to(device)

# 选择性地冻结骨干网络参数
# 注意: 对于迁移学习，先解冻更多层以适应新任务
for name, param in model.backbone.named_parameters():
    # 解冻最后两个块，冻结前面的块
    if "blocks3" not in name and "blocks4" not in name:
        param.requires_grad = False

# 确保所有头部是可训练的
for param in model.classification_head.parameters():
    param.requires_grad = True
for param in model.logic_head.parameters():
    param.requires_grad = True

# 列出可训练的参数
trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"可训练参数数量: {sum(p.numel() for p in trainable_params)}")

# 定义损失函数和优化器
# 使用带有类别权重的损失函数来处理不平衡
class_weights = torch.tensor([
    1.0 / (train_class_counts[i] if i in train_class_counts else 1.0) 
    for i in range(5)
], device=device)
class_weights = class_weights / class_weights.sum() * 5  # 归一化权重

criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
criterion_logic = nn.BCELoss()

# 使用更小的学习率和更强的权重衰减来防止过拟合
optimizer = optim.AdamW(trainable_params, lr=0.0001, weight_decay=1e-3)

# 使用学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max', 
    factor=0.5, 
    patience=5,
    verbose=True
)

# 创建输出目录
os.makedirs('./out', exist_ok=True)

# 添加早停策略
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            return False
            
        if val_score > self.best_score + self.delta:
            self.best_score = val_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

# 训练模型
def train_model(model, criterion_cls, criterion_logic, optimizer, scheduler, 
               train_loader, test_loader, epochs=100):
    train_losses, test_losses = [], []
    train_accs_cls, train_accs_logic, test_accs_cls, test_accs_logic = [], [], [], []
    best_acc = 0.0
    early_stopping = EarlyStopping(patience=15)
    
    print("\n开始训练...")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        correct_cls = 0
        correct_logic = 0
        total = 0
        
        for batch_idx, (data, target_cls, target_logic) in enumerate(train_loader):
            data = data.to(device)
            target_cls = target_cls.to(device)
            target_logic = target_logic.to(device).float()
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(data)
            output_cls = outputs['classification']
            output_logic = outputs['logic'].squeeze()
            
            # 计算损失
            loss_cls = criterion_cls(output_cls, target_cls)
            loss_logic = criterion_logic(output_logic, target_logic)
            loss = loss_cls + loss_logic
            
            # 反向传播和优化
            loss.backward()
            # 梯度裁剪以稳定训练
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted_cls = torch.max(output_cls.data, 1)
            predicted_logic = (output_logic > 0.5).float()
            total += target_cls.size(0)
            correct_cls += (predicted_cls == target_cls).sum().item()
            correct_logic += (predicted_logic == target_logic).sum().item()
            
            # 打印每个批次的进度
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        # 计算训练指标
        train_loss = train_loss / len(train_loader)
        train_acc_cls = correct_cls / total
        train_acc_logic = correct_logic / total
        train_losses.append(train_loss)
        train_accs_cls.append(train_acc_cls)
        train_accs_logic.append(train_acc_logic)
        
        # 评估阶段
        model.eval()
        test_loss = 0
        correct_cls = 0
        correct_logic = 0
        total = 0
        
        # 为每个类别分别计算准确率
        class_correct = [0] * 5
        class_total = [0] * 5
        
        with torch.no_grad():
            for data, target_cls, target_logic in test_loader:
                data = data.to(device)
                target_cls = target_cls.to(device)
                target_logic = target_logic.to(device).float()
                
                # 前向传播
                outputs = model(data)
                output_cls = outputs['classification']
                output_logic = outputs['logic'].squeeze()
                
                # 计算损失
                loss_cls = criterion_cls(output_cls, target_cls)
                loss_logic = criterion_logic(output_logic, target_logic)
                loss = loss_cls + loss_logic
                
                # 统计
                test_loss += loss.item()
                _, predicted_cls = torch.max(output_cls.data, 1)
                predicted_logic = (output_logic > 0.5).float()
                total += target_cls.size(0)
                correct_cls += (predicted_cls == target_cls).sum().item()
                correct_logic += (predicted_logic == target_logic).sum().item()
                
                # 统计每个类别的准确率
                for i in range(len(target_cls)):
                    label = target_cls[i]
                    class_total[label] += 1
                    if predicted_cls[i] == label:
                        class_correct[label] += 1
        
        # 计算测试指标
        test_loss = test_loss / len(test_loader)
        test_acc_cls = correct_cls / total
        test_acc_logic = correct_logic / total
        test_losses.append(test_loss)
        test_accs_cls.append(test_acc_cls)
        test_accs_logic.append(test_acc_logic)
        
        # 显示每个类别的准确率
        print("\n各类别准确率:")
        for i in range(5):
            if class_total[i] > 0:
                accuracy = class_correct[i] / class_total[i]
                print(f"类别 {i} ({test_dataset.classes[i]}): {accuracy:.4f} ({class_correct[i]}/{class_total[i]})")
            else:
                print(f"类别 {i} ({test_dataset.classes[i]}): 无测试样本")
        
        # 打印当前epoch的结果
        print(f"\nEpoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc Class: {train_acc_cls:.4f}, "
              f"Train Acc Logic: {train_acc_logic:.4f}, "
              f"Test Loss: {test_loss:.4f}, "
              f"Test Acc Class: {test_acc_cls:.4f}, "
              f"Test Acc Logic: {test_acc_logic:.4f}")
        
        # 调整学习率
        scheduler.step(test_acc_cls)
        
        # 保存最佳模型
        current_acc = test_acc_cls + test_acc_logic
        if current_acc > best_acc:
            best_acc = current_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
                'acc_cls': test_acc_cls,
                'acc_logic': test_acc_logic,
                'class_weights': class_weights
            }, './out/lsnet_multitask_best_improved.pth')
            print(f"模型已保存，准确率: {best_acc:.4f}")
        
        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
                'acc_cls': test_acc_cls,
                'acc_logic': test_acc_logic,
                'class_weights': class_weights
            }, f'./out/lsnet_multitask_epoch_{epoch+1}_improved.pth')
        
        # 检查早停
        if early_stopping(test_acc_cls):
            print(f"早停激活! 最佳验证准确率: {early_stopping.best_score:.4f}")
            break
    
    return train_losses, test_losses, train_accs_cls, train_accs_logic, test_accs_cls, test_accs_logic

# 训练模型
train_losses, test_losses, train_accs_cls, train_accs_logic, test_accs_cls, test_accs_logic = train_model(
    model, criterion_cls, criterion_logic, optimizer, scheduler, train_loader, test_loader, epochs=100
)

# 保存最后的模型
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': test_losses[-1],
    'acc_cls': test_accs_cls[-1],
    'acc_logic': test_accs_logic[-1],
    'class_weights': class_weights
}, './out/lsnet_multitask_final_improved.pth')

# 绘制训练和验证损失曲线
plt.figure(figsize=(15, 5))

# 绘制损失曲线
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

# 绘制分类准确率曲线
plt.subplot(1, 3, 2)
plt.plot(train_accs_cls, label='Train Accuracy')
plt.plot(test_accs_cls, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Classification Accuracy')

# 绘制逻辑判断准确率曲线
plt.subplot(1, 3, 3)
plt.plot(train_accs_logic, label='Train Accuracy')
plt.plot(test_accs_logic, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Logic Accuracy')

plt.tight_layout()
plt.savefig('./out/lsnet_multitask_training_curves_improved.png')
plt.show()

print("训练完成！最终模型和训练曲线已保存。")
