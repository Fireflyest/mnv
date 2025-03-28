import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from match_model import ImageMatchingModel
from match_dataset import MatchingDataset
import argparse
import numpy as np
from tqdm import tqdm
import os

# 解析命令行参数
parser = argparse.ArgumentParser(description='训练图像匹配模型')
parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument('--alpha', type=float, default=0.5, help='对比损失权重')
parser.add_argument('--temperature', type=float, default=0.07, help='对比学习温度参数')
parser.add_argument('--data_dir', type=str, default='./data/huali/match3', help='数据目录')
parser.add_argument('--output_dir', type=str, default='checkpoints', help='模型保存目录')
parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例')
args = parser.parse_args()

# 确保输出目录存在
os.makedirs(args.output_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(m):
    """初始化模型权重"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def main():
    # 数据预处理
    transform_p = transforms.Compose([
        transforms.RandomApply([
            transforms.Resize(int(720 * scale)) for scale in [0.5, 0.6, 0.7, 0.8]
        ], p=0.8),
        transforms.RandomCrop(224),  # Random crop
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Random color jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_c = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    full_dataset = MatchingDataset(args.data_dir, transform_p=transform_p, transform_c=transform_c)
    
    # 划分训练集和验证集
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * args.val_split)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"数据集总大小: {dataset_size}, 训练集: {train_size}, 验证集: {val_size}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 创建模型
    model = ImageMatchingModel(pretrained=True, backbone='small', temperature=args.temperature)
    
    # 仅对匹配器部分进行初始化，因为backbone是预训练模型
    if hasattr(model, 'matcher'):
        model.matcher.apply(init_weights)
    elif hasattr(model, 'fc'):
        model.fc.apply(init_weights)
    
    print("模型匹配器权重已初始化")

    # 优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    model = model.to(device)

    # 训练循环
    print("开始训练...")
    print(f"设备: {device}")

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # 训练一个轮次
        train_loss, train_bce, train_contrast, train_acc = train_epoch(model, train_loader, optimizer, args.alpha)
        
        # 验证
        val_loss, val_bce, val_contrast, val_acc = validate(model, val_loader, args.alpha)
        
        # 打印统计信息
        print(f"训练 - 损失: {train_loss:.4f}, BCE: {train_bce:.4f}, 对比: {train_contrast:.4f}, 准确率: {train_acc:.2f}%")
        print(f"验证 - 损失: {val_loss:.4f}, BCE: {val_bce:.4f}, 对比: {val_contrast:.4f}, 准确率: {val_acc:.2f}%")
        
        # 学习率调度 (使用验证损失)
        scheduler.step(val_loss)
        
        # 保存最佳模型 (基于验证损失)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, "matcher_best.pth")
            model.save_matcher(checkpoint_path)
            print(f"模型保存到 {checkpoint_path} (验证损失: {val_loss:.4f})")
        
        # 每个轮次保存一次检查点
        checkpoint_path = os.path.join(args.output_dir, f"matcher_epoch_{epoch+1}.pth")
        model.save_matcher(checkpoint_path)


# 训练函数
def train_epoch(model, loader, optimizer, alpha):
    model.train()
    running_loss = 0.0
    running_bce_loss = 0.0
    running_contrast_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc='Training')
    
    for img1, img2, labels in progress_bar:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        total_loss, bce_loss, contrast_loss = model.compute_loss(img1, img2, labels, alpha)
        running_contrast_loss += contrast_loss.item()
        
        # 计算前向传播结果（不重复计算）
        with torch.no_grad():
            scores = model(img1, img2).squeeze()
        
        # 反向传播和优化
        total_loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += total_loss.item()
        running_bce_loss += bce_loss.item()
        
        # 计算准确率 - 修正：使用已经计算的scores而非重新计算
        preds = (scores > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': running_loss / (progress_bar.n + 1),
            'bce': running_bce_loss / (progress_bar.n + 1),
            'acc': 100 * correct / total
        })
    
    # 返回平均损失和准确率
    avg_loss = running_loss / len(loader)
    avg_bce = running_bce_loss / len(loader)
    avg_contrast = running_contrast_loss / len(loader)
    accuracy = 100 * correct / total
    
    return avg_loss, avg_bce, avg_contrast, accuracy

# 验证函数
def validate(model, loader, alpha):
    model.eval()
    running_loss = 0.0
    running_bce_loss = 0.0
    running_contrast_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc='Validating')
        
        for img1, img2, labels in progress_bar:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            # 计算损失和前向传播结果
            scores = model(img1, img2).squeeze()
            
            total_loss, bce_loss, contrast_loss = model.compute_loss(img1, img2, labels, alpha)
            running_contrast_loss += contrast_loss.item()
            
            # 统计
            running_loss += total_loss.item()
            running_bce_loss += bce_loss.item()
            
            # 计算准确率 - 使用阈值0.5
            preds = (scores > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # 收集所有预测和标签用于详细分析
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
            # 更新进度条
            progress_bar.set_postfix({
                'val_loss': running_loss / (progress_bar.n + 1),
                'val_bce': running_bce_loss / (progress_bar.n + 1),
                'val_acc': 100 * correct / total
            })
    
    # 合并所有批次的预测和标签
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # 计算整体准确率
    accuracy = 100 * (all_preds == all_labels).sum().item() / len(all_labels)
    
    # 打印详细的分类统计
    positive_correct = ((all_preds == 1) & (all_labels == 1)).sum().item()
    negative_correct = ((all_preds == 0) & (all_labels == 0)).sum().item()
    false_positives = ((all_preds == 1) & (all_labels == 0)).sum().item()
    false_negatives = ((all_preds == 0) & (all_labels == 1)).sum().item()
    
    total_positive = (all_labels == 1).sum().item()
    total_negative = (all_labels == 0).sum().item()
    
    print(f"验证集分类详情:")
    print(f"- 正样本: {positive_correct}/{total_positive} 正确 ({100*positive_correct/total_positive:.2f}%)")
    print(f"- 负样本: {negative_correct}/{total_negative} 正确 ({100*negative_correct/total_negative:.2f}%)")
    print(f"- 假阳性: {false_positives}, 假阴性: {false_negatives}")
    
    # 返回平均损失和准确率
    avg_loss = running_loss / len(loader)
    avg_bce = running_bce_loss / len(loader)
    avg_contrast = running_contrast_loss / len(loader)
    
    return avg_loss, avg_bce, avg_contrast, accuracy

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
    print("训练完成！")
