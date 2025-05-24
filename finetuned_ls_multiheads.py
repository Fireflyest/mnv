import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os

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

# 设置数据加载和增强
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=.2, hue=.2),
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    transforms.RandomAffine(degrees=(-15, 15)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 加载数据集
dataset = data.HuaLiDataset_Multi(root_dir='./data/huali/train7', transform=transform)

# 划分训练集和验证集
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载LSNet多任务模型
model = load_lsnet_multitask(backbone='lsnet_t', num_classes=5, pretrained=True)

# 将模型移动到设备
model.to(device)

# 冻结骨干网络的所有参数，只训练头部
for name, param in model.backbone.named_parameters():
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
criterion_cls = nn.CrossEntropyLoss()
criterion_logic = nn.BCELoss()
optimizer = optim.AdamW(trainable_params, lr=0.0005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# 创建输出目录
os.makedirs('./out', exist_ok=True)

# 训练模型
def train_model(model, criterion_cls, criterion_logic, optimizer, scheduler, train_loader, test_loader, epochs=50):
    train_losses, test_losses = [], []
    train_accs_cls, train_accs_logic, test_accs_cls, test_accs_logic = [], [], [], []
    best_acc = 0.0

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
        
        # 更新学习率
        scheduler.step()
        
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
        
        # 计算测试指标
        test_loss = test_loss / len(test_loader)
        test_acc_cls = correct_cls / total
        test_acc_logic = correct_logic / total
        test_losses.append(test_loss)
        test_accs_cls.append(test_acc_cls)
        test_accs_logic.append(test_acc_logic)
        
        # 打印当前epoch的结果
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc Class: {train_acc_cls:.4f}, "
              f"Train Acc Logic: {train_acc_logic:.4f}, "
              f"Test Loss: {test_loss:.4f}, "
              f"Test Acc Class: {test_acc_cls:.4f}, "
              f"Test Acc Logic: {test_acc_logic:.4f}")
        
        # 保存最佳模型
        current_acc = test_acc_cls + test_acc_logic
        if epoch > epochs * 0.4 and current_acc > best_acc:
            best_acc = current_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
                'acc_cls': test_acc_cls,
                'acc_logic': test_acc_logic
            }, './out/lsnet_multitask_best.pth')
            print(f"Model saved with accuracy: {best_acc:.4f}")
        
        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
                'acc_cls': test_acc_cls,
                'acc_logic': test_acc_logic
            }, f'./out/lsnet_multitask_epoch_{epoch+1}.pth')
    
    return train_losses, test_losses, train_accs_cls, train_accs_logic, test_accs_cls, test_accs_logic

# 训练模型
train_losses, test_losses, train_accs_cls, train_accs_logic, test_accs_cls, test_accs_logic = train_model(
    model, criterion_cls, criterion_logic, optimizer, scheduler, train_loader, test_loader, epochs=10
)

# 保存最后的模型
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': test_losses[-1],
    'acc_cls': test_accs_cls[-1],
    'acc_logic': test_accs_logic[-1]
}, './out/lsnet_multitask_final.pth')

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
plt.savefig('./out/lsnet_multitask_training_curves.png')
plt.show()

print("训练完成！最终模型和训练曲线已保存。")
