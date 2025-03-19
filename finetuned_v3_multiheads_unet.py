import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

import data
import mobilenetv3_u

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# 设置数据加载器
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

# 用于重建任务的预处理，不应用随机擦除
transform_original = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = data.HuaLiDataset_Unet_Multi(root_dir='./data/huali/train7', transform=transform, 
                                 original_transform=transform_original)

# 划分训练集和验证集
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载 MobileNetV3UNet 模型
model = mobilenetv3_u.MobileNetV3UNet(num_classes=5, output_channels=3)
print(model)

# 冻结特征提取器的层
for param in model.features.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = False

# 训练分类头和解码器部分
for param in model.head1.parameters():
    param.requires_grad = True
for param in model.head2.parameters():
    param.requires_grad = True
for param in model.decoder1.parameters():
    param.requires_grad = True
for param in model.decoder2.parameters():
    param.requires_grad = True
for param in model.decoder3.parameters():
    param.requires_grad = True
for param in model.decoder4.parameters():
    param.requires_grad = True
for param in model.decoder5.parameters():
    param.requires_grad = True

# 将模型移动到设备
model.to(device)

# 定义损失函数和优化器
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.BCELoss()
criterion_recon = nn.MSELoss()  # 重建损失
optimizer = optim.Adam([
    {'params': model.head1.parameters()},
    {'params': model.head2.parameters()},
    {'params': model.decoder1.parameters()},
    {'params': model.decoder2.parameters()},
    {'params': model.decoder3.parameters()},
    {'params': model.decoder4.parameters()},
    {'params': model.decoder5.parameters()},
], lr=0.001)

# 计算PSNR
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

# 训练模型
def train_model(model, criterion1, criterion2, criterion_recon, optimizer, train_loader, test_loader, epochs=50):
    train_losses, test_losses = [], []
    train_accs1, train_accs2, test_accs1, test_accs2 = [], [], [], []
    train_psnr, test_psnr = [], []
    best_metric = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct1 = 0
        correct2 = 0
        total = 0
        epoch_psnr = 0
        
        for batch_idx, (data, original, target1, target2) in enumerate(train_loader):
            data = data.to(device)
            original = original.to(device)  # 原始图像用于重建任务
            target1 = target1.to(device)
            target2 = target2.to(device).float()
            
            optimizer.zero_grad()
            output1, output2, _, reconstruction = model(data)
            
            loss1 = criterion1(output1, target1)
            loss2 = criterion2(output2.squeeze(), target2)
            loss_recon = criterion_recon(reconstruction, original)  # 重建损失
            
            # 总损失 = 分类损失1 + 分类损失2 + λ*重建损失
            lambda_recon = 1.0  # 可调整的权重
            loss = loss1 + loss2 + lambda_recon * loss_recon
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted1 = torch.max(output1.data, 1)
            predicted2 = (output2 > 0.5).float()
            total += target1.size(0)
            correct1 += (predicted1 == target1).sum().item()
            correct2 += (predicted2.squeeze() == target2).sum().item()
            
            # 计算当前批次的PSNR
            batch_psnr = calculate_psnr(reconstruction, original)
            epoch_psnr += batch_psnr.item()
        
        # 计算平均指标
        train_losses.append(train_loss / len(train_loader))
        train_accs1.append(correct1 / total)
        train_accs2.append(correct2 / total)
        train_psnr.append(epoch_psnr / len(train_loader))
        
        # 验证阶段
        model.eval()
        test_loss = 0
        correct1 = 0
        correct2 = 0
        total = 0
        epoch_psnr = 0
        
        # 保存一批验证图像用于可视化
        sample_images = None
        sample_recon = None
        
        with torch.no_grad():
            for batch_idx, (data, original, target1, target2) in enumerate(test_loader):
                data = data.to(device)
                original = original.to(device)
                target1 = target1.to(device)
                target2 = target2.to(device).float()
                
                output1, output2, _, reconstruction = model(data)
                
                loss1 = criterion1(output1, target1)
                loss2 = criterion2(output2.squeeze(), target2)
                loss_recon = criterion_recon(reconstruction, original)
                loss = loss1 + loss2 + lambda_recon * loss_recon
                
                test_loss += loss.item()
                _, predicted1 = torch.max(output1.data, 1)
                predicted2 = (output2 > 0.5).float()
                total += target1.size(0)
                correct1 += (predicted1 == target1).sum().item()
                correct2 += (predicted2.squeeze() == target2).sum().item()
                
                # 计算PSNR
                batch_psnr = calculate_psnr(reconstruction, original)
                epoch_psnr += batch_psnr.item()
                
                # 保存第一批次的图像用于可视化
                if batch_idx == 0:
                    sample_images = original.detach().cpu()
                    sample_recon = reconstruction.detach().cpu()
        
        test_losses.append(test_loss / len(test_loader))
        test_accs1.append(correct1 / total)
        test_accs2.append(correct2 / total)
        test_psnr.append(epoch_psnr / len(test_loader))
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, "
              f"Train Acc Class: {train_accs1[-1]:.4f}, Train Acc Logic: {train_accs2[-1]:.4f}, "
              f"Train PSNR: {train_psnr[-1]:.2f}, Test Loss: {test_losses[-1]:.4f}, "
              f"Test Acc Class: {test_accs1[-1]:.4f}, Test Acc Logic: {test_accs2[-1]:.4f}, "
              f"Test PSNR: {test_psnr[-1]:.2f}")
        
        # 保存最佳模型 (基于分类准确率和PSNR的加权平均)
        current_metric = test_accs1[-1] + test_accs2[-1] + 0.01 * test_psnr[-1]
        if epoch > epochs * 0.4 and current_metric > best_metric:
            best_metric = current_metric
            torch.save(model.state_dict(), './out/mobilenetv3_unet_best.pth')
            print(f"Model saved with combined metric: {best_metric:.4f}")
        
        # 每几个epoch保存一次重建图像的可视化结果
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # 将图像反归一化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
            fig, axes = plt.subplots(4, 2, figsize=(10, 12))
            for i in range(4):
                if i < len(sample_images):
                    # 显示原始图像
                    img = sample_images[i]
                    axes[i, 0].imshow(np.transpose((img).numpy(), (1, 2, 0)))
                    axes[i, 0].set_title(f"Original {i+1}")
                    axes[i, 0].axis('off')
                    
                    # 显示重建图像
                    recon = sample_recon[i]
                    axes[i, 1].imshow(np.transpose((recon).numpy(), (1, 2, 0)))
                    axes[i, 1].set_title(f"Reconstructed {i+1}")
                    axes[i, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'./out/reconstruction/epoch_{epoch+1}.png')
            plt.close()

    return train_losses, test_losses, train_accs1, train_accs2, test_accs1, test_accs2, train_psnr, test_psnr

# 训练模型
train_losses, test_losses, train_accs1, train_accs2, test_accs1, test_accs2, train_psnr, test_psnr = train_model(
    model, criterion1, criterion2, criterion_recon, optimizer, train_loader, test_loader, epochs=100)

# 保存最后的模型
torch.save(model.state_dict(), './out/mobilenetv3_unet_last.pth')

# 绘制训练和验证损失曲线
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

# 绘制训练和验证准确率曲线
plt.subplot(1, 3, 2)
plt.plot(train_accs1, label='Train Accuracy Head 1')
plt.plot(train_accs2, label='Train Accuracy Head 2')
plt.plot(test_accs1, label='Test Accuracy Head 1')
plt.plot(test_accs2, label='Test Accuracy Head 2')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curves')

# 绘制PSNR曲线
plt.subplot(1, 3, 3)
plt.plot(train_psnr, label='Train PSNR')
plt.plot(test_psnr, label='Test PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.title('PSNR Curves')

plt.tight_layout()
plt.savefig('./out/mobilenetv3_unet_training_curves.png')
plt.close()