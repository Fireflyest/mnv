import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os

import data
import mobilenetv3_u

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# Configure matplotlib to handle Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font for Chinese characters
plt.rcParams['axes.unicode_minus'] = False    # Ensure minus signs display correctly
plt.rcParams['font.family'] = 'sans-serif'    # Set the font family

# Alternative fonts if SimHei is not available
import matplotlib.font_manager as fm
font_paths = fm.findSystemFonts()
chinese_fonts = [f for f in font_paths if any(name in f.lower() for name in ['simhei', 'microsoftyahei', 'simsun', 'fangsong', 'kaiti'])]
if chinese_fonts:
    plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=chinese_fonts[0]).get_name()] + plt.rcParams['font.sans-serif']

# Make sure output directory exists
os.makedirs('./out/feature_guided', exist_ok=True)

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

# 用于重建任务的预处理
transform_original = transforms.Compose([
    transforms.Resize((224, 224)),  # Match size with encoder input
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

# 加载特征引导的模型
model = mobilenetv3_u.MobileNetV3UNetFeatureLoss(num_classes=5, output_channels=3)
encoder = model.encoder
decoder = model.decoder

print("初始化特征引导UNet模型...")
print(encoder)
print(decoder)

# 冻结特征提取器的层
for param in encoder.features.parameters():
    param.requires_grad = False
for param in encoder.classifier.parameters():
    param.requires_grad = False

# 训练编码器的分类头和下采样层
for param in encoder.head1.parameters():
    param.requires_grad = True
for param in encoder.head2.parameters():
    param.requires_grad = True
for param in encoder.extra_downsample.parameters():
    param.requires_grad = True

# 解码器的所有参数都可训练
for param in decoder.parameters():
    param.requires_grad = True

# 将模型移动到设备
model.to(device)

# 定义损失函数和优化器
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.BCELoss()
feature_loss = mobilenetv3_u.FeatureSimilarityLoss(alpha=0.5)  # 特征相似度损失，alpha控制重建损失和特征相似损失的比例

optimizer = optim.Adam([
    {'params': encoder.head1.parameters()},
    {'params': encoder.head2.parameters()},
    {'params': encoder.extra_downsample.parameters()},
    {'params': decoder.parameters()},  # 整个解码器都训练
], lr=0.001)

# 计算PSNR
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

# 训练模型
def train_model(model, criterion1, criterion2, feature_loss_fn, optimizer, 
                train_loader, test_loader, epochs=50, alpha=0.5):
    train_losses, test_losses = [], []
    train_accs1, train_accs2, test_accs1, test_accs2 = [], [], [], []
    train_psnr, test_psnr = [], []
    train_feature_losses, test_feature_losses = [], []
    best_metric = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct1 = 0
        correct2 = 0
        total = 0
        epoch_psnr = 0
        epoch_feature_loss = 0
        
        for batch_idx, (data, original, target1, target2) in enumerate(train_loader):
            data = data.to(device)
            original = original.to(device)  # 原始图像用于重建任务
            target1 = target1.to(device)
            target2 = target2.to(device).float()
            
            optimizer.zero_grad()
            
            # 前向传播 - 特别注意这里的输出格式
            out1, out2, features, reconstruction, encoder_features, decoder_features = model(data)
            
            # 计算分类损失
            loss1 = criterion1(out1, target1)
            loss2 = criterion2(out2.squeeze(), target2)
            
            # 计算特征引导损失 (包括重建损失和特征相似度损失)
            loss_feature_guided, loss_components = feature_loss_fn(reconstruction, original, encoder_features, decoder_features)
            
            # 总损失 = 分类损失1 + 分类损失2 + 特征引导损失
            total_loss = loss1 + loss2 + loss_feature_guided
            
            total_loss.backward()
            optimizer.step()
            
            # 计算统计数据
            train_loss += total_loss.item()
            epoch_feature_loss += loss_components['feature_loss']
            _, predicted1 = torch.max(out1.data, 1)
            predicted2 = (out2 > 0.5).float()
            total += target1.size(0)
            correct1 += (predicted1 == target1).sum().item()
            correct2 += (predicted2.squeeze() == target2).sum().item()
            
            # 计算当前批次的PSNR
            batch_psnr = calculate_psnr(reconstruction, original)
            epoch_psnr += batch_psnr.item()
            
            if batch_idx % 50 == 0:
                print(f"Train Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss.item():.4f}, "
                      f"Feature Loss: {loss_components['feature_loss']:.4f}, Recon Loss: {loss_components['rec_loss']:.4f}")
        
        # 计算平均指标
        train_losses.append(train_loss / len(train_loader))
        train_accs1.append(correct1 / total)
        train_accs2.append(correct2 / total)
        train_psnr.append(epoch_psnr / len(train_loader))
        train_feature_losses.append(epoch_feature_loss / len(train_loader))
        
        # 验证阶段
        model.eval()
        test_loss = 0
        correct1 = 0
        correct2 = 0
        total = 0
        epoch_psnr = 0
        epoch_feature_loss = 0
        
        # 保存一批验证图像用于可视化
        sample_images = None
        sample_recon = None
        
        with torch.no_grad():
            for batch_idx, (data, original, target1, target2) in enumerate(test_loader):
                data = data.to(device)
                original = original.to(device)
                target1 = target1.to(device)
                target2 = target2.to(device).float()
                
                # 前向传播
                out1, out2, features, reconstruction, encoder_features, decoder_features = model(data)
                
                # 计算分类损失
                loss1 = criterion1(out1, target1)
                loss2 = criterion2(out2.squeeze(), target2)
                
                # 计算特征引导损失
                loss_feature_guided, loss_components = feature_loss_fn(reconstruction, original, encoder_features, decoder_features)
                
                # 总损失
                total_loss = loss1 + loss2 + loss_feature_guided
                
                # 统计数据
                test_loss += total_loss.item()
                epoch_feature_loss += loss_components['feature_loss']
                _, predicted1 = torch.max(out1.data, 1)
                predicted2 = (out2 > 0.5).float()
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
        test_feature_losses.append(epoch_feature_loss / len(test_loader))
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, "
              f"Train Acc Class: {train_accs1[-1]:.4f}, Train Acc Logic: {train_accs2[-1]:.4f}, "
              f"Train PSNR: {train_psnr[-1]:.2f}, Train Feature Loss: {train_feature_losses[-1]:.4f}")
        print(f"Test Loss: {test_losses[-1]:.4f}, "
              f"Test Acc Class: {test_accs1[-1]:.4f}, Test Acc Logic: {test_accs2[-1]:.4f}, "
              f"Test PSNR: {test_psnr[-1]:.2f}, Test Feature Loss: {test_feature_losses[-1]:.4f}")
        
        # 保存最佳模型 (基于分类准确率、PSNR和特征损失的加权平均)
        current_metric = test_accs1[-1] + test_accs2[-1] + 0.01 * test_psnr[-1] - 0.1 * test_feature_losses[-1]
        if epoch > epochs * 0.4 and current_metric > best_metric:
            best_metric = current_metric
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, './out/feature_guided/mobilenetv3_feature_guided_best.pth')
            print(f"模型已保存，加权指标: {best_metric:.4f}")
        
        # 每几个epoch保存一次重建图像的可视化结果
        if (epoch + 1) % 5 == 0 or epoch == 0:
            fig, axes = plt.subplots(4, 2, figsize=(10, 12))
            for i in range(4):
                if i < len(sample_images):
                    # 显示原始图像
                    img = sample_images[i]
                    axes[i, 0].imshow(np.transpose(img.numpy(), (1, 2, 0)))
                    axes[i, 0].set_title(f"原始图像 {i+1}")
                    axes[i, 0].axis('off')
                    
                    # 显示重建图像
                    recon = sample_recon[i]
                    axes[i, 1].imshow(np.transpose(recon.numpy(), (1, 2, 0)))
                    axes[i, 1].set_title(f"重建图像 {i+1}\nPSNR: {calculate_psnr(sample_recon[i], sample_images[i]):.2f}")
                    axes[i, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'./out/feature_guided/epoch_{epoch+1}.png')
            plt.close()
        
        # 保存检查点用于恢复训练
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'test_losses': test_losses,
                'train_accs1': train_accs1,
                'train_accs2': train_accs2,
                'test_accs1': test_accs1,
                'test_accs2': test_accs2,
                'train_psnr': train_psnr,
                'test_psnr': test_psnr,
                'train_feature_losses': train_feature_losses,
                'test_feature_losses': test_feature_losses,
            }, f'./out/feature_guided/checkpoint_epoch_{epoch+1}.pth')

    return {
        'train_losses': train_losses, 
        'test_losses': test_losses, 
        'train_accs1': train_accs1, 
        'train_accs2': train_accs2, 
        'test_accs1': test_accs1, 
        'test_accs2': test_accs2,
        'train_psnr': train_psnr,
        'test_psnr': test_psnr,
        'train_feature_losses': train_feature_losses,
        'test_feature_losses': test_feature_losses
    }

# 训练模型
print(f"开始训练特征引导UNet模型，设备: {device}")
metrics = train_model(
    model, criterion1, criterion2, feature_loss, optimizer, 
    train_loader, test_loader, epochs=300, alpha=0.5)

# 保存最后的模型
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, './out/feature_guided/mobilenetv3_feature_guided_last.pth')

# 绘制各项指标的训练和验证曲线
plt.figure(figsize=(20, 12))

plt.subplot(2, 2, 1)
plt.plot(metrics['train_losses'], label='训练损失')
plt.plot(metrics['test_losses'], label='验证损失')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.legend()
plt.title('总损失曲线')

plt.subplot(2, 2, 2)
plt.plot(metrics['train_accs1'], label='训练准确率(头1)')
plt.plot(metrics['train_accs2'], label='训练准确率(头2)')
plt.plot(metrics['test_accs1'], label='验证准确率(头1)')
plt.plot(metrics['test_accs2'], label='验证准确率(头2)')
plt.xlabel('Epoch')
plt.ylabel('准确率')
plt.legend()
plt.title('分类准确率曲线')

plt.subplot(2, 2, 3)
plt.plot(metrics['train_psnr'], label='训练PSNR')
plt.plot(metrics['test_psnr'], label='验证PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.title('PSNR曲线')

plt.subplot(2, 2, 4)
plt.plot(metrics['train_feature_losses'], label='训练特征损失')
plt.plot(metrics['test_feature_losses'], label='验证特征损失')
plt.xlabel('Epoch')
plt.ylabel('特征损失')
plt.legend()
plt.title('特征相似度损失曲线')

plt.tight_layout()
plt.savefig('./out/feature_guided/training_curves.png')
plt.close()

print("训练完成，结果已保存!")
