import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import data
import mobilenetv3_u

# 配置环境
device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
os.makedirs('./out/feature_guided_plus', exist_ok=True)

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 高级图像增强 - 更强的数据增强策略
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  # 添加垂直翻转
    transforms.RandomRotation(30),    # 更大范围的旋转
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_original = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 加载数据集
print("正在加载数据集...")
dataset = data.HuaLiDataset_Unet_Multi(
    root_dir='./data/huali/train7', 
    transform=transform_train,
    original_transform=transform_original
)

# 划分数据集 - 增加训练集比例
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 数据加载器 - 使用较大的批量和多线程
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# 初始化改进版模型
print("初始化增强特征引导UNet模型...")
model = mobilenetv3_u.MobileNetV3UNetFeatureLossPlus(num_classes=5, output_channels=3)
encoder = model.encoder
decoder = model.decoder

# 冻结部分编码器参数
for param in encoder.features.parameters():
    param.requires_grad = False
for param in encoder.classifier.parameters():
    param.requires_grad = False

# 需要训练的参数
for param in encoder.head1.parameters():
    param.requires_grad = True
for param in encoder.head2.parameters():
    param.requires_grad = True
for param in encoder.extra_downsample.parameters():
    param.requires_grad = True
for param in decoder.parameters():
    param.requires_grad = True

# 将模型移至设备
model.to(device)

# 损失函数
criterion1 = nn.CrossEntropyLoss(label_smoothing=0.1)  # 使用标签平滑
criterion2 = nn.BCELoss()
feature_loss = mobilenetv3_u.EnhancedFeatureLoss(
    content_weight=1.0,
    style_weight=0.1,
    feature_weight=0.5,
    consistency_weight=0.2,
    alpha=0.4
)

# 优化器 - 使用差异化学习率和权重衰减
optimizer = optim.AdamW([
    {'params': encoder.head1.parameters(), 'lr': 0.001},
    {'params': encoder.head2.parameters(), 'lr': 0.001},
    {'params': encoder.extra_downsample.parameters(), 'lr': 0.0005},
    {'params': decoder.parameters(), 'lr': 0.001}
], weight_decay=1e-4)

# 学习率调度器
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

# 计算PSNR
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(100.0).to(img1.device)
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

# 训练循环
def train_epoch(model, loaders, optimizer, criteria, epoch, total_epochs):
    model.train()
    train_loader = loaders['train']
    criterion1, criterion2, feature_loss_fn = criteria
    
    total_loss = 0
    correct1, correct2 = 0, 0
    total_samples = 0
    total_psnr = 0
    total_components = {'rec_loss': 0, 'feature_loss': 0, 'content_loss': 0, 'style_loss': 0}
    
    start_time = time.time()
    
    for batch_idx, (data, original, target1, target2) in enumerate(train_loader):
        data = data.to(device)
        original = original.to(device)
        target1 = target1.to(device)
        target2 = target2.to(device).float()
        
        optimizer.zero_grad()
        
        # 前向传播
        out1, out2, features, reconstruction, encoder_features, decoder_features = model(data)
        
        # 计算分类损失
        loss1 = criterion1(out1, target1)
        loss2 = criterion2(out2.squeeze(), target2)
        
        # 计算增强的特征损失
        loss_feature, loss_components = feature_loss_fn(
            reconstruction, original, encoder_features, decoder_features
        )
        
        # 总损失 - 动态调整权重
        lambda1 = 1.0
        lambda2 = 1.0
        lambda_feature = max(0.5, 2.0 - epoch * 0.05)  # 随训练进行逐渐降低特征损失的权重
        
        total_batch_loss = lambda1 * loss1 + lambda2 * loss2 + lambda_feature * loss_feature
        
        # 反向传播
        total_batch_loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 统计
        total_loss += total_batch_loss.item()
        _, predicted1 = torch.max(out1, 1)
        predicted2 = (out2 > 0.5).float()
        
        correct1 += (predicted1 == target1).sum().item()
        correct2 += (predicted2.squeeze() == target2).sum().item()
        total_samples += target1.size(0)
        
        # 计算PSNR
        batch_psnr = calculate_psnr(reconstruction, original)
        total_psnr += batch_psnr.item()
        
        # 累计组件损失
        for k, v in loss_components.items():
            if k in total_components:
                total_components[k] += v
            else:
                total_components[k] = v
        
        # 打印进度
        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            progress = batch_idx / len(train_loader) * 100
            print(f"Epoch {epoch+1}/{total_epochs} [{batch_idx}/{len(train_loader)} ({progress:.1f}%)] "
                  f"Time: {elapsed:.1f}s | Loss: {total_batch_loss.item():.4f} | "
                  f"Cls1: {loss1.item():.3f} | Cls2: {loss2.item():.3f} | "
                  f"Feat: {loss_components['feature_loss']:.3f} | "
                  f"Content: {loss_components['content_loss']:.3f} | "
                  f"Style: {loss_components['style_loss']:.3f}")
    
    # 计算平均指标
    avg_loss = total_loss / len(train_loader)
    avg_acc1 = correct1 / total_samples
    avg_acc2 = correct2 / total_samples
    avg_psnr = total_psnr / len(train_loader)
    
    # 平均组件损失
    for k in total_components:
        total_components[k] /= len(train_loader)
    
    return {
        'loss': avg_loss,
        'acc1': avg_acc1,
        'acc2': avg_acc2,
        'psnr': avg_psnr,
        'components': total_components
    }

# 验证循环
def validate(model, loaders, criteria, epoch=0):
    model.eval()
    test_loader = loaders['test']
    criterion1, criterion2, feature_loss_fn = criteria
    
    total_loss = 0
    correct1, correct2 = 0, 0
    total_samples = 0
    total_psnr = 0
    total_components = {'rec_loss': 0, 'feature_loss': 0, 'content_loss': 0, 'style_loss': 0}
    
    # 保存示例图像
    sample_images = []
    
    with torch.no_grad():
        for batch_idx, (data, original, target1, target2) in enumerate(test_loader):
            data = data.to(device)
            original = original.to(device)
            target1 = target1.to(device)
            target2 = target2.to(device).float()
            
            # 前向传播
            out1, out2, features, reconstruction, encoder_features, decoder_features = model(data)
            
            # 计算各种损失
            loss1 = criterion1(out1, target1)
            loss2 = criterion2(out2.squeeze(), target2)
            loss_feature, loss_components = feature_loss_fn(
                reconstruction, original, encoder_features, decoder_features
            )
            
            # 总损失
            total_batch_loss = loss1 + loss2 + loss_feature
            total_loss += total_batch_loss.item()
            
            # 累积组件损失
            for k, v in loss_components.items():
                if k in total_components:
                    total_components[k] += v
                else:
                    total_components[k] = v
            
            # 统计准确率
            _, predicted1 = torch.max(out1, 1)
            predicted2 = (out2 > 0.5).float()
            
            correct1 += (predicted1 == target1).sum().item()
            correct2 += (predicted2.squeeze() == target2).sum().item()
            total_samples += target1.size(0)
            
            # 计算PSNR
            batch_psnr = calculate_psnr(reconstruction, original)
            total_psnr += batch_psnr.item()
            
            # 保存第一批次的图像
            if batch_idx == 0:
                for i in range(min(4, data.size(0))):
                    sample_images.append({
                        'original': original[i].cpu().numpy(),
                        'reconstruction': reconstruction[i].cpu().numpy(),
                        'psnr': calculate_psnr(reconstruction[i:i+1], original[i:i+1]).item()
                    })
    
    # 计算平均指标
    avg_loss = total_loss / len(test_loader)
    avg_acc1 = correct1 / total_samples
    avg_acc2 = correct2 / total_samples
    avg_psnr = total_psnr / len(test_loader)
    
    # 平均组件损失
    for k in total_components:
        total_components[k] /= len(test_loader)
    
    return {
        'loss': avg_loss,
        'acc1': avg_acc1,
        'acc2': avg_acc2,
        'psnr': avg_psnr,
        'components': total_components,
        'sample_images': sample_images
    }

# 主训练循环
def train(model, loaders, criteria, optimizer, scheduler, epochs=100, start_epoch=0):
    train_metrics = {'loss': [], 'acc1': [], 'acc2': [], 'psnr': [], 'components': []}
    val_metrics = {'loss': [], 'acc1': [], 'acc2': [], 'psnr': [], 'components': []}
    best_val_metric = float('-inf')
    
    for epoch in range(start_epoch, epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        
        # 训练阶段
        train_results = train_epoch(model, loaders, optimizer, criteria, epoch, epochs)
        
        # 验证阶段
        val_results = validate(model, loaders, criteria, epoch)
        
        # 保存指标
        train_metrics['loss'].append(train_results['loss'])
        train_metrics['acc1'].append(train_results['acc1'])
        train_metrics['acc2'].append(train_results['acc2'])
        train_metrics['psnr'].append(train_results['psnr'])
        train_metrics['components'].append(train_results['components'])
        
        val_metrics['loss'].append(val_results['loss'])
        val_metrics['acc1'].append(val_results['acc1'])
        val_metrics['acc2'].append(val_results['acc2'])
        val_metrics['psnr'].append(val_results['psnr'])
        val_metrics['components'].append(val_results['components'])
        
        # 打印详细结果
        print(f"\n训练结果: 损失: {train_results['loss']:.4f} | "
              f"分类1准确率: {train_results['acc1']:.4f} | 分类2准确率: {train_results['acc2']:.4f} | "
              f"PSNR: {train_results['psnr']:.2f}")
        print(f"验证结果: 损失: {val_results['loss']:.4f} | "
              f"分类1准确率: {val_results['acc1']:.4f} | 分类2准确率: {val_results['acc2']:.4f} | "
              f"PSNR: {val_results['psnr']:.2f}")
        print(f"组件损失(验证): 重建: {val_results['components']['rec_loss']:.4f} | "
              f"特征: {val_results['components']['feature_loss']:.4f} | "
              f"内容: {val_results['components']['content_loss']:.4f} | "
              f"风格: {val_results['components']['style_loss']:.4f}")
        
        # 调整学习率
        scheduler.step(val_results['loss'])
        
        # 保存示例图像
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            fig, axes = plt.subplots(len(val_results['sample_images']), 2, figsize=(10, 12))
            for i, sample in enumerate(val_results['sample_images']):
                axes[i, 0].imshow(np.transpose(sample['original'], (1, 2, 0)))
                axes[i, 0].set_title(f"原始图像 {i+1}")
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(np.transpose(sample['reconstruction'], (1, 2, 0)))
                axes[i, 1].set_title(f"重建图像 {i+1}\nPSNR: {sample['psnr']:.2f}")
                axes[i, 1].axis('off')
                
            plt.tight_layout()
            plt.savefig(f'./out/feature_guided_plus/samples_epoch_{epoch+1}.png')
            plt.close()
        
        # 保存模型 - 以加权组合指标为标准
        val_metric = val_results['acc1'] + val_results['acc2'] + 0.01 * val_results['psnr'] - 0.1 * val_results['components']['feature_loss']
        if (epoch > epochs * 0.2) and (val_metric > best_val_metric):
            best_val_metric = val_metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metric': val_metric,
            }, './out/feature_guided_plus/best_model.pth')
            print(f"已保存最佳模型，验证指标: {val_metric:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_val_metric': best_val_metric,
            }, f'./out/feature_guided_plus/checkpoint_epoch_{epoch+1}.pth')
            
            # 绘制当前进度的训练曲线
            plot_training_curves(train_metrics, val_metrics, epoch+1)
    
    # 保存最终模型
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
    }, './out/feature_guided_plus/final_model.pth')
    
    # 绘制最终训练曲线
    plot_training_curves(train_metrics, val_metrics, epochs)
    
    return train_metrics, val_metrics

# 绘制训练曲线
def plot_training_curves(train_metrics, val_metrics, epochs):
    plt.figure(figsize=(20, 15))
    
    # 绘制总损失
    plt.subplot(3, 2, 1)
    plt.plot(train_metrics['loss'], label='训练损失')
    plt.plot(val_metrics['loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.title('总损失曲线')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制分类准确率
    plt.subplot(3, 2, 2)
    plt.plot(train_metrics['acc1'], label='训练准确率(头1)')
    plt.plot(train_metrics['acc2'], label='训练准确率(头2)')
    plt.plot(val_metrics['acc1'], label='验证准确率(头1)')
    plt.plot(val_metrics['acc2'], label='验证准确率(头2)')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    plt.title('分类准确率曲线')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制PSNR
    plt.subplot(3, 2, 3)
    plt.plot(train_metrics['psnr'], label='训练PSNR')
    plt.plot(val_metrics['psnr'], label='验证PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.title('PSNR曲线')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制重建损失
    plt.subplot(3, 2, 4)
    rec_losses_train = [comp['rec_loss'] for comp in train_metrics['components']]
    rec_losses_val = [comp['rec_loss'] for comp in val_metrics['components']]
    plt.plot(rec_losses_train, label='训练')
    plt.plot(rec_losses_val, label='验证')
    plt.xlabel('Epoch')
    plt.ylabel('重建损失')
    plt.legend()
    plt.title('重建损失曲线')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制特征损失
    plt.subplot(3, 2, 5)
    feat_losses_train = [comp['feature_loss'] for comp in train_metrics['components']]
    feat_losses_val = [comp['feature_loss'] for comp in val_metrics['components']]
    plt.plot(feat_losses_train, label='训练')
    plt.plot(feat_losses_val, label='验证')
    plt.xlabel('Epoch')
    plt.ylabel('特征损失')
    plt.legend()
    plt.title('特征损失曲线')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制内容和风格损失
    plt.subplot(3, 2, 6)
    content_losses_val = [comp['content_loss'] for comp in val_metrics['components']]
    style_losses_val = [comp['style_loss'] for comp in val_metrics['components']]
    plt.plot(content_losses_val, label='内容损失')
    plt.plot(style_losses_val, label='风格损失')
    plt.xlabel('Epoch')
    plt.ylabel('感知损失')
    plt.legend()
    plt.title('内容和风格损失曲线')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f'./out/feature_guided_plus/training_curves_epoch_{epochs}.png')
    plt.close()

# 主程序
if __name__ == "__main__":
    print(f"正在训练增强特征引导UNet模型，使用设备: {device}")
    
    # 数据加载器
    loaders = {
        'train': train_loader,
        'test': test_loader
    }
    
    # 损失函数
    criteria = (criterion1, criterion2, feature_loss)
    
    # 加载checkpoint（如果存在）
    start_epoch = 0
    checkpoint_path = './out/feature_guided_plus/latest_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"继续从 epoch {start_epoch} 训练")
    
    # 训练模型
    train_metrics, val_metrics = train(
        model, loaders, criteria, optimizer, scheduler, 
        epochs=200, start_epoch=start_epoch  # 增加训练轮数以获得更好的结果
    )
    
    print("训练完成！最终模型已保存。")
