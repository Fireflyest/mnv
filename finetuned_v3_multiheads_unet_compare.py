import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader, Subset

import data
import mobilenetv3_u

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 创建目录
os.makedirs('./out/compare', exist_ok=True)

# 命令行参数
parser = argparse.ArgumentParser(description="比较不同UNet变体的性能")
parser.add_argument('--skip_model_path', type=str, default='./out/mobilenetv3_split_best.pth',
                   help='带跳跃连接UNet模型的路径')
parser.add_argument('--feature_model_path', type=str, default='./out/feature_guided/mobilenetv3_feature_guided_best.pth',
                   help='特征引导UNet模型的路径')
parser.add_argument('--num_samples', type=int, default=16,
                   help='用于比较的样本数量')
args = parser.parse_args()

# 预处理
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_original = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 加载测试数据集
dataset = data.HuaLiDataset_Unet_Multi(root_dir='./data/huali/train7', 
                                      transform=transform_test,
                                      original_transform=transform_original)

# 只选择少量样本进行比较
indices = list(range(min(args.num_samples, len(dataset))))
test_dataset = Subset(dataset, indices)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 计算PSNR
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def load_skip_model():
    # 加载带跳跃连接的模型
    model = mobilenetv3_u.MobileNetV3UNetDeep(num_classes=5, output_channels=3)
    checkpoint = torch.load(args.skip_model_path, map_location=device)
    
    # 兼容不同格式的检查点
    if 'encoder' in checkpoint and 'decoder' in checkpoint:
        # 分别加载编码器和解码器
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder.load_state_dict(checkpoint['decoder'])
    elif 'model_state_dict' in checkpoint:
        # 加载整个模型
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    return model

def load_feature_model():
    # 加载特征引导的模型
    model = mobilenetv3_u.MobileNetV3UNetFeatureLoss(num_classes=5, output_channels=3)
    checkpoint = torch.load(args.feature_model_path, map_location=device)
    
    # 兼容不同格式的检查点
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    return model

def create_noskip_model():
    # 创建无跳跃连接的模型 (不加载预训练权重，只用于比较结构差异)
    model = mobilenetv3_u.MobileNetV3UNetNoSkip(num_classes=5, output_channels=3)
    model.to(device)
    model.eval()
    return model

# 加载模型
skip_model = load_skip_model()
feature_model = load_feature_model()
noskip_model = create_noskip_model()

print("模型加载完成，开始进行比较...")

# 创建无跳跃连接的基本UNet（未训练）作为对照
all_images = []
all_psnr_skip = []
all_psnr_feature = []
all_psnr_noskip = []

# 比较重建质量
with torch.no_grad():
    for i, (data, original, _, _) in enumerate(test_loader):
        data = data.to(device)
        original = original.to(device)
        
        # 带跳跃连接的UNet
        _, _, _, recon_skip = skip_model(data)
        
        # 特征引导的UNet
        _, _, _, recon_feature, _, _ = feature_model(data)
        
        # 无跳跃连接的UNet
        _, _, _, recon_noskip = noskip_model(data)
        
        # 计算PSNR
        psnr_skip = calculate_psnr(recon_skip, original).item()
        psnr_feature = calculate_psnr(recon_feature, original).item()
        psnr_noskip = calculate_psnr(recon_noskip, original).item()
        
        all_psnr_skip.append(psnr_skip)
        all_psnr_feature.append(psnr_feature)
        all_psnr_noskip.append(psnr_noskip)
        
        # 保存图像结果
        all_images.append({
            'original': original.detach().cpu().numpy()[0],
            'skip': recon_skip.detach().cpu().numpy()[0],
            'feature': recon_feature.detach().cpu().numpy()[0],
            'noskip': recon_noskip.detach().cpu().numpy()[0],
            'psnr_skip': psnr_skip,
            'psnr_feature': psnr_feature,
            'psnr_noskip': psnr_noskip,
        })

# 可视化结果
print("生成比较结果可视化...")
num_images = len(all_images)
cols = 4
rows = min(4, num_images)  # 最多显示4行
fig = plt.figure(figsize=(cols * 4, rows * 4))

for i in range(rows):
    # 原始图像
    ax = fig.add_subplot(rows, cols, i*cols + 1)
    ax.imshow(np.transpose(all_images[i]['original'], (1, 2, 0)))
    ax.set_title("原始图像")
    ax.axis('off')
    
    # 带跳跃连接的UNet
    ax = fig.add_subplot(rows, cols, i*cols + 2)
    ax.imshow(np.transpose(all_images[i]['skip'], (1, 2, 0)))
    ax.set_title(f"带跳跃连接\nPSNR: {all_images[i]['psnr_skip']:.2f}")
    ax.axis('off')
    
    # 特征引导的UNet
    ax = fig.add_subplot(rows, cols, i*cols + 3)
    ax.imshow(np.transpose(all_images[i]['feature'], (1, 2, 0)))
    ax.set_title(f"特征引导\nPSNR: {all_images[i]['psnr_feature']:.2f}")
    ax.axis('off')
    
    # 无跳跃连接的UNet
    ax = fig.add_subplot(rows, cols, i*cols + 4)
    ax.imshow(np.transpose(all_images[i]['noskip'], (1, 2, 0)))
    ax.set_title(f"无跳跃连接\nPSNR: {all_images[i]['psnr_noskip']:.2f}")
    ax.axis('off')

plt.tight_layout()
plt.savefig('./out/compare/reconstruction_comparison.png')

# 统计结果
avg_psnr_skip = np.mean(all_psnr_skip)
avg_psnr_feature = np.mean(all_psnr_feature)
avg_psnr_noskip = np.mean(all_psnr_noskip)

print("\n===== 重建质量比较结果 =====")
print(f"带跳跃连接的UNet平均PSNR: {avg_psnr_skip:.2f} dB")
print(f"特征引导的UNet平均PSNR: {avg_psnr_feature:.2f} dB")
print(f"无跳跃连接的UNet平均PSNR: {avg_psnr_noskip:.2f} dB")

# 绘制PSNR对比条形图
plt.figure(figsize=(10, 6))
plt.bar(['带跳跃连接', '特征引导', '无跳跃连接'], 
        [avg_psnr_skip, avg_psnr_feature, avg_psnr_noskip],
        color=['green', 'blue', 'red'])
plt.title('不同UNet变体的平均PSNR比较')
plt.ylabel('PSNR (dB)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('./out/compare/psnr_comparison.png')

# 生成特征相似度比较 - 从各个重建中提取特征并比较相似度
print("分析特征相似度...")

def extract_features(model, image):
    # 提取特征的函数
    with torch.no_grad():
        # 使用编码器部分提取特征
        if isinstance(model, mobilenetv3_u.MobileNetV3UNetFeatureLoss):
            _, _, features, *_ = model.encoder(image)
        else:
            _, _, features, *_ = model.encoder(image)
    return features

# 计算特征余弦相似度
def cosine_similarity(feat1, feat2):
    return torch.nn.functional.cosine_similarity(feat1, feat2, dim=1).mean().item()

feature_similarities = []

with torch.no_grad():
    for i, (data, original, _, _) in enumerate(test_loader):
        if i >= rows:  # 只对可视化的样本进行分析
            break
            
        data = data.to(device)
        
        # 提取原始图像的特征
        orig_features = extract_features(skip_model, data)
        
        # 提取重建图像的特征
        recon_skip, _, _, _ = skip_model(data)
        skip_features = extract_features(skip_model, recon_skip)
        
        _, _, _, recon_feature, _, _ = feature_model(data)
        feature_features = extract_features(skip_model, recon_feature)
        
        _, _, _, recon_noskip = noskip_model(data)
        noskip_features = extract_features(skip_model, recon_noskip)
        
        # 计算与原始特征的相似度
        sim_skip = cosine_similarity(orig_features, skip_features)
        sim_feature = cosine_similarity(orig_features, feature_features)
        sim_noskip = cosine_similarity(orig_features, noskip_features)
        
        feature_similarities.append({
            'skip': sim_skip,
            'feature': sim_feature,
            'noskip': sim_noskip
        })

# 打印特征相似度结果
print("\n===== 特征相似度比较 =====")
avg_sim_skip = np.mean([fs['skip'] for fs in feature_similarities])
avg_sim_feature = np.mean([fs['feature'] for fs in feature_similarities])
avg_sim_noskip = np.mean([fs['noskip'] for fs in feature_similarities])

print(f"带跳跃连接的UNet平均特征相似度: {avg_sim_skip:.4f}")
print(f"特征引导的UNet平均特征相似度: {avg_sim_feature:.4f}")
print(f"无跳跃连接的UNet平均特征相似度: {avg_sim_noskip:.4f}")

# 绘制特征相似度对比条形图
plt.figure(figsize=(10, 6))
plt.bar(['带跳跃连接', '特征引导', '无跳跃连接'], 
        [avg_sim_skip, avg_sim_feature, avg_sim_noskip],
        color=['green', 'blue', 'red'])
plt.title('不同UNet变体的特征相似度比较')
plt.ylabel('特征相似度')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('./out/compare/feature_similarity_comparison.png')

print("\n分析完成，结果已保存至 ./out/compare/ 目录")
