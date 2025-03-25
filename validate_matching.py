import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from match_model import ImageMatchingModel
from match_dataset import MatchingDataset
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import random
from PIL import Image

# 解析命令行参数
parser = argparse.ArgumentParser(description='验证图像匹配模型')
parser.add_argument('--model_path', type=str, default='./checkpoints/contrastive_best.pth', help='模型检查点路径')
parser.add_argument('--matcher', type=str, default='contrastive', choices=['mlp', 'cosine', 'euclidean', 'attention', 'combined', 'contrastive'], help='匹配器类型')
parser.add_argument('--data_dir', type=str, default='./data/huali/match1', help='数据目录')
parser.add_argument('--temperature', type=float, default=0.07, help='对比学习温度参数')
parser.add_argument('--num_samples', type=int, default=10, help='要显示的样本数')
parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def unnormalize(tensor):
    """将标准化的张量转换回原始图像"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def visualize_predictions(model, dataset, num_samples=10):
    """可视化模型预测结果"""
    model.eval()
    
    # 创建索引列表并随机打乱
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    # 设置图表大小
    plt.figure(figsize=(15, num_samples * 3))
    
    sample_count = 0
    i = 0
    
    while sample_count < num_samples and i < len(indices):
        idx = indices[i]
        img1, img2, label = dataset[idx]
        
        # 使用模型进行预测
        with torch.no_grad():
            img1_batch = img1.unsqueeze(0).to(device)
            img2_batch = img2.unsqueeze(0).to(device)
            output = model(img1_batch, img2_batch).item()
            prediction = int(output > 0.5)
            confidence = output if prediction == 1 else 1 - output
        
        # 转换图像以显示
        img1_display = unnormalize(img1).permute(1, 2, 0).cpu().numpy()
        img1_display = np.clip(img1_display, 0, 1)
        
        img2_display = unnormalize(img2).permute(1, 2, 0).cpu().numpy()
        img2_display = np.clip(img2_display, 0, 1)
        
        # 显示图像和预测
        plt.subplot(num_samples, 3, sample_count * 3 + 1)
        plt.imshow(img1_display)
        plt.title("img 1")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, sample_count * 3 + 2)
        plt.imshow(img2_display)
        plt.title("img 2")
        plt.axis('off')
        
        # 结果面板
        plt.subplot(num_samples, 3, sample_count * 3 + 3)
        plt.text(0.5, 0.5, 
                 f"Prediction: {'1' if prediction == 1 else '0'} ({confidence:.2f})\n"
                 f"GroundTrue: {'1' if label == 1 else '0'}\n"
                 f"{'true' if prediction == label else 'false'}",
                 ha='center', va='center', fontsize=12)
        plt.axis('off')
        
        sample_count += 1
        i += 1
    
    plt.tight_layout()
    plt.savefig('./out/match_validation_results.png', dpi=300)

def main():
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    dataset = MatchingDataset(args.data_dir, transform=transform)
    print(f"数据集大小: {len(dataset)} 个样本")
    
    # 创建数据加载器 (仅用于整体评估)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 创建模型
    model = ImageMatchingModel(pretrained=True, backbone='small', matcher_type=args.matcher, temperature=args.temperature)
    
    # 加载检查点
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已加载，来自检查点: {args.model_path}")
    else:
        print(f"错误：找不到检查点 {args.model_path}")
        return

    model = model.to(device)
    model.eval()
    
    # 计算整体模型指标
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for img1, img2, labels in loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            outputs = model(img1, img2).squeeze()
            preds = (outputs > 0.5).float()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    accuracy = 100 * correct / total
    
    # 计算更详细的指标
    positive_correct = ((all_preds == 1) & (all_labels == 1)).sum().item()
    negative_correct = ((all_preds == 0) & (all_labels == 0)).sum().item()
    false_positives = ((all_preds == 1) & (all_labels == 0)).sum().item()
    false_negatives = ((all_preds == 0) & (all_labels == 1)).sum().item()
    
    total_positive = (all_labels == 1).sum().item()
    total_negative = (all_labels == 0).sum().item()
    
    print(f"模型整体准确率: {accuracy:.2f}%")
    print(f"正样本: {positive_correct}/{total_positive} 正确 ({100*positive_correct/total_positive if total_positive > 0 else 0:.2f}%)")
    print(f"负样本: {negative_correct}/{total_negative} 正确 ({100*negative_correct/total_negative if total_negative > 0 else 0:.2f}%)")
    print(f"假阳性: {false_positives}, 假阴性: {false_negatives}")
    
    # 可视化随机样本的预测结果
    print(f"\n可视化 {args.num_samples} 个随机样本的预测结果...")
    visualize_predictions(model, dataset, num_samples=args.num_samples)
    print("可视化完成，结果已保存为 'validation_results.png'")

if __name__ == '__main__':
    main()
