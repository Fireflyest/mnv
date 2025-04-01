import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import random
from PIL import Image
from tqdm import tqdm
from match_dataset import MatchingDataset

# 解析命令行参数
parser = argparse.ArgumentParser(description='使用SIFT特征进行图像匹配')
parser.add_argument('--data_dir', type=str, default='./data/huali/match3', help='数据目录')
parser.add_argument('--num_samples', type=int, default=10, help='要显示的样本数')
parser.add_argument('--threshold', type=float, default=0.7, help='SIFT匹配阈值')
parser.add_argument('--min_matches', type=int, default=10, help='判定为匹配的最小特征点数')
parser.add_argument('--output_dir', type=str, default='./out', help='输出目录')
args = parser.parse_args()

def sift_match(img1, img2, threshold=0.7):
    """
    使用SIFT特征和KNN匹配两幅图像
    
    参数:
    - img1, img2: 输入图像 (OpenCV格式)
    - threshold: Lowe's比例测试的阈值
    
    返回:
    - 匹配数量
    - 两幅图像拼接结果 (用于可视化)
    """
    # 确保图像是uint8类型
    if img1.dtype != np.uint8:
        img1 = (img1 * 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = (img2 * 255).astype(np.uint8)
    
    # 保存原始图像用于后续显示
    img1_display = img1.copy()
    img2_display = img2.copy()
    
    # 确保图像是灰度或BGR格式
    if len(img1.shape) == 2 or img1.shape[2] == 1:
        img1_gray = img1
    else:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        
    if len(img2.shape) == 2 or img2.shape[2] == 1:
        img2_gray = img2
    else:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # 初始化SIFT检测器
    sift = cv2.SIFT_create()
    
    # 找到关键点和描述符
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    
    # 如果没有检测到关键点或描述符，返回0匹配
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        if len(img1.shape) == 3 and img1.shape[2] == 3 and len(img2.shape) == 3 and img2.shape[2] == 3:
            concat_img = np.hstack((img1_display, img2_display))
        else:
            # 如果是灰度图像，需要转换为3通道后连接
            if len(img1.shape) == 2:
                img1_display = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
            else:
                img1_display = img1
            
            if len(img2.shape) == 2:
                img2_display = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
            else:
                img2_display = img2
            
            concat_img = np.hstack((img1_display, img2_display))
        
        return 0, concat_img
    
    # 使用FLANN匹配器进行匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # 进行KNN匹配
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except:
        # 无法匹配的情况
        concat_img = np.hstack((img1_display, img2_display))
        return 0, concat_img
    
    # 应用Lowe's比例测试
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)
    
    # 绘制匹配结果 (OpenCV默认使用BGR格式，所以要确保输入正确的颜色格式)
    # 如果输入是RGB格式，先转换为BGR以便OpenCV处理
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        img1_for_draw = cv2.cvtColor(img1_display, cv2.COLOR_RGB2BGR) 
    else:
        img1_for_draw = img1_display
        
    if len(img2.shape) == 3 and img2.shape[2] == 3:
        img2_for_draw = cv2.cvtColor(img2_display, cv2.COLOR_RGB2BGR)
    else:
        img2_for_draw = img2_display
        
    match_img = cv2.drawMatches(img1_for_draw, kp1, img2_for_draw, kp2, good_matches, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # 转换回RGB以便后续matplotlib显示
    match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
    
    return len(good_matches), match_img

def visualize_predictions(dataset, threshold, min_matches, num_samples=10):
    """可视化SIFT匹配预测结果"""
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建索引列表并随机打乱
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    # 设置图表大小
    plt.figure(figsize=(15, num_samples * 5))
    
    sample_count = 0
    i = 0
    
    while sample_count < num_samples and i < len(indices):
        idx = indices[i]
        img1, img2, label = dataset[idx]
        
        # 将图像转换为numpy数组
        if isinstance(img1, Image.Image):
            img1 = np.array(img1)
        if isinstance(img2, Image.Image):
            img2 = np.array(img2)
            
        # 将PyTorch张量转换为numpy数组
        if hasattr(img1, 'numpy'):
            img1 = img1.numpy()
        if hasattr(img2, 'numpy'):
            img2 = img2.numpy()
        
        # 处理通道顺序 - 保持RGB格式
        if isinstance(img1, np.ndarray) and img1.shape[0] == 3 and len(img1.shape) == 3:
            img1 = np.transpose(img1, (1, 2, 0))
        if isinstance(img2, np.ndarray) and img2.shape[0] == 3 and len(img2.shape) == 3:
            img2 = np.transpose(img2, (1, 2, 0))
        
        # 确保图像是uint8类型
        if img1.dtype != np.uint8:
            img1 = (img1 * 255).astype(np.uint8)
        if img2.dtype != np.uint8:
            img2 = (img2 * 255).astype(np.uint8)
        
        # 执行SIFT匹配
        num_matches, match_img = sift_match(img1, img2, threshold)
        
        # 基于匹配点数量进行预测
        prediction = 1 if num_matches >= min_matches else 0
        
        # 转换回RGB用于matplotlib显示
        if len(img1.shape) == 3:
            img1_display = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        else:
            img1_display = img1
            
        if len(img2.shape) == 3:
            img2_display = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        else:
            img2_display = img2
            
        match_display = match_img
        
        # 显示图像和预测
        plt.subplot(num_samples, 2, sample_count * 2 + 1)
        plt.imshow(match_display)
        plt.title(f"num_matches: {num_matches}")
        plt.axis('off')
        
        # 结果面板
        plt.subplot(num_samples, 2, sample_count * 2 + 2)
        plt.text(0.5, 0.5, 
                 f"num_matches: {num_matches}",
                 ha='center', va='center', fontsize=18)
        plt.axis('off')
        
        sample_count += 1
        i += 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'sift_matching_results.png'), dpi=300)
    print(f"可视化结果已保存至 {os.path.join(args.output_dir, 'sift_matching_results.png')}")

def main():
    # 创建原始数据集以获取图像和标签
    dataset = MatchingDataset(args.data_dir, transform_p=None, transform_c=None)
    print(f"数据集大小: {len(dataset)} 个样本")
    
    # 评估SIFT匹配性能
    correct = 0
    total = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    print("正在使用SIFT进行图像匹配...")
    for i in tqdm(range(len(dataset))):
        img1, img2, label = dataset[i]
        
        # 将图像转换为numpy数组
        if isinstance(img1, Image.Image):
            img1 = np.array(img1)
        if isinstance(img2, Image.Image):
            img2 = np.array(img2)
            
        # 将PyTorch张量转换为numpy数组
        if hasattr(img1, 'numpy'):
            img1 = img1.numpy()
        if hasattr(img2, 'numpy'):
            img2 = img2.numpy()
        
        # 处理通道顺序
        if isinstance(img1, np.ndarray) and img1.shape[0] == 3 and len(img1.shape) == 3:
            img1 = np.transpose(img1, (1, 2, 0))
        if isinstance(img2, np.ndarray) and img2.shape[0] == 3 and len(img2.shape) == 3:
            img2 = np.transpose(img2, (1, 2, 0))
        
        # 确保图像是uint8类型
        if img1.dtype != np.uint8:
            img1 = (img1 * 255).astype(np.uint8)
        if img2.dtype != np.uint8:
            img2 = (img2 * 255).astype(np.uint8)
        
        # 使用SIFT匹配
        num_matches, _ = sift_match(img1, img2, args.threshold)
        prediction = 1 if num_matches >= args.min_matches else 0
        
        # 更新统计信息
        if prediction == label:
            correct += 1
        
        if prediction == 1 and label == 1:
            true_positives += 1
        elif prediction == 0 and label == 0:
            true_negatives += 1
        elif prediction == 1 and label == 0:
            false_positives += 1
        elif prediction == 0 and label == 1:
            false_negatives += 1
            
        total += 1
    
    # 计算整体指标
    accuracy = 100 * correct / total
    
    total_positive = true_positives + false_negatives
    total_negative = true_negatives + false_positives
    
    print(f"\nSIFT匹配结果:")
    print(f"匹配阈值: {args.threshold}, 最小匹配点数: {args.min_matches}")
    print(f"模型整体准确率: {accuracy:.2f}%")
    print(f"正样本: {true_positives}/{total_positive} 正确 ({100*true_positives/total_positive if total_positive > 0 else 0:.2f}%)")
    print(f"负样本: {true_negatives}/{total_negative} 正确 ({100*true_negatives/total_negative if total_negative > 0 else 0:.2f}%)")
    print(f"假阳性: {false_positives}, 假阴性: {false_negatives}")
    
    # 可视化随机样本的预测结果
    print(f"\n可视化 {args.num_samples} 个随机样本的预测结果...")
    visualize_predictions(dataset, args.threshold, args.min_matches, num_samples=args.num_samples)

if __name__ == '__main__':
    main()
