import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

# 加载预训练的 MobileNetV3 模型
weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = models.mobilenet_v3_small(weights=weights)
model.classifier = nn.Identity()  # 移除分类层，只保留特征提取层
model.eval()

# 定义图像预处理
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)  # 添加批次维度
    with torch.no_grad():
        x1 = model.features[:2](image)       # [B, 16, H/2, W/2]
        x2 = model.features[2:4](x1)         # [B, 24, H/4, W/4]
        x3 = model.features[4:7](x2)         # [B, 40, H/8, W/8]
        x4 = model.features[7:10](x3)        # [B, 80, H/16, W/16]
        x5 = model.features[10:](x4)         # [B, 576, H/32, W/32]
        
        spatial_attention = torch.mean(x5, dim=1, keepdim=True)  # [B, 1, H/32, W/32]
        spatial_attention = torch.sigmoid(spatial_attention)
        x5 = x5 * (1 - spatial_attention)  # 逐元素相乘，广播机制会自动扩展
        
        # Apply avgpool to each feature map
        pool = nn.AdaptiveAvgPool2d((1, 1))
        f1 = torch.flatten(pool(x1), 1)      # [B, 16]
        f2 = torch.flatten(pool(x2), 1)      # [B, 24]
        f3 = torch.flatten(pool(x3), 1)      # [B, 40]
        f4 = torch.flatten(pool(x4), 1)      # [B, 80]
        f5 = torch.flatten(pool(x5), 1)  # [B, 576] - 使用注意力加权后的特征
        
        # Concatenate all features
        features = torch.cat([f1, f2, f3, f4, f5], dim=1)  # [B, 16+24+40+80+576=736]
        return features.squeeze().numpy() # (576,) 的特征向量

def match_images(image1_path, image2_path):
    features1 = extract_features(image1_path)
    features2 = extract_features(image2_path)
    similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    return similarity
    # similarity = np.linalg.norm(features1 - features2)
    # similarity = 1 / (1 + similarity)
    # return similarity


def save_match_result(image1_path, image_paths, output_path):
    # 读取原图
    image1 = cv2.imread(image1_path)

    # 初始化一个空的列表来存储合并后的图像
    combined_images = []

    for image_path in image_paths:
        # 计算相似度
        similarity = match_images(image1_path, image_path)
        print(f'Similarity with {image_path}: {similarity}')

        # 读取对比图像
        image2 = cv2.imread(image_path)

        # 调整图像尺寸，使它们相同
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

        # 将两张图片并排显示
        combined_image = np.hstack((image1, image2))

        # 在图片上显示相似度
        cv2.putText(combined_image, f'Similarity: {similarity:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 将合并后的图像添加到列表中
        combined_images.append(combined_image)

    # 将所有合并后的图像垂直堆叠
    final_image = np.vstack(combined_images)

    # 保存图片
    cv2.imwrite(output_path, final_image)

image1_path = './temp/1.jpg'
image_paths = [
    # './temp/1.jpg',
    './temp/2.jpg',
    './temp/3.jpg',
    './temp/4.jpg',
    './temp/5.jpg',
    # './temp/ice.jpg',
    # './temp/waterpolo.jpg'
]
output_path = './out/match_mv3.jpg'
save_match_result(image1_path, image_paths, output_path)