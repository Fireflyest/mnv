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
        features = model(image)
    return features.squeeze().numpy()

def match_images(image1_path, image2_path):
    features1 = extract_features(image1_path)
    features2 = extract_features(image2_path)
    # 计算余弦相似度
    similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    return similarity

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

image1_path = './temp/image.jpg'
image_paths = [
    './temp/1.jpg',
    './temp/2.jpg',
    './temp/3.jpg',
    './temp/4.jpg',
    './temp/ice.jpg',
    './temp/waterpolo.jpg'
]
output_path = './out/match_v3.jpg'
save_match_result(image1_path, image_paths, output_path)