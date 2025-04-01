import base64
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import zlib
import time
import numpy as np

# 最后转向量
# Image to Base64 size: 60888 bytes
# Features size: 6760 bytes
# Compression time: 0.0 seconds
# Compressed Base64 size: 3892 bytes
# Decompression time: 0.000000 seconds
# Decompressed size: 6760 bytes

# 浅层转向量
# Image to Base64 size: 60888 bytes
# Features size: 183 bytes
# Compression time: 0.0009970664978027344 seconds
# Compressed Base64 size: 148 bytes
# Decompression time: 0.000000 seconds
# Decompressed size: 183 bytes

# 所有层特征拼接
# Image to Base64 size: 60888 bytes
# Features size: 8782 bytes
# Compression time: 0.0 seconds
# Compressed Base64 size: 5100 bytes
# Decompression time: 0.000000 seconds
# Decompressed size: 8782 bytes

# 加载预训练的 MobileNetV3 模型
weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = models.mobilenet_v3_small(weights=weights)
# model.avgpool = nn.Identity()  # 移除平均池化层
model.classifier = nn.Identity()  # 移除分类层，只保留特征提取层
model.eval()

# 定义图像预处理
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open('./temp/1.jpg').convert('RGB')

# 将图像保存到字节流
buffered = io.BytesIO()
image.save(buffered, format="JPEG")

# 获取图像的字节数据
img_bytes = buffered.getvalue()

# 将字节数据转换为 Base64 文本
img_base64 = base64.b64encode(img_bytes).decode('utf-8')

# 计算 Base64 文本的大小
base64_size = len(img_base64)
# print(f'img_base64: {img_base64}')
print(f"Image to Base64 size: {base64_size} bytes")



image = preprocess(image).unsqueeze(0)  # 添加批次维度
with torch.no_grad():
    x1 = model.features[:2](image)       # [B, 16, H/2, W/2]
    x2 = model.features[2:4](x1)         # [B, 24, H/4, W/4]
    x3 = model.features[4:7](x2)         # [B, 40, H/8, W/8]
    x4 = model.features[7:10](x3)        # [B, 80, H/16, W/16]
    x5 = model.features[10:](x4)         # [B, 576, H/32, W/32]
    
    # Apply avgpool to each feature map
    pool = nn.AdaptiveAvgPool2d((1, 1))
    f1 = torch.flatten(pool(x1), 1)      # [B, 16]
    f2 = torch.flatten(pool(x2), 1)      # [B, 24]
    f3 = torch.flatten(pool(x3), 1)      # [B, 40]
    f4 = torch.flatten(pool(x4), 1)      # [B, 80]
    f5 = torch.flatten(pool(x5), 1)      # [B, 576]
    
    # Concatenate all features
    features = torch.cat([f1, f2, f3, f4, f5], dim=1)  # [B, 16+24+40+80+576=736]
    # features = x5

    # features = torch.flatten(model.avgpool(model.features[:2](image)), 1)
    # features = model(image)
features = features.squeeze().numpy() # (576,) 的特征向量

# Convert features to bytes directly (binary representation)
features_bytes = features.tobytes()
print(f'Features size: {len(features_bytes)} bytes')

# Compress the binary data
start_time: float = time.time()
compressed_data = zlib.compress(features_bytes)
compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')
end_time = time.time()
compression_time = end_time - start_time
print(f'Compression time: {compression_time} seconds')
print(f'Compressed Base64 size: {len(compressed_base64)} bytes')

# Decompress
start_time = time.time()
decompressed_bytes = zlib.decompress(base64.b64decode(compressed_base64))
decompressed_features = np.frombuffer(decompressed_bytes, dtype=features.dtype)
end_time = time.time()
decompression_time = end_time - start_time
print(f'Decompression time: {decompression_time:.6f} seconds')
print(f'Decompressed size: {len(decompressed_bytes)} bytes')

# Verify decompression worked correctly
print(f'Decompression successful: {np.array_equal(features, decompressed_features)}')


print("\n=== Grayscale Image Direct Compression Test ===")

# Convert original image to grayscale and resize to 224x224
gray_image = Image.open('./temp/1.jpg').convert('L').resize((224, 224), Image.LANCZOS)

# Save grayscale image to bytes
gray_buffered = io.BytesIO()
gray_image.save(gray_buffered, format="JPEG", quality=85)
gray_bytes = gray_buffered.getvalue()

print(f"Grayscale Image (224x224) bytes size: {len(gray_bytes)} bytes")

# First compress the bytes directly
start_time = time.time()
gray_compressed = zlib.compress(gray_bytes)
compression_time = time.time() - start_time
print(f'Grayscale Compression time: {compression_time:.6f} seconds')
print(f'Compressed bytes size: {len(gray_compressed)} bytes')

# Then convert the compressed data to Base64
gray_compressed_base64 = base64.b64encode(gray_compressed).decode('utf-8')
print(f'Compressed data as Base64: {len(gray_compressed_base64)} bytes')

# For comparison, show original image as Base64 without compression
gray_base64 = base64.b64encode(gray_bytes).decode('utf-8')
print(f"Uncompressed grayscale as Base64: {len(gray_base64)} bytes")
print(f"Size reduction: {(1 - len(gray_compressed_base64)/len(gray_base64))*100:.2f}%")

# Decompress grayscale image
start_time = time.time()
gray_decompressed_bytes = zlib.decompress(base64.b64decode(gray_compressed_base64))
decompression_time = time.time() - start_time
print(f'Grayscale Decompression time: {decompression_time:.6f} seconds')
print(f'Grayscale Decompressed size: {len(gray_decompressed_bytes)} bytes')

# Verify decompression worked correctly
print(f'Grayscale decompression successful: {gray_bytes == gray_decompressed_bytes}')

# Optional: Save the resized grayscale image for visual verification
gray_image.save('./temp/grayscale_224x224.jpg')

# Reconstruct and verify the image
reconstructed_gray_image = Image.open(io.BytesIO(gray_decompressed_bytes))
# reconstructed_gray_image.save('./temp/reconstructed_gray_224x224.jpg')
