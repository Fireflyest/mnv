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

image = Image.open('./temp/1.png').convert('RGB')

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
    features = torch.flatten(model.avgpool(model.features[:2](image)), 1)
    # features = model(image)
features = features.squeeze().numpy() # (576,) 的特征向量

# features转为文本，不要省略
features_str = ' '.join([str(f) for f in features])
# print(f'features: {features_str}')
print(f'Features size: {len(features_str)} bytes')


# 计算压缩所需时间
start_time: float = time.time()
# 压缩文本
compressed_data = zlib.compress(features_str.encode('utf-8'))
# 将压缩后的数据转换为 Base64 编码
compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')
end_time = time.time()
compression_time = end_time - start_time
print(f'Compression time: {compression_time} seconds')
# print(f'compressed_base64: {compressed_base64}')
print(f'Compressed Base64 size: {len(compressed_base64)} bytes')

start_time = time.time()
# 解压缩文本
decompressed_data = zlib.decompress(base64.b64decode(compressed_base64)).decode('utf-8')
end_time = time.time()
decompression_time = end_time - start_time
print(f'Decompression time: {decompression_time:.6f} seconds')
print(f'Decompressed size: {len(decompressed_data)} bytes')
