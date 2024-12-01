import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import random
import os
from PIL import Image

import mobilenetv4

class HuaLiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = ['waterpolo', 'lantern', 'ice', 'woodelf']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# 设置数据加载器
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整MNIST图像大小以匹配MobileNetV4的输入
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Assuming HuaLiDataset and model are already defined and loaded
dataset = HuaLiDataset(root_dir='./data/huali', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Randomly select five images
random_indices = random.sample(range(len(dataset)), 5)
selected_images = [dataset[i] for i in random_indices]

# Prepare the images for prediction
images = torch.stack([img for img, _ in selected_images])
images = images.to(device)

# Load the model
model = mobilenetv4.MobileNetV4("MobileNetV4ConvSmall").to(device)
model.load_state_dict(torch.load('./out/mobilenetv4.pth', weights_only=True))
model.eval()

# Predict the classes of the selected images
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# Print the predicted classes
for i, idx in enumerate(random_indices):
    print(f"Image {idx} predicted class: {predicted[i].item()}")

# 用plt显示五张图片的预测结果
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, (img, label) in enumerate(selected_images):
    # Convert the image tensor to a NumPy array
    img = img.permute(1, 2, 0)
    img = img * 0.3081 + 0.1307
    axes[i].imshow(img)
    axes[i].set_title(f"Predicted: {predicted[i].item()}\nActual: {dataset.classes[label]}")
    axes[i].axis("off")

plt.savefig('./out/val.png')
