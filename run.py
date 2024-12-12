import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import random

import mobilenetv4
import data

device = (
    "cuda:3"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# 设置数据加载器
transform = transforms.Compose([
    transforms.Resize(224),
    # transforms.ColorJitter(brightness=.2, hue=.08),
    # transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    # transforms.RandomAffine(degrees=0, shear=(-30, 30)),  # 添加水平翻转角度
    # data.HorizontalRandomPerspective(distortion_scale=0.6, p=0.6),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Assuming HuaLiDataset and model are already defined and loaded
dataset = data.HuaLiDataset(root_dir='./data/huali/test', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Randomly select five images
random_indices = random.sample(range(len(dataset)), 12)
selected_images = [dataset[i] for i in random_indices]

# Prepare the images for prediction
images = torch.stack([img for img, _ in selected_images])
images = images.to(device)

# Load the model
model = mobilenetv4.MobileNet(len(dataset.classes)).to(device)
model.load_state_dict(torch.load('./out/mobilenetv4.pth', weights_only=True))
model.eval()

# Predict the classes of the selected images
with torch.no_grad():
    outputs = model(images)
    probabilities, predicted = torch.max(outputs, 1)

# Print the predicted classes and probabilities
for i, idx in enumerate(random_indices):
    print(f"Image {idx} predicted class: {predicted[i].item()} with probability: {probabilities[i].item()}")

# 用plt显示五张图片的预测结果
fig, axes = plt.subplots(3, 4, figsize=(20, 8))
axes = axes.flatten()
for i, (img, label) in enumerate(selected_images):
    # Convert the image tensor to a NumPy array
    img = img.permute(1, 2, 0)
    # Unnormalize
    img = img * torch.tensor((0.229, 0.224, 0.225)) + torch.tensor((0.485, 0.456, 0.406))
    axes[i].imshow(img)
    axes[i].set_title(f"Predicted: {predicted[i].item()} ({probabilities[i].item():.2f})\nActual: {dataset.classes[label]}")
    axes[i].axis("off")

plt.savefig('./out/val.png')
