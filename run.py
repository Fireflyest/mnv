import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2

import mobilenetv4
import data
import vision

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
    # transforms.ColorJitter(brightness=.2, hue=.2),
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
images = torch.stack([img for img, _, _ in selected_images])
images = images.to(device)

# Load the model
model = mobilenetv4.MobileNet(len(dataset.classes)).to(device)
model.load_state_dict(torch.load('./out/mobilenetv4.pth', weights_only=True))
model.eval()

# Select the target layer for Grad-CAM
target_layer = model.layer4[-1]

# Initialize Grad-CAM
grad_cam = vision.GradCAM(model, target_layer)

# Predict the classes of the selected images and generate CAMs
with torch.no_grad():
    outputs = model(images)
    class_outputs, moire_outputs = outputs
    probabilities, predicted = torch.max(class_outputs, 1)
    moire_predictions = (moire_outputs > 0.5).float()

# Print the predicted classes and probabilities
for i, idx in enumerate(random_indices):
    print(f"Image {idx} predicted class: {predicted[i].item()} with probability: {probabilities[i].item()}, moire: {moire_predictions[i].item()}")

# 用plt显示十二张图片的预测结果和Grad-CAM
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()
for i, (img, label, moire_label) in enumerate(selected_images):
    # Convert the image tensor to a NumPy array
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)

    # Generate CAM
    cam = grad_cam.generate_cam(images[i].unsqueeze(0))

    # Show CAM on image
    cam_image = vision.show_cam_on_image(img_np, cam)

    axes[i].imshow(cam_image)
    axes[i].set_title(f"Predicted: {predicted[i].item()} ({probabilities[i].item():.2f})\nActual: {dataset.classes[label]}, Moire: {moire_predictions[i].item()}")
    axes[i].axis("off")

plt.savefig('./out/val_cam.png')
