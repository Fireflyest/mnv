import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

import timm

import data

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# 设置数据加载器
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(400),
    transforms.Resize(224),
    # transforms.ColorJitter(brightness=.2, hue=.2),
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    transforms.RandomAffine(degrees=0, shear=(-30, 30)),  # 添加水平翻转角度
    transforms.RandomAffine(degrees=(-15, 15)),  # 添加不同倾斜角度
    # data.HorizontalRandomPerspective(distortion_scale=0.6, p=0.6),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

dataset = data.HuaLiDataset(root_dir='./data/huali/train7', transform=transform)

# 划分训练集和验证集
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载预训练的 MobileNetV3 模型
weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = models.mobilenet_v3_small(weights=weights)

# 修改最后一层
num_classes = 5
model.classifier[3] = nn.Sequential(
    nn.Linear(model.classifier[3].in_features, num_classes),
    nn.Softmax(dim=1)
)
# 冻结部分层
for param in model.parameters():
    param.requires_grad = False

# 只训练最后一层
for param in model.classifier[3].parameters():
    param.requires_grad = True

# 将模型移动到设备
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier[3].parameters(), lr=0.001)

# 训练模型
def train_model(model, criterion, optimizer, train_loader, test_loader, epochs=50):
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(correct / total)
        
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        test_losses.append(test_loss / len(test_loader))
        test_accs.append(correct / total)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accs[-1]:.4f}')
    
        # Save the best model
        if epoch > epochs * 0.4 and test_accs[-1] > best_acc:
            best_acc = test_accs[-1]
            torch.save(model.state_dict(), './out/mobilenetv3_best_finetuned.pth')
            print(f'Save the best model with accuracy: {best_acc:.4f}')
    
    # save the last model
    torch.save(model.state_dict(), './out/mobilenetv3_last_finetuned.pth')

    return train_losses, test_losses, train_accs, test_accs

# 运行训练
train_losses, test_losses, train_accs, test_accs = train_model(model, criterion, optimizer, train_loader, test_loader)

# 绘制准确率曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig('./out/acc_loss_finetuned.png')