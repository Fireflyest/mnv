import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

import data
import mobilenetv3_u

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# 设置数据加载器
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    # transforms.RandomCrop(400),
    transforms.Resize(224),
    transforms.ColorJitter(brightness=.2, hue=.2),
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    # transforms.RandomAffine(degrees=0, shear=(-30, 30)),  # 添加水平翻转角度
    transforms.RandomAffine(degrees=(-15, 15)),  # 添加不同倾斜角度
    # data.HorizontalRandomPerspective(distortion_scale=0.6, p=0.6),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

dataset = data.HuaLiDataset_Multi(root_dir='./data/huali/train7', transform=transform)

# 划分训练集和验证集
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载预训练的 MobileNetV3 模型
model = mobilenetv3_u.MultiHeadMobileNetV3(num_classes=5)
print(model)

# 冻结前面的层
for param in model.features.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = False

# 只训练最后的多头输出层
for param in model.head1.parameters():
    param.requires_grad = True
for param in model.head2.parameters():
    param.requires_grad = True

# 将模型移动到设备
model.to(device)

# 定义损失函数和优化器
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.BCELoss()
optimizer = optim.Adam(list(model.head1.parameters()) + list(model.head2.parameters()), lr=0.001)

# 训练模型
def train_model(model, criterion1, criterion2, optimizer, train_loader, test_loader, epochs=50):
    train_losses, test_losses = [], []
    train_accs1, train_accs2, test_accs1, test_accs2 = [], [], [], []
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct1 = 0
        correct2 = 0
        total = 0
        for batch_idx, (data, target1, target2) in enumerate(train_loader):
            data = data.to(device)
            target1 = target1.to(device)
            target2 = target2.to(device).float()
            optimizer.zero_grad()
            output1, output2, _ = model(data)
            loss1 = criterion1(output1, target1)
            loss2 = criterion2(output2.squeeze(), target2)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted1 = torch.max(output1.data, 1)
            predicted2 = (output2 > 0.5).float()
            total += target1.size(0)
            correct1 += (predicted1 == target1).sum().item()
            correct2 += (predicted2.squeeze() == target2).sum().item()
        
        train_losses.append(train_loss / len(train_loader))
        train_accs1.append(correct1 / total)
        train_accs2.append(correct2 / total)
        
        model.eval()
        test_loss = 0
        correct1 = 0
        correct2 = 0
        total = 0
        with torch.no_grad():
            for data, target1, target2 in test_loader:
                data = data.to(device)
                target1 = target1.to(device)
                target2 = target2.to(device).float()
                output1, output2, _ = model(data)
                loss1 = criterion1(output1, target1)
                loss2 = criterion2(output2.squeeze(), target2)
                loss = loss1 + loss2
                test_loss += loss.item()
                _, predicted1 = torch.max(output1.data, 1)
                predicted2 = (output2 > 0.5).float()
                total += target1.size(0)
                correct1 += (predicted1 == target1).sum().item()
                correct2 += (predicted2.squeeze() == target2).sum().item()
        
        test_losses.append(test_loss / len(test_loader))
        test_accs1.append(correct1 / total)
        test_accs2.append(correct2 / total)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Train Acc Class: {train_accs1[-1]:.4f}, Train Acc Logic: {train_accs2[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Test Acc Class: {test_accs1[-1]:.4f}, Test Acc Logic: {test_accs2[-1]:.4f}")

        # 保存最佳模型
        if epoch > epochs * 0.4 and test_accs1[-1] + test_accs2[-1] > best_acc:
            best_acc = test_accs1[-1] + test_accs2[-1]
            torch.save(model.state_dict(), './out/mobilenetv3_u_best_finetuned.pth')
            print(f"Model saved with accuracy: {best_acc:.4f}")

    return train_losses, test_losses, train_accs1, train_accs2, test_accs1, test_accs2

# 训练模型
train_losses, test_losses, train_accs1, train_accs2, test_accs1, test_accs2 = train_model(model, criterion1, criterion2, optimizer, train_loader, test_loader, epochs=3)

# 保存最后的模型
torch.save(model.state_dict(), './out/mobilenetv3_u_last_finetuned.pth')

# 绘制训练和验证损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

# 绘制训练和验证准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accs1, label='Train Accuracy Head 1')
plt.plot(train_accs2, label='Train Accuracy Head 2')
plt.plot(test_accs1, label='Test Accuracy Head 1')
plt.plot(test_accs2, label='Test Accuracy Head 2')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')

plt.savefig('./out/mobilenetv3_u_training_curves.png')