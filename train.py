import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

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
    transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(400),
    transforms.Resize(224),
    # transforms.ColorJitter(brightness=.2, hue=.2),
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    transforms.RandomAffine(degrees=0, shear=(-30, 30)),  # 添加水平翻转角度
    # data.HorizontalRandomPerspective(distortion_scale=0.6, p=0.6),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataset = data.HuaLiDataset(root_dir='./data/huali/train5', transform=transform)

# 划分训练集和验证集
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 设置模型，损失函数和优化器
# Support ['MobileNetV4ConvSmall', 'MobileNetV4ConvMedium', 'MobileNetV4ConvLarge']
# Also supported ['MobileNetV4HybridMedium', 'MobileNetV4HybridLarge']
model = mobilenetv4.MobileNet(class_nums=len(dataset.classes)).to(device)
criterion_class = nn.CrossEntropyLoss()
criterion_moire = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# 训练模型
def train_model(model, criterion_class, criterion_moire, optimizer, train_loader, test_loader, epochs=50):
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_class = 0
        correct_moire = 0
        total = 0
        for batch_idx, (data, target_class, target_moire) in enumerate(train_loader):
            data = data.to(device)
            target_class = target_class.to(device)
            target_moire = target_moire.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            output_class, output_moire = model(data)
            loss_class = criterion_class(output_class, target_class)
            loss_moire = criterion_moire(output_moire, target_moire)
            loss = loss_class + loss_moire
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted_class = torch.max(output_class.data, 1)
            predicted_moire = (output_moire > 0.5).float()
            total += target_class.size(0)
            correct_class += (predicted_class == target_class).sum().item()
            correct_moire += (predicted_moire == target_moire).sum().item()

        train_losses.append(train_loss / len(train_loader))
        train_accs.append(correct_class / total)
        
        model.eval()
        test_loss = 0
        correct_class = 0
        correct_moire = 0
        total = 0
        with torch.no_grad():
            for data, target_class, target_moire in test_loader:
                data = data.to(device)
                target_class = target_class.to(device)
                target_moire = target_moire.to(device).float().unsqueeze(1)
                output_class, output_moire = model(data)
                loss_class = criterion_class(output_class, target_class)
                loss_moire = criterion_moire(output_moire, target_moire)
                loss = loss_class + loss_moire
                test_loss += loss.item()
                _, predicted_class = torch.max(output_class.data, 1)
                predicted_moire = (output_moire > 0.5).float()
                total += target_class.size(0)
                correct_class += (predicted_class == target_class).sum().item()
                correct_moire += (predicted_moire == target_moire).sum().item()
        
        test_losses.append(test_loss / len(test_loader))
        test_accs.append(correct_class / total)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Train Class Acc: {correct_class / total:.4f}, Train Moire Acc: {correct_moire / total:.4f}, "
              f"Test Loss: {test_loss / len(test_loader):.4f}, Test Class Acc: {correct_class / total:.4f}, "
              f"Test Moire Acc: {correct_moire / total:.4f}")
    
        # Save the best model
        if epoch > epochs * 0.4 and test_accs[-1] > best_acc:
            best_acc = test_accs[-1]
            torch.save(model.state_dict(), './out/mobilenetv4.pth')
            print(f'Save the best model with accuracy: {best_acc:.4f}')
    
    # save the last model
    torch.save(model.state_dict(), './out/mobilenetv4_last.pth')

    return train_losses, test_losses, train_accs, test_accs

# 运行训练
train_losses, test_losses, train_accs, test_accs = train_model(model, criterion_class, criterion_moire, optimizer, train_loader, test_loader)

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

plt.savefig('./out/acc.png')