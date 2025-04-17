import mobilenetv4
import mobilenetv3
import data

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

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
modelv3 = mobilenetv3.MultiHeadMobileNetV3(num_classes=5)
state_dict = torch.load('./out/mobilenetv3_best_finetuned.pth', weights_only=True)
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('mobilenet.classifier'):
        new_key = k.replace('mobilenet.', '')
        new_state_dict[new_key] = v
    else:
        new_state_dict[k] = v

modelv3.load_state_dict(new_state_dict)
modelv3.eval()
modelv3.to(device)


modelv4 = mobilenetv4.MobileNetV4("MobileNetV4ConvSmall", num_classes=5)
modelv4.to(device)


class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temp=4.0, feature_weights=[0.1, 0.2, 0.3, 0.2, 0.2]):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temp = temp
        self.feature_weights = feature_weights
        self.cls_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.BCELoss()
        self.feat_criterion = nn.MSELoss()
        self.kl_criterion = nn.KLDivLoss(reduction='batchmean')
        
        # 添加通道适配器 - 只处理通道数不匹配的情况
        self.adapters = nn.ModuleList([
            nn.Conv2d(32, 16, kernel_size=1),    # v4.conv0 → v3.f1: 32→16
            nn.Conv2d(32, 24, kernel_size=1),    # v4.layer1 → v3.f2: 32→24
            nn.Conv2d(64, 40, kernel_size=1),    # v4.layer2 → v3.f3: 64→40
            nn.Conv2d(96, 112, kernel_size=1),   # v4.layer3 → v3.f4: 96→112
            nn.Identity()                        # v4.layer5 → v3.f5: 960→960 (保持一致)
        ])
        
    def forward(self, student_outputs, teacher_outputs, targets):
        # 解包输出 - 注意这里调整v4的特征提取方式
        s_cls, s_reg, s_f1, s_f2, s_f3, s_f4, s_f5 = student_outputs
        t_cls, t_reg, t_f1, t_f2, t_f3, t_f4, t_f5 = teacher_outputs
        cls_target, reg_target = targets
        
        # 硬目标损失
        cls_loss = self.cls_criterion(s_cls, cls_target)
        reg_loss = self.reg_criterion(s_reg, reg_target)
        hard_loss = cls_loss + reg_loss
        
        # 软目标损失 - KL散度
        s_cls_temp = torch.log_softmax(s_cls / self.temp, dim=1)
        t_cls_temp = torch.softmax(t_cls / self.temp, dim=1)
        soft_cls_loss = self.kl_criterion(s_cls_temp, t_cls_temp) * (self.temp ** 2)
        
        # 回归输出蒸馏
        soft_reg_loss = self.feat_criterion(s_reg, t_reg)
        
        # 特征蒸馏损失 - 使用正确的对应关系和通道适配
        adapted_features = [
            self.adapters[0](s_f1),  # v4.layer1 → v3.f1
            self.adapters[1](s_f2),  # v4.layer2 → v3.f2
            self.adapters[2](s_f3),  # v4.layer3 → v3.f3
            self.adapters[3](s_f4),  # v4.layer4 → v3.f4
            self.adapters[4](s_f5)   # v4.layer5 → v3.f5
        ]
        
        feature_losses = [
            self.feat_criterion(adapted_features[0], t_f1),
            self.feat_criterion(adapted_features[1], t_f2),
            self.feat_criterion(adapted_features[2], t_f3),
            self.feat_criterion(adapted_features[3], t_f4),
            self.feat_criterion(adapted_features[4], t_f5)
        ]
        
        # 计算特征损失的加权和
        feature_loss = sum(w * loss for w, loss in zip(self.feature_weights, feature_losses))
        
        # 蒸馏总损失
        soft_loss = soft_cls_loss + soft_reg_loss + feature_loss
        
        # 总损失 = 硬目标损失 * (1-alpha) + 软目标损失 * alpha
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        
        return total_loss, cls_loss, reg_loss, soft_cls_loss, soft_reg_loss, feature_loss


def visualize_feature_maps(teacher_model, student_model):
    """可视化教师和学生模型的特征图，确保它们语义上可比"""
    
    # 创建测试输入
    test_input = torch.randn(1, 3, 224, 224).to(device)
    
    # 获取特征图
    with torch.no_grad():
        teacher_model.eval()
        student_model.eval()
        
        _, _, t_f1, t_f2, t_f3, t_f4, t_f5 = teacher_model(test_input)
        _, _, s_f0, s_f1, s_f2, s_f3, s_f5 = student_model(test_input)
    
    # 打印特征图维度
    print("MobileNetV3 (教师) 特征图维度:")
    print(f"f1: {t_f1.shape}, f2: {t_f2.shape}, f3: {t_f3.shape}, f4: {t_f4.shape}, f5: {t_f5.shape}")
    
    print("\nMobileNetV4 (学生) 特征图维度:")
    print(f"f0: {s_f0.shape}, f1: {s_f1.shape}, f2: {s_f2.shape}, f3: {s_f3.shape}, f5: {s_f5.shape}")
    
    # 可视化每个通道的平均激活图
    import matplotlib.pyplot as plt
    
    # 对特征图取通道平均值，进行二维可视化
    def plot_feature_map(feature, title):
        avg_feature = torch.mean(feature, dim=1).squeeze().cpu().numpy()
        plt.figure(figsize=(5, 5))
        plt.imshow(avg_feature, cmap='viridis')
        plt.title(title)
        plt.colorbar()
        plt.axis('off')
        return plt.gcf()
    
    # 可视化和比较特征图对
    pairs = [
        (t_f1, s_f0, "v3.f1 vs v4.conv0"),
        (t_f2, s_f1, "v3.f2 vs v4.layer1"),
        (t_f3, s_f2, "v3.f3 vs v4.layer2"),
        (t_f4, s_f3, "v3.f4 vs v4.layer3"),
        (t_f5, s_f5, "v3.f5 vs v4.layer5")
    ]
    
    figs = []
    for t_feat, s_feat, title in pairs:
        fig = plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        avg_t_feat = torch.mean(t_feat, dim=1).squeeze().cpu().numpy()
        plt.imshow(avg_t_feat, cmap='viridis')
        plt.title(f"教师 {title.split(' vs ')[0]}")
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        avg_s_feat = torch.mean(s_feat, dim=1).squeeze().cpu().numpy()
        plt.imshow(avg_s_feat, cmap='viridis')
        plt.title(f"学生 {title.split(' vs ')[1]}")
        plt.colorbar()
        plt.axis('off')
        
        plt.suptitle(title)
        figs.append(fig)
    
    return figs





# 设置优化器和学习率调度器
optimizer = optim.Adam(modelv4.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
distill_criterion = DistillationLoss(alpha=0.7, temp=4.0)

# 训练循环
def train(epoch):
    modelv4.train()
    modelv3.eval()  # 教师模型始终为评估模式
    
    running_loss = 0.0
    cls_losses = 0.0
    reg_losses = 0.0
    soft_cls_losses = 0.0
    soft_reg_losses = 0.0
    feature_losses = 0.0
    
    for i, (images, cls_targets, reg_targets) in enumerate(train_loader):
        images = images.to(device)
        cls_targets = cls_targets.to(device)
        reg_targets = reg_targets.to(device)
        
        # 获取教师模型输出
        with torch.no_grad():
            teacher_outputs = modelv3(images)
        
        # 获取学生模型输出
        student_outputs = modelv4(images)
        
        # 计算损失
        loss, cls_loss, reg_loss, soft_cls_loss, soft_reg_loss, feature_loss = distill_criterion(
            student_outputs, teacher_outputs, (cls_targets, reg_targets)
        )
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 累加损失
        running_loss += loss.item()
        cls_losses += cls_loss.item()
        reg_losses += reg_loss.item()
        soft_cls_losses += soft_cls_loss.item()
        soft_reg_losses += soft_reg_loss.item()
        feature_losses += feature_loss.item()
        
        if (i+1) % 10 == 0:
            print(f'Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, Cls: {cls_loss.item():.4f}, '
                  f'Reg: {reg_loss.item():.4f}, Feat: {feature_loss.item():.4f}')
    
    # 返回平均损失
    return (running_loss / len(train_loader), 
            cls_losses / len(train_loader),
            reg_losses / len(train_loader),
            soft_cls_losses / len(train_loader),
            soft_reg_losses / len(train_loader),
            feature_losses / len(train_loader))

# 验证函数
def validate():
    modelv4.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, cls_targets, reg_targets in test_loader:
            images = images.to(device)
            cls_targets = cls_targets.to(device)
            reg_targets = reg_targets.to(device)
            
            # 仅评估学生模型
            s_cls, s_reg, _, _, _, _, _ = modelv4(images)
            
            # 计算验证损失
            cls_loss = nn.CrossEntropyLoss()(s_cls, cls_targets)
            reg_loss = nn.BCELoss()(s_reg, reg_targets)
            loss = cls_loss + reg_loss
            val_loss += loss.item()
            
            # 计算分类准确率
            _, predicted = torch.max(s_cls.data, 1)
            total += cls_targets.size(0)
            correct += (predicted == cls_targets).sum().item()
    
    return val_loss / len(test_loader), 100 * correct / total


# 在训练前调用可视化函数
feature_figs = visualize_feature_maps(modelv3, modelv4)
for i, fig in enumerate(feature_figs):
    fig.savefig(f'./out/feature_pair_{i}.png')


# 训练主循环
num_epochs = 100
best_val_acc = 0.0
train_losses = []
val_losses = []
val_accs = []

for epoch in range(num_epochs):
    # 训练一个epoch
    train_loss, cls_loss, reg_loss, soft_cls_loss, soft_reg_loss, feature_loss = train(epoch)
    train_losses.append(train_loss)
    
    # 验证
    val_loss, val_acc = validate()
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # 调整学习率
    scheduler.step(val_loss)
    
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    print(f'  Detail - Cls: {cls_loss:.4f}, Reg: {reg_loss:.4f}, '
          f'Soft-Cls: {soft_cls_loss:.4f}, Soft-Reg: {soft_reg_loss:.4f}, Features: {feature_loss:.4f}')
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(modelv4.state_dict(), './out/mobilenetv4_distilled.pth')
        print(f'Model saved with accuracy: {val_acc:.2f}%')

# 绘制训练过程
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(val_accs, label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Validation Accuracy')

plt.tight_layout()
plt.savefig('./out/distillation_training.png')
plt.show()