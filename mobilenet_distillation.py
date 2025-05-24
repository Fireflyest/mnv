import mobilenetv4
import mobilenetv3
import data

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
import numpy as np

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
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
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


modelv4 = mobilenetv4.MobileNetV4("MobileNetV4ConvSmall")
modelv4.to(device)


class ImprovedDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temp=4.0, feature_weights=[0.1, 0.2, 0.3, 0.2, 0.2],
                 attention_weight=0.5, channel_weight=0.3, contrast_weight=0.2):
        super(ImprovedDistillationLoss, self).__init__()
        # 现有初始化代码
        self.alpha = alpha
        self.temp = temp
        self.feature_weights = feature_weights
        self.attention_weight = attention_weight
        self.channel_weight = channel_weight  # 新增
        self.contrast_weight = contrast_weight  # 新增
        self.cls_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.BCELoss()
        self.feat_criterion = nn.MSELoss()
        self.kl_criterion = nn.KLDivLoss(reduction='batchmean')
        
        # 添加通道适配器
        self.adapters = nn.ModuleList([
            nn.Conv2d(32, 16, kernel_size=1),
            nn.Conv2d(32, 24, kernel_size=1),
            nn.Conv2d(64, 40, kernel_size=1),
            nn.Conv2d(96, 112, kernel_size=1),
            nn.Conv2d(128, 960, kernel_size=1)
        ])
        
    def _generate_spatial_attention(self, feature):
        """生成空间注意力图"""
        return torch.mean(feature, dim=1, keepdim=True)
        
    def _generate_channel_attention(self, feature):
        """生成通道注意力图"""
        return torch.mean(feature, dim=[2, 3], keepdim=True)
    
    def _weighted_attention_loss(self, s_attention, t_attention):
        """加权注意力损失"""
        weight_map = torch.sigmoid(5 * t_attention)  # 使高激活区域权重更大
        return torch.mean((s_attention - t_attention).pow(2) * weight_map)
    
    def _contrast_attention_loss(self, s_attention, t_attention):
        """对比注意力损失，帮助区分前景和背景"""
        batch_size = t_attention.size(0)
        
        loss = 0
        for b in range(batch_size):
            # 使用教师注意力作为指导，区分前景和背景
            threshold = torch.mean(t_attention[b]) + 0.5 * torch.std(t_attention[b])
            foreground_mask = (t_attention[b] > threshold).float()
            background_mask = 1.0 - foreground_mask
            
            # 计算前景和背景的平均激活
            s_fg = torch.sum(s_attention[b] * foreground_mask) / (torch.sum(foreground_mask) + 1e-8)
            s_bg = torch.sum(s_attention[b] * background_mask) / (torch.sum(background_mask) + 1e-8)
            
            # 鼓励前景激活高于背景激活
            margin = 0.5
            loss += torch.relu(margin - (s_fg - s_bg))
        
        return loss / batch_size
    
    def _adapt_spatial_size(self, x, target_size):
        """调整特征图的空间尺寸以匹配目标尺寸"""
        if x.shape[2:] == target_size:
            return x
        
        return nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
    
    def forward(self, student_outputs, teacher_outputs, targets):
        # 解包输出
        s_cls, s_reg, s_f1, s_f2, s_f3, s_f4, s_f5 = student_outputs
        t_cls, t_reg, t_f1, t_f2, t_f3, t_f4, t_f5 = teacher_outputs
        cls_target, reg_target = targets
        
        reg_target = reg_target.float().view(-1, 1)
        
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
        
        # 特征适配和损失计算
        student_features = [s_f1, s_f2, s_f3, s_f4, s_f5]
        teacher_features = [t_f1, t_f2, t_f3, t_f4, t_f5]
        adapted_features = []
        feature_losses = []
        spatial_attention_losses = []
        channel_attention_losses = []
        contrast_losses = []
        
        for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
            # 1. 通道适配
            adapted_feat = self.adapters[i](s_feat)
            
            # 2. 空间尺寸适配
            adapted_feat = self._adapt_spatial_size(adapted_feat, t_feat.shape[2:])
            adapted_features.append(adapted_feat)
            
            # 3. 特征损失
            feature_losses.append(self.feat_criterion(adapted_feat, t_feat))
            
            # 4. 空间注意力损失
            s_spatial_att = self._generate_spatial_attention(adapted_feat)
            t_spatial_att = self._generate_spatial_attention(t_feat)
            
            # 归一化注意力图
            s_spatial_att = torch.sigmoid(s_spatial_att)
            t_spatial_att = torch.sigmoid(t_spatial_att)
            
            # 加权空间注意力损失
            spatial_att_loss = self._weighted_attention_loss(s_spatial_att, t_spatial_att)
            spatial_attention_losses.append(spatial_att_loss)
            
            # 5. 通道注意力损失
            s_channel_att = self._generate_channel_attention(adapted_feat)
            t_channel_att = self._generate_channel_attention(t_feat)
            
            # 归一化通道注意力
            s_channel_att = nn.functional.normalize(s_channel_att, p=1, dim=1)
            t_channel_att = nn.functional.normalize(t_channel_att, p=1, dim=1)
            
            channel_att_loss = self.feat_criterion(s_channel_att, t_channel_att)
            channel_attention_losses.append(channel_att_loss)
            
            # 6. 对比注意力损失
            contrast_loss = self._contrast_attention_loss(s_spatial_att, t_spatial_att)
            contrast_losses.append(contrast_loss)
        
        # 汇总各类损失
        feature_loss = sum(w * loss for w, loss in zip(self.feature_weights, feature_losses))
        spatial_att_loss = sum(w * loss for w, loss in zip(self.feature_weights, spatial_attention_losses))
        channel_att_loss = sum(w * loss for w, loss in zip(self.feature_weights, channel_attention_losses))
        contrast_loss = sum(w * loss for w, loss in zip(self.feature_weights, contrast_losses))
        
        # 总注意力损失
        attention_loss = (
            self.attention_weight * spatial_att_loss + 
            self.channel_weight * channel_att_loss + 
            self.contrast_weight * contrast_loss
        )
        
        # 最终损失计算
        soft_loss = soft_cls_loss + soft_reg_loss + feature_loss + attention_loss
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        
        return total_loss, cls_loss, reg_loss, soft_cls_loss, soft_reg_loss, feature_loss, attention_loss

def visualize_attention_maps(teacher_model, student_model, sample_images, epoch=0):
    """可视化并比较教师和学生模型的注意力图，每轮选5张保存"""
    teacher_model.eval()
    student_model.eval()
    
    # 创建可视化输出目录
    os.makedirs('./out/distillation_attention', exist_ok=True)
    
    # 只取前5张图片
    max_samples = 5
    sample_count = min(max_samples, len(sample_images))
    selected_images = sample_images[:sample_count]
    
    with torch.no_grad():
        for idx, image in enumerate(selected_images):
            # 添加批次维度并转移到设备
            image_tensor = image.unsqueeze(0).to(device)
            
            # 获取模型输出
            t_outputs = teacher_model(image_tensor)
            s_outputs = student_model(image_tensor)
            
            # 提取特征图和获取注意力图
            t_feats = [t_outputs[i+2] for i in range(5)]  # 跳过cls和reg输出
            s_feats = [s_outputs[i+2] for i in range(5)]  # 跳过cls和reg输出
            
            # 生成注意力图
            t_attentions = [torch.mean(feat, dim=1).squeeze().cpu().numpy() for feat in t_feats]
            s_attentions = [torch.mean(feat, dim=1).squeeze().cpu().numpy() for feat in s_feats]
            
            # 可视化对比图
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            fig.suptitle(f'Attention Maps Comparison - Sample {idx+1}')
            
            # 原始图像
            orig_img = image.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
            orig_img = (orig_img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])  # 反标准化
            orig_img = np.clip(orig_img, 0, 1)
            
            # 层级名称
            layer_names = ['conv0/f1', 'layer1/f2', 'layer2/f3', 'layer3/f4', 'layer5/f5']
            
            for i, (t_att, s_att) in enumerate(zip(t_attentions, s_attentions)):
                # 教师注意力图
                axes[0, i].imshow(orig_img)
                axes[0, i].imshow(t_att, cmap='jet', alpha=0.5)
                axes[0, i].set_title(f'Teacher: {layer_names[i]}')
                axes[0, i].axis('off')
                
                # 学生注意力图
                axes[1, i].imshow(orig_img)
                axes[1, i].imshow(s_att, cmap='jet', alpha=0.5)
                axes[1, i].set_title(f'Student: {layer_names[i]}')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.savefig(f'./out/distillation_attention/attention_comparison_{epoch}_{idx+1}.png')
            plt.close()


def visualize_feature_maps(teacher_model, student_model, epoch=0):
    """可视化教师和学生模型的特征图，确保它们语义上可比"""
    
    # 确保输出目录存在
    os.makedirs('./out/distillation_feature', exist_ok=True)
    
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
    def plot_feature_map(feature, title, filename):
        avg_feature = torch.mean(feature, dim=1).squeeze().cpu().numpy()
        plt.figure(figsize=(5, 5))
        plt.imshow(avg_feature, cmap='viridis')
        plt.title(title)
        plt.colorbar()
        plt.axis('off')
        plt.savefig(filename)
        plt.close()
    
    # 可视化和比较特征图对
    pairs = [
        (t_f1, s_f0, "v3.f1 vs v4.conv0"),
        (t_f2, s_f1, "v3.f2 vs v4.layer1"),
        (t_f3, s_f2, "v3.f3 vs v4.layer2"),
        (t_f4, s_f3, "v3.f4 vs v4.layer3"),
        (t_f5, s_f5, "v3.f5 vs v4.layer5")
    ]
    
    figs = []
    for i, (t_feat, s_feat, title) in enumerate(pairs):
        fig = plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        avg_t_feat = torch.mean(t_feat, dim=1).squeeze().cpu().numpy()
        plt.imshow(avg_t_feat, cmap='viridis')
        plt.title(f"教师 {title.split(' vs ')[0]}")
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        avg_s_feat = torch.mean(s_feat, dim=1).squeeze().cpu().numpy()
        # Check if the feature map has valid dimensions before visualizing
        if avg_s_feat.size > 0 and avg_s_feat.ndim >= 2:
            plt.imshow(avg_s_feat, cmap='viridis')
        else:
            plt.text(0.5, 0.5, "Empty feature map", ha='center', va='center')
            plt.gca().set_xlim([0, 1])
            plt.gca().set_ylim([0, 1])
        plt.title(f"学生 {title.split(' vs ')[1]}")
        plt.colorbar()
        plt.axis('off')
        
        plt.suptitle(title)
        filename = f'./out/distillation_feature/feature_pair_{epoch}_{i}.png'
        plt.savefig(filename)
        plt.close()
        figs.append(fig)
    
    return figs





# 设置优化器和学习率调度器
optimizer = optim.Adam(modelv4.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
distill_criterion = ImprovedDistillationLoss(alpha=0.7, temp=4.0).to(device)

def train(epoch):
    modelv4.train()
    modelv3.eval()
    
    running_loss = 0.0
    cls_losses = 0.0
    reg_losses = 0.0
    soft_cls_losses = 0.0
    soft_reg_losses = 0.0
    feature_losses = 0.0
    attention_losses = 0.0  # 新增
    
    for i, (images, cls_targets, reg_targets) in enumerate(train_loader):
        images = images.to(device)
        cls_targets = cls_targets.to(device)
        reg_targets = reg_targets.to(device)
        
        # 获取模型输出
        with torch.no_grad():
            teacher_outputs = modelv3(images)
        student_outputs = modelv4(images)
        
        # 计算损失
        losses = distill_criterion(
            student_outputs, teacher_outputs, (cls_targets, reg_targets)
        )
        
        # 更新损失项
        loss, cls_loss, reg_loss, soft_cls_loss, soft_reg_loss, feature_loss, attention_loss = losses
        
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
        attention_losses += attention_loss.item()  # 新增
        
        if (i+1) % 10 == 0:
            print(f'Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, Cls: {cls_loss.item():.4f}, '
                  f'Reg: {reg_loss.item():.4f}, Feat: {feature_loss.item():.4f}, '
                  f'Attention: {attention_loss.item():.4f}')  # 新增
    
    visualize_attention_maps(modelv3, modelv4, images, epoch)
    visualize_feature_maps(modelv3, modelv4, epoch)

    # 返回平均损失
    return (running_loss / len(train_loader), 
            cls_losses / len(train_loader),
            reg_losses / len(train_loader),
            soft_cls_losses / len(train_loader),
            soft_reg_losses / len(train_loader),
            feature_losses / len(train_loader),
            attention_losses / len(train_loader))  # 新增



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
            
            reg_targets = reg_targets.float().view(-1, 1)  # 重塑为[batch_size, 1]


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


# 训练主循环
num_epochs = 100
best_val_acc = 0.0
train_losses = []
val_losses = []
val_accs = []

for epoch in range(num_epochs):
    # 训练一个epoch - 更新解包以匹配7个返回值
    train_loss, cls_loss, reg_loss, soft_cls_loss, soft_reg_loss, feature_loss, attention_loss = train(epoch)
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
          f'Soft-Cls: {soft_cls_loss:.4f}, Soft-Reg: {soft_reg_loss:.4f}, '
          f'Features: {feature_loss:.4f}, Attention: {attention_loss:.4f}')  # 添加注意力损失打印
    
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