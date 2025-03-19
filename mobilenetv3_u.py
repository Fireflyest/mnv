import torch
import torch.nn as nn
import torchvision.models as models

class MultiHeadMobileNetV3(nn.Module):
    def __init__(self, num_classes):
        super(MultiHeadMobileNetV3, self).__init__()
        # 加载预训练的 MobileNetV3 模型
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        mobilenet = models.mobilenet_v3_small(weights=weights)
        self.features = mobilenet.features
        # 分类系列拆出来用于输出特征
        self.avgpool = mobilenet.avgpool
        self.classifier = mobilenet.classifier
        mobilenet.classifier = nn.Identity()
        mobilenet.avgpool = nn.Identity()
        # 获取分类层的输入特征数，把分类层的线性层替换为一个空层用于连接自定义的头
        classifier_in_features = self.classifier[3].in_features
        self.classifier[3] = nn.Identity()
        # 添加多头输出
        self.head1 = nn.Sequential(
            nn.Linear(classifier_in_features, num_classes),
            nn.Softmax(dim=1)
        )
        self.head2 = nn.Sequential(
            nn.Linear(classifier_in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.features[:2](x)       # [B, 16, H/2, W/2]
        x2 = self.features[2:4](x1)     # [B, 24, H/4, W/4]
        x3 = self.features[4:7](x2)     # [B, 40, H/8, W/8]
        x4 = self.features[7:10](x3)    # [B, 80, H/16, W/16]
        x5 = self.features[10:](x4)     # [B, 576, H/32, W/32]
        x6 = torch.flatten(self.avgpool(x5), 1)
        x7 = self.classifier(x6)
        out1 = self.head1(x7)
        out2 = self.head2(x7)

        # 反注意力，加强背景特征
        spatial_attention = torch.mean(x5, dim=1, keepdim=True)  # [B, 1, H/32, W/32]
        spatial_attention = torch.sigmoid(spatial_attention)
        x5 = x5 * (1 - spatial_attention)  # 逐元素相乘，广播机制会自动扩展

        pool = nn.AdaptiveAvgPool2d((1, 1))
        f1 = torch.flatten(pool(x1), 1)      # [B, 16]
        f2 = torch.flatten(pool(x2), 1)      # [B, 24]
        f3 = torch.flatten(pool(x3), 1)      # [B, 40]
        f4 = torch.flatten(pool(x4), 1)      # [B, 80]
        f5 = torch.flatten(pool(x5), 1)      # [B, 576]
        
        # Concatenate all features
        features = torch.cat([f1, f2, f3, f4, f5], dim=1)  # [B, 16+24+40+80+576=736]
        return out1, out2, features

class MobileNetV3UNet(nn.Module):
    def __init__(self, num_classes=5, output_channels=3):
        super(MobileNetV3UNet, self).__init__()
        # Load pretrained MobileNetV3 as encoder
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        mobilenet = models.mobilenet_v3_small(weights=weights)
        self.features = mobilenet.features
        
        # Original classification components
        self.avgpool = mobilenet.avgpool
        self.classifier = mobilenet.classifier
        mobilenet.classifier = nn.Identity()
        mobilenet.avgpool = nn.Identity()
        classifier_in_features = self.classifier[3].in_features
        self.classifier[3] = nn.Identity()
        
        # Multi-head outputs for classification tasks
        self.head1 = nn.Sequential(
            nn.Linear(classifier_in_features, num_classes),
            nn.Softmax(dim=1)
        )
        self.head2 = nn.Sequential(
            nn.Linear(classifier_in_features, 1),
            nn.Sigmoid()
        )
        
        # Decoder path (upsampling)
        # Decoder block 1: x5 [B, 576, H/32, W/32] -> [B, 80, H/16, W/16]
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(576, 80, kernel_size=2, stride=2),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True)
        )
        
        # Decoder block 2: [B, 176, H/16, W/16] -> [B, 40, H/8, W/8]
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(176, 40, kernel_size=2, stride=2),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True)
        )
        
        # Decoder block 3: [B, 80, H/8, W/8] -> [B, 24, H/4, W/4]
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(80, 24, kernel_size=2, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        
        # Decoder block 4: [B, 48, H/4, W/4] -> [B, 16, H/2, W/2]
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(48, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Final upsampling: [B, 32, H/2, W/2] -> [B, output_channels, H, W]
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(32, output_channels, kernel_size=2, stride=2),
            nn.Sigmoid()  # Sigmoid for output in range [0,1]
        )

    def forward(self, x):
        # Encoder path (same as original)
        x1 = self.features[:2](x)       # [B, 16, H/2, W/2]
        x2 = self.features[2:4](x1)     # [B, 24, H/4, W/4]
        x3 = self.features[4:7](x2)     # [B, 40, H/8, W/8]
        x4 = self.features[7:10](x3)    # [B, 80, H/16, W/16]
        x5 = self.features[10:](x4)     # [B, 576, H/32, W/32]
        
        # Classification path
        x6 = torch.flatten(self.avgpool(x5), 1)
        x7 = self.classifier(x6)
        out1 = self.head1(x7)
        out2 = self.head2(x7)
        
        # Feature extraction for original multi-task output
        pool = nn.AdaptiveAvgPool2d((1, 1))
        f1 = torch.flatten(pool(x1), 1)      # [B, 16]
        f2 = torch.flatten(pool(x2), 1)      # [B, 24]
        f3 = torch.flatten(pool(x3), 1)      # [B, 40]
        f4 = torch.flatten(pool(x4), 1)      # [B, 80]
        f5 = torch.flatten(pool(x5), 1)      # [B, 576]
        features = torch.cat([f1, f2, f3, f4, f5], dim=1)  # [B, 736]
        
        # Decoder path with skip connections
        d1 = self.decoder1(x5)                    # [B, 80, H/16, W/16]
        # Resize x4 to match d1's spatial dimensions
        x4_resized = nn.functional.interpolate(x4, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, x4_resized], dim=1)   # [B, 160(80+80), H/16, W/16]
        
        d2 = self.decoder2(d1)                    # [B, 40, H/8, W/8]
        # Resize x3 to match d2's spatial dimensions
        x3_resized = nn.functional.interpolate(x3, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, x3_resized], dim=1)   # [B, 80, H/8, W/8]
        
        d3 = self.decoder3(d2)                    # [B, 24, H/4, W/4]
        # Resize x2 to match d3's spatial dimensions
        x2_resized = nn.functional.interpolate(x2, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, x2_resized], dim=1)   # [B, 48, H/4, W/4]
        
        d4 = self.decoder4(d3)                    # [B, 16, H/2, W/2]
        # Resize x1 to match d4's spatial dimensions
        x1_resized = nn.functional.interpolate(x1, size=d4.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, x1_resized], dim=1)   # [B, 32, H/2, W/2]
        
        reconstruction = self.decoder5(d4)        # [B, output_channels, H, W]
        
        return out1, out2, features, reconstruction

# Create and display the model
model = MobileNetV3UNet(num_classes=5, output_channels=3)
print(model)

# 输出网络结构
model = MultiHeadMobileNetV3(num_classes=5)
print(model)