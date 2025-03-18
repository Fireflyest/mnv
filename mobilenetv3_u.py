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

        pool = nn.AdaptiveAvgPool2d((1, 1))
        f1 = torch.flatten(pool(x1), 1)      # [B, 16]
        f2 = torch.flatten(pool(x2), 1)      # [B, 24]
        f3 = torch.flatten(pool(x3), 1)      # [B, 40]
        f4 = torch.flatten(pool(x4), 1)      # [B, 80]
        f5 = torch.flatten(pool(x5), 1)      # [B, 576]
        
        # Concatenate all features
        features = torch.cat([f1, f2, f3, f4, f5], dim=1)  # [B, 16+24+40+80+576=736]
        return out1, out2, features


# 输出网络结构
model = MultiHeadMobileNetV3(num_classes=5)
print(model)