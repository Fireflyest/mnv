import torch
import torch.nn as nn
import torchvision.models as models

class MultiHeadMobileNetV3(nn.Module):
    def __init__(self, num_classes):
        super(MultiHeadMobileNetV3, self).__init__()
        # 加载预训练的 MobileNetV3 模型
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        mobilenet = models.mobilenet_v3_large(weights=weights)
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
        # MobileNetV3-Large has a different architecture with different feature dimensions
        x1 = self.features[:2](x)        # Early layers
        x2 = self.features[2:4](x1)       # First bottleneck blocks
        x3 = self.features[4:7](x2)       # Second set of bottleneck blocks
        x4 = self.features[7:13](x3)     # Third set of bottleneck blocks
        x5 = self.features[13:](x4)      # Final layers
        
        x6 = torch.flatten(self.avgpool(x5), 1)
        x7 = self.classifier(x6)
        out1 = self.head1(x7)
        out2 = self.head2(x7)

        # 16 channels, 112×112   （stride=2）
        # 24 channels, 56×56     （stride=2）
        # 40 channels, 28×28     （stride=2）
        # 112 channels, 14×14    （stride=2）
        # 960 channels, 7×7      （stride=2）

        return out1, out2, x1, x2, x3, x4, x5



