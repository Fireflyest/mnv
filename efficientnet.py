import torch
import torch.nn as nn
import torchvision.models as models

class MultiHeadEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiHeadEfficientNet, self).__init__()
        # 加载预训练的 EfficientNet-B0 模型
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.efficientnet = models.efficientnet_b0(weights=weights)

        # 分类系列拆出来用于输出特征
        self.classifier = self.efficientnet.classifier
        self.efficientnet.classifier = nn.Identity()
        # 获取分类层的输入特征数，把分类层的线性层替换为一个空层用于连接自定义的头
        in_features = self.classifier[1].in_features
        self.classifier[1] = nn.Identity()
        # 添加多头输出
        self.head1 = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.Softmax(dim=1)
        )
        self.head2 = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.efficientnet(x)
        x2 = self.classifier(x1)
        out1 = self.head1(x2)
        out2 = self.head2(x2)
        return out1, out2, x1

weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
efficientnet = models.efficientnet_b0(weights=weights)
print(efficientnet)
