import torch
import torch.nn as nn
import torchvision.models as models

class MultiHeadMobileNetV3(nn.Module):
    def __init__(self, num_classes):
        super(MultiHeadMobileNetV3, self).__init__()
        # 加载预训练的 MobileNetV3 模型
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self.mobilenet = models.mobilenet_v3_small(weights=weights)
        # 获取分类层的输入特征数
        in_features = self.mobilenet.classifier[3].in_features
        # 移除分类层
        self.mobilenet.classifier[3] = nn.Identity()
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
        x = self.mobilenet(x)
        # x: torch.Tensor = nn.functional.adaptive_avg_pool2d(x, 1)
        # x = torch.flatten(x, 1)
        out1 = self.head1(x)
        out2 = self.head2(x)
        return out1, out2, x



