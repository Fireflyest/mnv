import torch
import torch.nn as nn
from lsnet import lsnet_t, lsnet_s, lsnet_b
import os
import timm


class LSNetMultiTask(nn.Module):
    def __init__(self, backbone='lsnet_t', num_classes=5, pretrained=True):
        super().__init__()
        
        # 加载预训练的LSNet骨干网络
        if backbone == 'lsnet_t':
            self.backbone = lsnet_t()  # 先不加载预训练权重
        elif backbone == 'lsnet_s':
            self.backbone = lsnet_s()
        elif backbone == 'lsnet_b':
            self.backbone = lsnet_b()
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")
        
        # 手动加载预训练权重
        if pretrained:
            pretrain_path = f"./pretrain/{backbone}.pth"
            if os.path.exists(pretrain_path):
                print(f"从本地加载预训练权重: {pretrain_path}")
                checkpoint = torch.load(pretrain_path, map_location='cpu', weights_only=True)
                # 检查checkpoint的格式，适配不同的保存方式
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # 移除可能不匹配的键
                model_dict = self.backbone.state_dict()
                state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                self.backbone.load_state_dict(state_dict, strict=False)
                print(f"成功加载预训练权重")
            else:
                print(f"未找到预训练权重文件: {pretrain_path}")
        
        # 获取特征维度
        self.feature_dim = self.backbone.num_features
        
        # 移除原始分类头
        if hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
        if hasattr(self.backbone, 'head_dist'):
            self.backbone.head_dist = nn.Identity()
        
        # 定义新的输出头
        # 1. 分类头 - 5个类别
        self.classification_head = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        # 2. 逻辑判断头 - 二分类问题
        self.logic_head = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Linear(self.feature_dim, 1),
            nn.Sigmoid()
        )
        
        # 3. 特征图转换 - 保持最后一层特征维度不变
        self.feature_proj = nn.Identity()
        
        # 初始化新添加的层
        self._init_weights()
    
    def _init_weights(self):
        # 初始化分类头
        nn.init.trunc_normal_(self.classification_head[1].weight, std=0.02)
        nn.init.constant_(self.classification_head[1].bias, 0)
        
        # 初始化逻辑头
        nn.init.trunc_normal_(self.logic_head[1].weight, std=0.02)
        nn.init.constant_(self.logic_head[1].bias, 0)
    
    def forward(self, x):
        # 提取特征
        features = self.backbone.patch_embed(x)
        features = self.backbone.blocks1(features)
        features = self.backbone.blocks2(features)
        features = self.backbone.blocks3(features)
        features = self.backbone.blocks4(features)
        
        # 保存最后一层特征图
        feature_maps = features
        
        # 全局平均池化得到特征向量
        pooled_features = torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
        
        # 分类预测
        classification_output = self.classification_head(pooled_features)
        
        # 逻辑判断
        logic_output = self.logic_head(pooled_features)
        
        # 特征图输出
        feature_output = self.feature_proj(feature_maps)
        
        return {
            'classification': classification_output,
            'logic': logic_output,
            'features': feature_output
        }


def load_lsnet_multitask(backbone='lsnet_t', num_classes=5, pretrained=True, checkpoint_path=None):
    """
    加载多任务LSNet模型
    
    Args:
        backbone: 使用的骨干网络，可选 'lsnet_t', 'lsnet_s', 'lsnet_b'
        num_classes: 分类头的类别数
        pretrained: 是否使用预训练权重
        checkpoint_path: 微调后的检查点路径，如果提供则从此加载权重
        
    Returns:
        多任务LSNet模型
    """
    model = LSNetMultiTask(backbone=backbone, num_classes=num_classes, pretrained=pretrained)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"成功加载检查点: {checkpoint_path}")
    
    return model


# 使用示例
if __name__ == "__main__":
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = load_lsnet_multitask(backbone='lsnet_t', num_classes=5, pretrained=True)
    model = model.to(device)
    
    # 将模型设置为评估模式
    model.eval()
    
    # 测试前向传播
    with torch.no_grad():  # 在评估模式下不需要计算梯度
        x = torch.randn(1, 3, 224, 224).to(device)
        outputs = model(x)
    
    # 打印输出形状
    print(f"分类输出形状: {outputs['classification'].shape}")  # 应为 [1, 5]
    print(f"逻辑输出形状: {outputs['logic'].shape}")  # 应为 [1, 1]
    print(f"特征图输出形状: {outputs['features'].shape}")  # 应为 [1, 384, h, w]