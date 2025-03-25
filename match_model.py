import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large
import torch.nn.functional as F

class BaseMatchingNetwork(nn.Module):
    """
    特征匹配网络的基类
    """
    def __init__(self, feature_dim):
        super(BaseMatchingNetwork, self).__init__()
        self.feature_dim = feature_dim
        
    def forward(self, feat1, feat2):
        """
        前向传播方法，需要在子类中实现
        """
        raise NotImplementedError("子类必须实现forward方法")

class MatchingMLP(BaseMatchingNetwork):
    """
    用于匹配两个特征向量的MLP网络
    """
    def __init__(self, feature_dim, hidden_dims=[512, 256, 128]):
        super(MatchingMLP, self).__init__(feature_dim)
        
        # 输入维度是两个特征向量的拼接
        input_dim = feature_dim * 2
        
        # 构建MLP层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # 最终的分类层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, feat1, feat2):
        """
        前向传播，匹配两个特征向量
        
        参数:
            feat1: 第一个特征向量, 形状为 [batch_size, feature_dim]
            feat2: 第二个特征向量, 形状为 [batch_size, feature_dim]
            
        返回:
            匹配得分, 形状为 [batch_size, 1]
        """
        # 拼接特征向量
        combined = torch.cat([feat1, feat2], dim=1)
        
        # 通过MLP进行匹配
        logits = self.mlp(combined)
        
        return torch.sigmoid(logits)

class CosineSimilarityMatcher(BaseMatchingNetwork):
    """
    基于余弦相似度的特征匹配网络
    """
    def __init__(self, feature_dim):
        super(CosineSimilarityMatcher, self).__init__(feature_dim)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )
        
    def forward(self, feat1, feat2):
        """
        使用余弦相似度计算两个特征向量的匹配得分
        """
        # 投影特征到同一空间
        feat1_proj = self.projection(feat1)
        feat2_proj = self.projection(feat2)
        
        # 标准化特征向量
        feat1_norm = F.normalize(feat1_proj, p=2, dim=1)
        feat2_norm = F.normalize(feat2_proj, p=2, dim=1)
        
        # 计算余弦相似度
        cos_sim = torch.sum(feat1_norm * feat2_norm, dim=1, keepdim=True)
        
        # 将相似度映射到[0,1]范围
        return (cos_sim + 1) / 2

class EuclideanMatcher(BaseMatchingNetwork):
    """
    基于欧氏距离的特征匹配网络
    """
    def __init__(self, feature_dim):
        super(EuclideanMatcher, self).__init__(feature_dim)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        
    def forward(self, feat1, feat2):
        """
        使用欧氏距离计算两个特征向量的匹配得分
        """
        # 投影特征到同一空间
        feat1_proj = self.projection(feat1)
        feat2_proj = self.projection(feat2)
        
        # 计算欧氏距离
        distance = torch.sqrt(torch.sum((feat1_proj - feat2_proj)**2, dim=1, keepdim=True) + 1e-8)
        
        # 将距离转换为相似度分数（距离越小，相似度越高）
        sim_score = torch.exp(-distance)
        
        return sim_score

class CrossAttentionMatcher(BaseMatchingNetwork):
    """
    基于交叉注意力机制的特征匹配网络
    """
    def __init__(self, feature_dim, num_heads=4):
        super(CrossAttentionMatcher, self).__init__(feature_dim)
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Make sure embed_dim is divisible by num_heads
        self.embed_dim = num_heads * (256 // num_heads)
        
        # 特征投影层
        self.projection = nn.Linear(feature_dim, self.embed_dim)
        
        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim, 
            num_heads=num_heads
        )
        
        # 最终分类层
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, feat1, feat2):
        """
        使用交叉注意力机制计算两个特征向量的匹配得分
        """
        # 投影特征
        feat1_proj = self.projection(feat1).unsqueeze(1)  # [B, 1, 256]
        feat2_proj = self.projection(feat2).unsqueeze(1)  # [B, 1, 256]
        
        # 应用交叉注意力
        attn_output1, _ = self.cross_attention(feat1_proj, feat2_proj, feat2_proj)
        attn_output2, _ = self.cross_attention(feat2_proj, feat1_proj, feat1_proj)
        
        # 拼接注意力特征
        combined = torch.cat([
            attn_output1.squeeze(1), 
            attn_output2.squeeze(1)
        ], dim=1)
        
        # 计算匹配得分
        return self.classifier(combined)

class CombinedMLPCosineMatcher(BaseMatchingNetwork):
    """
    结合MLP和余弦相似度的特征匹配网络
    """
    def __init__(self, feature_dim, hidden_dims=[256, 128]):
        super(CombinedMLPCosineMatcher, self).__init__(feature_dim)
        
        # MLP部分
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1])
        )
        
        # 结合部分 - 将余弦相似度和原始特征一起送入MLP
        combined_dim = 1 + feature_dim * 2  # 余弦相似度(1) + 原始特征拼接(feature_dim*2)
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, feat1, feat2):
        """
        结合余弦相似度和MLP进行特征匹配
        """
        # 投影特征
        feat1_proj = self.projection(feat1)
        feat2_proj = self.projection(feat2)
        
        # 标准化特征向量以计算余弦相似度
        feat1_norm = F.normalize(feat1_proj, p=2, dim=1)
        feat2_norm = F.normalize(feat2_proj, p=2, dim=1)
        
        # 计算余弦相似度
        cos_sim = torch.sum(feat1_norm * feat2_norm, dim=1, keepdim=True)
        
        # 拼接余弦相似度和原始特征
        combined = torch.cat([cos_sim, feat1, feat2], dim=1)
        
        # 最终分类
        return self.classifier(combined)

class ContrastiveMatcher(BaseMatchingNetwork):
    """
    基于对比学习的特征匹配网络
    """
    def __init__(self, feature_dim, projection_dim=128, temperature=0.07):
        super(ContrastiveMatcher, self).__init__(feature_dim)
        self.temperature = temperature
        
        # 投影网络 - 将特征映射到较低维度的表示空间
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, projection_dim)
        )
        
        # 分类头 - 用于最终的匹配分数
        input_dim = 1 + projection_dim + projection_dim  # similarity(1) + feat_diff(projection_dim) + feat_prod(projection_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, feat1, feat2):
        """
        前向传播，基于对比学习进行特征匹配
        """
        # 投影特征到表示空间
        feat1_proj = self.projection(feat1)  # [B, projection_dim]
        feat2_proj = self.projection(feat2)  # [B, projection_dim]
        
        # L2归一化特征
        feat1_norm = F.normalize(feat1_proj, p=2, dim=1)
        feat2_norm = F.normalize(feat2_proj, p=2, dim=1)
        
        # 计算余弦相似度分数
        similarity = torch.sum(feat1_norm * feat2_norm, dim=1, keepdim=True)
        
        # 使用特征的相互关系而非简单拼接
        feat_diff = torch.abs(feat1_norm - feat2_norm)
        feat_prod = feat1_norm * feat2_norm
        
        # 结合相似度和特征交互信息
        combined_features = torch.cat([similarity, feat_diff, feat_prod], dim=1)
        # 拼接后shape: [B, projection_dim*2]
        score = self.classifier(combined_features)
        
        return score
    
    def get_embeddings(self, feat):
        """
        获取特征的归一化嵌入向量，用于对比学习
        """
        feat_proj = self.projection(feat)
        feat_norm = F.normalize(feat_proj, p=2, dim=1)
        return feat_norm
        
    def compute_infoNCE_loss(self, feat1_batch, feat2_batch, pos_mask=None):
        """
        计算InfoNCE损失
        
        参数:
            feat1_batch: 第一组特征向量 [batch_size, feature_dim]
            feat2_batch: 第二组特征向量 [batch_size, feature_dim]
            pos_mask: 指示哪些对是正样本对的掩码 [batch_size, batch_size]，若为None则假设对角线元素为正样本对
            
        返回:
            loss: InfoNCE损失值
        """
        # 获取归一化嵌入向量
        z1 = self.get_embeddings(feat1_batch)  # [B, projection_dim]
        z2 = self.get_embeddings(feat2_batch)  # [B, projection_dim]
        
        batch_size = z1.size(0)
        
        # 计算所有可能匹配的相似度矩阵
        similarity_matrix = torch.matmul(z1, z2.T) / self.temperature  # [B, B]
        
        # 如果未提供正样本掩码，默认对角线元素为正样本对
        if pos_mask is None:
            pos_mask = torch.eye(batch_size, device=z1.device)
            
        # 计算InfoNCE损失
        # 对于每个锚点，正样本对的对数似然
        exp_sim = torch.exp(similarity_matrix)
        
        # 对于每行(锚点)，计算所有可能匹配的分母之和
        log_prob = -torch.log(exp_sim / (exp_sim.sum(dim=1, keepdim=True) + 1e-8) + 1e-8)
        
        # 只保留正样本对的损失
        loss = (log_prob * pos_mask).sum() / pos_mask.sum()
        
        return loss


class InfoNCELoss(nn.Module):
    """
    计算对比学习的InfoNCE损失
    """
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features_1, features_2, labels=None):
        """
        计算成对的InfoNCE损失
        
        参数:
            features_1: 第一组特征 [batch_size, feature_dim]
            features_2: 第二组特征 [batch_size, feature_dim]
            labels: 如果提供，则具有相同标签的样本被视为正样本对
            
        返回:
            loss: InfoNCE损失
        """
        batch_size = features_1.size(0)
        device = features_1.device
        
        # L2归一化特征
        features_1 = F.normalize(features_1, p=2, dim=1)
        features_2 = F.normalize(features_2, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features_1, features_2.T) / self.temperature
        
        # 创建标签矩阵 - 如果未提供标签，则认为对角线元素为正样本对
        if labels is None:
            pos_mask = torch.eye(batch_size, device=device)
        else:
            labels = labels.view(-1, 1)
            pos_mask = (labels == labels.T).float()
        
        # 计算InfoNCE损失
        exp_sim = torch.exp(similarity_matrix)
        
        # 对于每行，计算分母（所有可能匹配的和）
        denominator = exp_sim.sum(dim=1, keepdim=True)
        
        # 计算每个正样本对的对数概率
        log_prob = torch.log(exp_sim / (denominator + 1e-8) + 1e-8)
        
        # 只对正样本对的概率求和
        loss = -(log_prob * pos_mask).sum() / pos_mask.sum()
        
        return loss


class ImageMatchingModel(nn.Module):
    """
    图像匹配模型，由特征提取器和匹配网络组成
    """
    def __init__(self, pretrained=True, backbone='small', matcher_type='mlp', num_heads=3, temperature=0.07):
        super(ImageMatchingModel, self).__init__()
        
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self.mobilenet = models.mobilenet_v3_small(weights=weights)
        self.mobilenet.classifier = nn.Identity()  # 移除分类层，只保留特征提取层
        
        feature_dim = 752  # 由MobileNetV3-Small提取的特征维度
        
        # 根据指定的匹配器类型创建匹配网络
        if matcher_type == 'mlp':
            self.matching_network = MatchingMLP(feature_dim=feature_dim)
        elif matcher_type == 'cosine':
            self.matching_network = CosineSimilarityMatcher(feature_dim=feature_dim)
        elif matcher_type == 'euclidean':
            self.matching_network = EuclideanMatcher(feature_dim=feature_dim)
        elif matcher_type == 'attention':
            self.matching_network = CrossAttentionMatcher(feature_dim=feature_dim, num_heads=num_heads)
        elif matcher_type == 'combined':
            self.matching_network = CombinedMLPCosineMatcher(feature_dim=feature_dim)
        elif matcher_type == 'contrastive':
            self.matching_network = ContrastiveMatcher(feature_dim=feature_dim, temperature=temperature)
            self.info_nce_loss = InfoNCELoss(temperature=temperature)
        else:
            raise ValueError(f"不支持的匹配器类型: {matcher_type}")
        
        self.matcher_type = matcher_type
    
    def forward(self, img1, img2):
        """
        前向传播，匹配两个图像
        
        参数:
            img1: 第一个图像, 形状为 [batch_size, channels, height, width]
            img2: 第二个图像, 形状为 [batch_size, channels, height, width]
            
        返回:
            匹配得分, 形状为 [batch_size, 1]
        """
        # 提取特征
        feat1 = self._extract_features(img1)
        feat2 = self._extract_features(img2)
        
        # 匹配特征
        score = self.matching_network(feat1, feat2)
        
        return score

    def _extract_features(self, image):
        with torch.no_grad():
            x1 = self.mobilenet.features[:2](image)       # [B, 16, H/2, W/2]
            x2 = self.mobilenet.features[2:4](x1)         # [B, 24, H/4, W/4]
            x3 = self.mobilenet.features[4:7](x2)         # [B, 40, H/8, W/8]
            x4 = self.mobilenet.features[7:10](x3)        # [B, 96, H/16, W/16]
            x5 = self.mobilenet.features[10:](x4)         # [B, 576, H/32, W/32]

            # Apply avgpool to each feature map
            pool = nn.AdaptiveAvgPool2d((1, 1))
            f1 = torch.flatten(pool(x1), 1)      # [B, 16]
            f2 = torch.flatten(pool(x2), 1)      # [B, 24]
            f3 = torch.flatten(pool(x3), 1)      # [B, 40]
            f4 = torch.flatten(pool(x4), 1)      # [B, 96]
            f5 = torch.flatten(pool(x5), 1)      # [B, 576]
            
            # Concatenate all features
            features = torch.cat([f1, f2, f3, f4, f5], dim=1)  # [B, 16+24+40+96+576=752]
            return features

    def compute_loss(self, img1_batch, img2_batch, labels, alpha=0.5):
        """
        计算组合损失（二元交叉熵 + InfoNCE对比损失）
        
        参数:
            img1_batch: 第一批图像 [batch_size, channels, height, width]
            img2_batch: 第二批图像 [batch_size, channels, height, width]
            labels: 标签，1表示匹配对，0表示非匹配对 [batch_size]
            alpha: InfoNCE损失的权重
            
        返回:
            total_loss: 总损失
            bce_loss: 二元交叉熵损失
            contrast_loss: 对比学习损失
        """
        # 提取特征
        feat1 = self._extract_features(img1_batch)
        feat2 = self._extract_features(img2_batch)
        
        # 计算预测分数
        scores = self.matching_network(feat1, feat2).squeeze()
        
        # 二元交叉熵损失
        bce_loss = F.binary_cross_entropy(scores, labels.float())
        
        # 如果是对比学习方法，计算额外的对比损失
        if self.matcher_type == 'contrastive':
            # 创建正样本掩码 - 相同标签的样本是正样本对
            labels_matrix = (labels.view(-1, 1) == labels.view(1, -1)).float()
            
            # 使用InfoNCE损失
            if isinstance(self.matching_network, ContrastiveMatcher):
                contrast_loss = self.matching_network.compute_infoNCE_loss(feat1, feat2, pos_mask=labels_matrix)
            else:
                contrast_loss = self.info_nce_loss(feat1, feat2, labels)
                
            # 组合损失
            total_loss = (1 - alpha) * bce_loss + alpha * contrast_loss
            
            return total_loss, bce_loss, contrast_loss
        else:
            # 对于非对比方法，只返回BCE损失
            return bce_loss, bce_loss, torch.tensor(0.0, device=bce_loss.device)


# 测试模型
if __name__ == "__main__":
    # 测试不同匹配方法
    matcher_types = ['mlp', 'cosine', 'euclidean', 'attention', 'combined', 'contrastive']
    batch_size = 4
    dummy_input1 = torch.randn(batch_size, 3, 224, 224)
    dummy_input2 = torch.randn(batch_size, 3, 224, 224)
    dummy_labels = torch.randint(0, 2, (batch_size,))  # 随机二进制标签
    
    for matcher_type in matcher_types:
        print(f"\n测试 {matcher_type} 匹配方法:")
        model = ImageMatchingModel(pretrained=True, backbone='small', matcher_type=matcher_type)
        
        # 测试前向传播
        with torch.no_grad():
            output = model(dummy_input1, dummy_input2)
            
        print(f"输出形状: {output.shape}")
        print(f"输出值: {output}")
        
        # 测试损失计算
        if matcher_type == 'contrastive':
            total_loss, bce_loss, contrast_loss = model.compute_loss(dummy_input1, dummy_input2, dummy_labels)
            print(f"总损失: {total_loss.item():.4f}, BCE损失: {bce_loss.item():.4f}, 对比损失: {contrast_loss.item():.4f}")
