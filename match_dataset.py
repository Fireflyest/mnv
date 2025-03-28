import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import glob
import re
import numpy as np


class MatchingDataset(Dataset):
    """
    数据集用于图像匹配任务，加载正样本和负样本对
    
    正样本：来自pos文件夹，标签为1
    负样本：来自neg文件夹，标签为0
    
    每对图像由形如'X_p.jpg'和'X_c.jpg'命名的文件组成
    """
    
    def __init__(self, root_dir, transform_p=None, transform_c=None, img_size=(224, 224)):
        """
        初始化数据集
        
        参数:
            root_dir (str): 包含pos和neg子文件夹的根目录
            transform (callable, optional): 可选的图像转换
            img_size (tuple): 调整图像大小的目标尺寸
        """
        self.root_dir = root_dir
        self.pos_dir = os.path.join(root_dir, 'pos')
        self.neg_dir = os.path.join(root_dir, 'neg')
        
        # 如果没有提供transform，创建一个默认的
        transform_default = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.transform_p = transform_p if transform_p is not None else transform_default
        self.transform_c = transform_c if transform_c is not None else transform_default
            
        # 收集所有样本对
        self.sample_pairs = self._collect_pairs()
        
    def _collect_pairs(self):
        """收集所有匹配的图像对及其标签"""
        pairs = []
        
        # 处理正样本 (label=1)
        pos_pairs = self._get_pairs_from_dir(self.pos_dir, label=1)
        pairs.extend(pos_pairs)
        
        # 处理负样本 (label=0)
        neg_pairs = self._get_pairs_from_dir(self.neg_dir, label=0)
        pairs.extend(neg_pairs)
        
        return pairs
    
    def _get_pairs_from_dir(self, directory, label):
        """从指定目录中找出所有匹配的图像对"""
        if not os.path.exists(directory):
            print(f"警告：目录不存在 {directory}")
            return []
        
        # 找出所有p图像
        p_images = glob.glob(os.path.join(directory, "*_p.*"))
        pairs = []
        
        for p_image in p_images:
            # 通过正则表达式提取索引
            basename = os.path.basename(p_image)
            match = re.match(r'(\d+)_p\.(jpg|png|jpeg)', basename, re.IGNORECASE)
            if match:
                index = match.group(1)
                ext = match.group(2)
                
                # 构建对应的c图像路径
                c_image = os.path.join(directory, f"{index}_c.{ext}")
                
                # 检查c图像是否存在
                if os.path.exists(c_image):
                    pairs.append((p_image, c_image, label))
                else:
                    print(f"警告：找不到与 {p_image} 匹配的图像")
        
        print(f"从 {directory} 加载了 {len(pairs)} 对图像")
        return pairs
    
    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.sample_pairs)
    
    def __getitem__(self, idx):
        """获取指定索引的样本对和标签"""
        img1_path, img2_path, label = self.sample_pairs[idx]
        
        # 加载图像
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # 应用变换
        img1 = self.transform_p(img1)
        img2 = self.transform_c(img2)
            
        return img1, img2, torch.tensor(label, dtype=torch.float32)


# 使用示例
if __name__ == "__main__":
    dataset_root = 'G:\\projects\\ai\\mnv\\data\\huali\\match1'
    
    # 创建数据集实例
    dataset = MatchingDataset(root_dir=dataset_root)
    print(f"数据集大小: {len(dataset)}")
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 获取一个批次的样本并显示
    for batch_idx, (imgs1, imgs2, labels) in enumerate(dataloader):
        print(f"批次 {batch_idx+1}:")
        print(f"图像1形状: {imgs1.shape}")
        print(f"图像2形状: {imgs2.shape}")
        print(f"标签: {labels}")
        
        # 只显示第一个批次然后退出
        break
    
    # 简单查看正负样本分布
    pos_count = sum(1 for _, _, label in dataset if label == 1)
    neg_count = sum(1 for _, _, label in dataset if label == 0)
    
    print(f"正样本数量: {pos_count}")
    print(f"负样本数量: {neg_count}")

    # 预览一些样本图像
    import matplotlib.pyplot as plt
    
    def show_sample_images(dataset, num_samples=3):
        """显示数据集中的一些样本图像"""
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3*num_samples))
        
        # 确保我们能获取到足够的样本
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        for i, idx in enumerate(indices):
            img1, img2, label = dataset[idx]
            
            # 将图像从张量转换为numpy数组用于显示
            img1_np = img1.permute(1, 2, 0).numpy()
            img2_np = img2.permute(1, 2, 0).numpy()
            
            # 反归一化以获得更自然的显示效果
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img1_np = std * img1_np + mean
            img2_np = std * img2_np + mean
            
            # 确保像素值在[0,1]范围内
            img1_np = np.clip(img1_np, 0, 1)
            img2_np = np.clip(img2_np, 0, 1)
            
            # 在图中显示图像
            axes[i, 0].imshow(img1_np)
            axes[i, 1].imshow(img2_np)
            
            label_text = "true" if label.item() == 1 else "false"
            axes[i, 0].set_title(f" {i+1} - {label_text}")
            axes[i, 1].set_title(f" {i+1} - {label_text}")
            
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # 显示一些样本图像
    show_sample_images(dataset, num_samples=3)

