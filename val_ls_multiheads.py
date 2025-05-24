import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import random
import numpy as np
import cv2

from lsnet_multi import load_lsnet_multitask
import data
import vision

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"使用设备: {device}")

# 设置数据加载器
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 加载测试数据集
dataset = data.HuaLiDataset_Multi(root_dir='./data/huali/test1', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 随机选择展示图片的数量
num_images = 12
random_indices = random.sample(range(len(dataset)), num_images)
selected_images = [dataset[i] for i in random_indices]

# 准备用于预测的图片张量
images = torch.stack([img for img, _, _ in selected_images])
images = images.to(device)

# 加载训练好的LSNet多任务模型
model = load_lsnet_multitask(
    backbone='lsnet_t', 
    num_classes=5, 
    pretrained=False,
    checkpoint_path='./out/lsnet_multitask_best.pth'
)
model.eval()
model.to(device)

# 确保模型中的所有组件（包括attention_biases）都在同一设备上
for module in model.modules():
    if hasattr(module, 'attention_biases'):
        module.attention_biases = module.attention_biases.to(device)
    if hasattr(module, 'ab'):
        module.ab = module.ab.to(device)

# 选择用于Grad-CAM的目标层
# LSNet的最后一个块作为可视化层
target_layer = model.backbone.blocks4[-1]

# 初始化Grad-CAM
class LSNetGradCAM(vision.GradCAM):
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        
        # 注册钩子函数
        self.handle_forward = target_layer.register_forward_hook(self.forward_hook)
        self.handle_backward = target_layer.register_full_backward_hook(self.backward_hook)
        
    def forward_hook(self, module, input, output):
        self.feature_maps = output
        
    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]
        
    def __del__(self):
        # 移除钩子
        self.handle_forward.remove()
        self.handle_backward.remove()
        
    def generate_cam(self, input_tensor, target_category=None):
        # 处理LSNet的输出字典格式
        self.model.eval()
        self.model.zero_grad()
        
        # 获取模型输出
        output = self.model(input_tensor)
        classification_output = output['classification']
        
        if target_category is None:
            target_category = torch.argmax(classification_output, dim=1).item()
            
        # 创建one-hot向量
        one_hot = torch.zeros(classification_output.size(), device=input_tensor.device)
        one_hot[0, target_category] = 1
        
        # 反向传播
        classification_output.backward(gradient=one_hot, retain_graph=True)
        
        # 计算权重
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # 计算CAM
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = torch.relu(cam)  # ReLU激活
        
        # 归一化
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        return cam.squeeze().cpu().detach().numpy()

# 初始化LSNet的Grad-CAM
grad_cam = LSNetGradCAM(model, target_layer)

# 进行预测并生成CAM
predictions = []
with torch.no_grad():
    outputs = model(images)
    classification_outputs = outputs['classification']
    logic_outputs = outputs['logic']
    
    # 获取分类预测和概率
    probabilities, predicted_classes = torch.max(classification_outputs, 1)
    # 获取逻辑预测
    logic_predictions = (logic_outputs > 0.5).float()

# 打印预测结果
for i, idx in enumerate(random_indices):
    print(f"图片 {idx} 预测分类: {predicted_classes[i].item()} ({dataset.classes[predicted_classes[i].item()]})， 概率: {probabilities[i].item():.4f}， 逻辑判断: {logic_predictions[i].item()}")

# 可视化预测结果和Grad-CAM
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

for i, (img, label, logic_label) in enumerate(selected_images):
    if i >= len(axes):
        break
        
    # 将图像张量转换为NumPy数组用于显示
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)
    
    # 为当前分类生成Grad-CAM
    input_tensor = images[i].unsqueeze(0)
    target_category = predicted_classes[i].item()
    cam = grad_cam.generate_cam(input_tensor, target_category)
    
    # 确保CAM与图像尺寸一致
    if cam.shape != img_np.shape[:2]:
        cam = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    
    # 在图像上显示Grad-CAM
    cam_image = vision.show_cam_on_image(img_np, cam, threshold=0.2)
    
    # 显示图像
    axes[i].imshow(cam_image)
    axes[i].set_title(
        f"Pred: {dataset.classes[predicted_classes[i].item()]} ({probabilities[i].item():.2f})\n"
        f"True: {dataset.classes[label]}, Logic: {logic_predictions[i].item()}"
    )
    axes[i].axis("off")

plt.tight_layout()
plt.savefig('./out/val_ls_cam_multiheads.png')
plt.show()

# 计算测试集上的整体性能
def evaluate_model(model, dataloader, device):
    model.eval()
    correct_cls = 0
    correct_logic = 0
    total = 0
    
    cls_confusion_matrix = torch.zeros(5, 5)  # 假设有5个类别
    
    with torch.no_grad():
        for data, target_cls, target_logic in dataloader:
            data = data.to(device)
            target_cls = target_cls.to(device)
            target_logic = target_logic.to(device).float()
            
            # 前向传播
            outputs = model(data)
            output_cls = outputs['classification']
            output_logic = outputs['logic'].squeeze()
            
            # 计算准确率
            _, predicted_cls = torch.max(output_cls.data, 1)
            predicted_logic = (output_logic > 0.5).float()
            
            # 更新混淆矩阵
            for t, p in zip(target_cls.view(-1), predicted_cls.view(-1)):
                cls_confusion_matrix[t.long(), p.long()] += 1
                
            # 统计正确预测数
            total += target_cls.size(0)
            correct_cls += (predicted_cls == target_cls).sum().item()
            correct_logic += (predicted_logic == target_logic).sum().item()
    
    # 计算各类别的精确率和召回率
    precision = torch.zeros(5)
    recall = torch.zeros(5)
    
    for i in range(5):
        precision[i] = cls_confusion_matrix[i, i] / cls_confusion_matrix[:, i].sum() if cls_confusion_matrix[:, i].sum() > 0 else 0
        recall[i] = cls_confusion_matrix[i, i] / cls_confusion_matrix[i, :].sum() if cls_confusion_matrix[i, :].sum() > 0 else 0
    
    # 计算总体准确率
    cls_acc = correct_cls / total
    logic_acc = correct_logic / total
    
    return {
        'cls_acc': cls_acc,
        'logic_acc': logic_acc,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cls_confusion_matrix
    }

# 评估模型
print("正在评估模型在测试集上的性能...")
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
results = evaluate_model(model, test_loader, device)

# 打印结果
print(f"分类准确率: {results['cls_acc']:.4f}")
print(f"逻辑判断准确率: {results['logic_acc']:.4f}")
print("\n分类精确率:")
for i, p in enumerate(results['precision']):
    print(f"类别 {i} ({dataset.classes[i]}): {p:.4f}")
print("\n分类召回率:")
for i, r in enumerate(results['recall']):
    print(f"类别 {i} ({dataset.classes[i]}): {r:.4f}")

# 可视化混淆矩阵
plt.figure(figsize=(10, 8))
plt.imshow(results['confusion_matrix'], interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# 设置坐标刻度和标签
classes = dataset.classes
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# 在混淆矩阵中显示数值
thresh = results['confusion_matrix'].max() / 2.
for i in range(results['confusion_matrix'].shape[0]):
    for j in range(results['confusion_matrix'].shape[1]):
        plt.text(j, i, int(results['confusion_matrix'][i, j]),
                 horizontalalignment="center",
                 color="white" if results['confusion_matrix'][i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('./out/confusion_matrix_ls.png')
plt.show()
