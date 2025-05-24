import torch
from lsnet_multi import load_lsnet_multitask
import os

# 确保输出目录存在
os.makedirs('./out', exist_ok=True)

# 强制使用CPU以避免Triton编译错误
device = torch.device("cpu")
torch.set_num_threads(4)  # 使用多线程加速CPU计算

# 加载多任务LSNet模型
model = load_lsnet_multitask(backbone='lsnet_t', num_classes=5, pretrained=False, 
                             checkpoint_path='./out/lsnet_multitask_best.pth')

# 将模型和所有子组件移至CPU
model = model.to(device)
model.eval()

# 创建一个包装类，使模型输出ONNX兼容格式
class LSNetWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        outputs = self.model(x)
        return (outputs['classification'], 
                outputs['logic'], 
                outputs['features'])

# 包装模型
wrapped_model = LSNetWrapper(model)
wrapped_model.eval()

# 导出为ONNX模型
out_name = './out/lsnet_multitask.onnx'
dummy_input = torch.randn(1, 3, 224, 224, device=device)

print('开始导出模型到ONNX...')
torch.onnx.export(wrapped_model, dummy_input, out_name,
                  export_params=True,
                  opset_version=12,
                  do_constant_folding=True,
                  input_names=['input'], 
                  output_names=['classification', 'logic', 'features'],
                  dynamic_axes={'input': {0: 'batch_size'}, 
                                'classification': {0: 'batch_size'}, 
                                'logic': {0: 'batch_size'}, 
                                'features': {0: 'batch_size'}})

print('Model exported to ONNX.')
