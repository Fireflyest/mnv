import torch
import torchvision.models as models
from match_model import ContrastiveMatcher


# 加载预训练的模型
model = ContrastiveMatcher(feature_dim=752, projection_dim=128, temperature=0.07)
model.load_from_path('.\checkpoints\matcher_best.pth')
model.eval()

# Export the model to ONNX
out_name = './out/contrastive_match.onnx'
# 创建两个752维向量作为输入（批量大小为1）
dummy_input1 = torch.randn(1, 752)
dummy_input2 = torch.randn(1, 752)
dummy_inputs = (dummy_input1, dummy_input2)

torch.onnx.export(model, dummy_inputs, out_name, 
                  input_names=['input1', 'input2'], 
                  output_names=['logic_output'],
                  dynamic_axes={'input1': {0: 'batch_size'},
                                'input2': {0: 'batch_size'},
                                'logic_output': {0: 'batch_size'}})

print('Model exported to ONNX.')