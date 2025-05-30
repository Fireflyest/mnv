import torch
import torchvision.models as models
import mobilenetv4


# 加载预训练的 MobileNetV4 模型
model = mobilenetv4.MobileNetV4("MobileNetV4ConvSmall")
state_dict = torch.load('./out/mobilenetv4_distilled.pth', weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# Export the model to ONNX
out_name = './out/mobilenetv4_multi.onnx'
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, out_name, 
                  input_names=['input'], 
                  output_names=['class_output', 'logic_output', 'features_output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 
                                'class_output': {0: 'batch_size'}, 
                                'logic_output': {0: 'batch_size'}, 
                                'features_output': {0: 'batch_size'}})

print('Model exported to ONNX.')