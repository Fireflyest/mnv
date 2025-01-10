import torch
import torchvision.models as models
import mobilenetv3


# 加载预训练的 MobileNetV3 模型
model = mobilenetv3.MultiHeadMobileNetV3(num_classes=5)
model.load_state_dict(torch.load('./out/mobilenetv3_multi_best_finetuned.pth', weights_only=True))
model.eval()

# Export the model to ONNX
out_name = './out/mobilenetv3_multi.onnx'
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, out_name, 
                  input_names=['input'], 
                  output_names=['class_output', 'logic_output', 'features'],
                  dynamic_axes={'input': {0: 'batch_size'}, 
                                'class_output': {0: 'batch_size'}, 
                                'logic_output': {0: 'batch_size'}, 
                                'features': {0: 'batch_size'}})

print('Model exported to ONNX.')