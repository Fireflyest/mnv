import torch
import torchvision.models as models
import mobilenetv3


# 加载预训练的 MobileNetV3 模型
model = mobilenetv3.MultiHeadMobileNetV3(num_classes=5)
state_dict = torch.load('./out/mobilenetv3_multi_best_finetuned.pth', weights_only=True)
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('mobilenet.classifier'):
        new_key = k.replace('mobilenet.', '')
        new_state_dict[new_key] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)
model.eval()

# Export the model to ONNX
out_name = './out/mobilenetv3_multi.onnx'
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, out_name, 
                  input_names=['input'], 
                  output_names=['class_output', 'logic_output', 'features_output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 
                                'class_output': {0: 'batch_size'}, 
                                'logic_output': {0: 'batch_size'}, 
                                'features_output': {0: 'batch_size'}})

print('Model exported to ONNX.')