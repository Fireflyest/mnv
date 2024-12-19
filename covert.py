import torch

import mobilenetv4

# Load the model
model = mobilenetv4.MobileNet(class_nums=5)
model.load_state_dict(torch.load('./out/mobilenetv4.pth', weights_only=True))
model.eval()

# Export the model to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, './out/mobilenetv4.onnx', 
                  input_names=['input'], 
                  output_names=['class_output', 'moire_output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'class_output': {0: 'batch_size'}, 'moire_output': {0: 'batch_size'}})

print('Model exported to ONNX.')