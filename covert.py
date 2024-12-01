import torch

import mobilenetv4

# Load the model
model = mobilenetv4.MobileNetV4("MobileNetV4ConvSmall")
model.load_state_dict(torch.load('./out/mobilenetv4.pth', weights_only=True))
model.eval()

# Export the model to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, './out/mobilenetv4.onnx')
