import torch
import torchvision.models as models
import mobilenetv4





# # Load the model
# model = mobilenetv4.MobileNet(class_nums=5)
# model.load_state_dict(torch.load('./out/mobilenetv4.pth', weights_only=True))
# model.eval()

# 加载预训练的 MobileNetV3 模型
model = models.mobilenet_v3_small()
# 修改最后一层
num_classes = 5
model.classifier[3] = torch.nn.Sequential(
    torch.nn.Linear(model.classifier[3].in_features, num_classes),
    torch.nn.Softmax(dim=1)
)
model.load_state_dict(torch.load('./out/mobilenetv3_best_finetuned.pth', weights_only=True))
model.eval()




# Export the model to ONNX
out_name = './out/mobilenet.onnx'
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, out_name, 
                  input_names=['input'], 
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print('Model exported to ONNX.')