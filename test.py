import torch
import torch.nn.functional as F

# 示例数据
data = torch.tensor([2.0, 1.0, 0.1])

# 计算 softmax
softmax_output = F.softmax(data, dim=0)
print("Softmax Output:", softmax_output)

# Min-Max 归一化
min_val = torch.min(data)
max_val = torch.max(data)
normalized_data_min_max = (data - min_val) / (max_val - min_val)
print("Min-Max Normalized Data:", normalized_data_min_max)

# Z-score 归一化
mean_val = torch.mean(data)
std_val = torch.std(data)
normalized_data_z_score = (data - mean_val) / std_val
print("Z-score Normalized Data:", normalized_data_z_score)