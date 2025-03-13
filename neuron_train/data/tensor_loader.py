import torch
import numpy as np

# 加载张量数据
tensor_path = "C:/Users/Joseph/Desktop/raybnn/neuron_train/data/neuron_graph_tensor.pt"
tensor_data = torch.load(tensor_path)

# 将张量转换为 NumPy 数组
numpy_data = tensor_data.numpy()

# 保存为 CSV 文件
csv_path = "C:/Users/Joseph/Desktop/raybnn/neuron_train/data/neuron_graph_tensor.csv"
np.savetxt(csv_path, numpy_data, delimiter=",")