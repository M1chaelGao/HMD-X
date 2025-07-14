import torch

# 替换成你任意一个大型 .pt 文件的真实路径
file_path = '/home/4T-2/gyf/ProcessedHMDdata/CMU/test/207.pt'

# 加载数据
data_list = torch.load(file_path)

# 检查列表中第一个动作序列（字典）的所有键
print("Keys in the first sequence dictionary:")
print(data_list[0].keys())