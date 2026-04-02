import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))

# 测试张量计算
x = torch.randn(100, 100).cuda()
print("Test tensor:", x.mean())