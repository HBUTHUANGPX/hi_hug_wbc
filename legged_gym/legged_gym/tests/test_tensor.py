# 假設 batch_size = 5
import torch

kp_gains_scaled = torch.randn(5, 6)  # 形狀為 [5, 6]
print(kp_gains_scaled)

# 使用 repeat 來重複列
kp_gains_expanded = kp_gains_scaled.repeat(1, 2)  # 重複每列 2 次
print(kp_gains_expanded)
print(kp_gains_expanded.shape)  # 應為 [5, 12]
aa = torch.clip(kp_gains_expanded,0.1,0.3)
print(aa)
