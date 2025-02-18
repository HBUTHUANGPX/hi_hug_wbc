import torch

# 创建两个 n*1 的 Tensor
n = 5
tensor1 = torch.rand(n, 1)  # 随机生成 n*1 的 Tensor
tensor2 = torch.rand(n, 1)  # 随机生成 n*1 的 Tensor

# 方法 1: 使用 torch.mul
result1 = torch.mul(tensor1, tensor2)

# 方法 2: 使用 * 运算符
result2 = tensor1 * tensor2

# 验证结果是否一致
print("Tensor 1:")
print(tensor1)
print("\nTensor 2:")
print(tensor2)
print("\nResult using torch.mul:")
print(result1)
print("\nResult using * operator:")
print(result2)
print("\nAre results equal?", torch.equal(result1, result2))
