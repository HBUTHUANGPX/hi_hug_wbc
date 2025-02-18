import torch
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower

x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8,9]).view(3, 3)
print(x)
b = torch.roll(x, 1,dims=0)
print(b)
c = torch_rand_float(
            -1.0,
            1.0,
            (5, 1)
            ,device="cuda"
        ).squeeze(1)

print(c)
random_tensor = torch.randint(0, 5, (10,1)).squeeze(1)
print(random_tensor)
fa = random_tensor==0
print(fa)
d= torch.logical_or(fa,fa)
print(d)


