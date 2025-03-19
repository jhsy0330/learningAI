import torch

x = torch.arange(4.0)
print(x)
x.requires_grad_(True)
y = 2 * torch.dot(x,x)
print(x.grad)
y = y * 2
print(y)
y.backward()
print(x.grad)