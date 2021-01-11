import torch

device = torch.device('cpu')

x = torch.tensor([1, 0, 3, 1, 2, 0, 2, 5, 1, 4])
y = torch.tensor([1, 0, 0, 1, 2, 0, 2, 5, 0, 4])
print((x == y).float())
accuracy_val = torch.mean((x == y).float())
print(accuracy_val.item())