import torch.nn as nn

a = [nn.Linear(in_features=28*28, out_features=800, bias=True), nn.Linear(in_features=28*28, out_features=800, bias=True), nn.Linear(in_features=28*28, out_features=800, bias=True),]
a = nn.ModuleList(a)
print(a[0:1])