import torch
from cnn import simpleCnn

x = torch.randn(32, 3, 224, 224)
model = simpleCnn(num_class=4)
output = model(x)
print(output.shape)

