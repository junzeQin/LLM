import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

torch.manual_seed(12046)
dataset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transforms.ToTensor())
train_set, validate_set = random_split(dataset, [50000, 10000])
test_set = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_set, batch_size=500, shuffle=True)
validate_loader = DataLoader(validate_set, batch_size=500, shuffle=True)
test_loader = DataLoader(test_set, batch_size=500, shuffle=True)

x, y = next(iter(train_loader))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass


model = MLP()

model = nn.Sequential(
    nn.Linear(784, 30, bias=False),
    nn.LayerNorm(30),
    nn.ReLU(),
    nn.Linear(30, 20, bias=False),
    nn.LayerNorm(20),
    nn.ReLU(),
    nn.Linear(20, 10)
)

eval_iters = 10


@torch.no_grad()
def _loss(model, dataloader):
    loss = []
    acc = []
    data_iter = iter(dataloader)
    for t in range(eval_iters):
        inputs, labels = next(data_iter)
        B, C, H, W = inputs.shape
        logits = model(inputs.view(B, -1))
        loss.append(F.cross_entropy(logits, labels))
        preds = torch.argmax(logits, dim=-1)
        acc.append((preds == labels).sum() / B)
    re = {
        'loss': torch.tensor(loss).mean().item(),
        'acc': torch.tensor(acc).mean().item()
    }

    return re


preds = torch.randint(4, (6,))
labels = torch.randint(4, (6,))
print(preds)
print(labels)

loss_result = _loss(model, train_loader)
print(loss_result)


def estimate_loss(model):
    re = {
        'train': _loss(model, train_loader),
        'validate': _loss(model, validate_loader),
        'test': _loss(model, test_loader)
    }

    return re


estimate_loss_result = estimate_loss(model)
print(estimate_loss_result)


def train_model(model, optimizer, epochs=10):
    result = []
    for epoch in range(epochs):
        for data in train_loader:
            inputs, labels = data
            B, C, H, W = inputs.shape
            logits = model(inputs.view(B, -1))
            loss = F.cross_entropy(logits, labels)
            result.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        states = estimate_loss(model)
        train_loss = f'{states["train"]["loss"]:.3f}'
        validate_loss = f'{states["validate"]["loss"]:.3f}'
        test_loss = f'{states["test"]["loss"]:.3f}'
        print(f'epoch:{epoch + 1} train_loss:{train_loss} validate_loss:{validate_loss} test_loss:{test_loss}')

    return result


loss = {'mlp': train_model(model, optimizer=optim.SGD(model.parameters(), lr=0.01))}
print(loss)

for i in ['mlp']:
    plt.plot(torch.tensor(loss[i]).view(-1, 10).mean(dim=-1), label=i)
plt.legend()
plt.show()
