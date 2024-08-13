import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from cnn import simpleCnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transformer = transforms.Compose([
    # 将数据裁剪为224*224
    transforms.Resize([224, 224]),
    # 将数据转换为Tensor张量，0-1的像素值
    transforms.ToTensor(),
    # 对数据进行标准化
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transformer = transforms.Compose([
    # 将数据裁剪为224*224
    transforms.Resize([224, 224]),
    # 将数据转换为Tensor张量，0-1的像素值
    transforms.ToTensor(),
    # 对数据进行标准化
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载训练集和测试集
train_dataset = datasets.ImageFolder(root=os.path.join(r"data\COVID_19_Radiography_Dataset", "train"),  # 拼接路径 找到训练集
                                     transform=train_transformer)  # 训练集做图像变换

test_dataset = datasets.ImageFolder(root=os.path.join(r"data\COVID_19_Radiography_Dataset", "test"),
                                    transform=test_transformer)
# 查看数据集大小
print("Training data shape:", len(train_dataset))
print("Testing data shape:", len(test_dataset))

"""
num_workers 数据加载多线程，为0代表不打开
shuffle为True代表打乱加载数据
"""
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)


def train(model, train_loader, criterion, optimizer, num_epochs):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # 训练时可看到对应的epoch和batch的进度
        for inputs, labels in tqdm(train_loader, desc=f"epoch:{epoch + 1}/{num_epochs}", unit="batch"):
            # 将数据传到设备上
            inputs, labels = inputs.to(device), labels.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            # loss的计算
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 用loss乘以批次大小，得到该批次的loss
            running_loss += loss.item() * inputs.size(0)
        # 总损失除总数据集大小，得到平均loss
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch: {epoch + 1}/{num_epochs}, Train_Loss: {epoch_loss:.4f}")
        accuracy = evaluate(model, test_loader, criterion)
        if accuracy > best_acc:
            best_acc = accuracy
            save_model(model, save_path)
            print(f"Best Accuracy: {best_acc:.2f}%")


def evaluate(model, test_loader, criterion):
    # 指定模型为验证模式
    model.eval()
    # 初始的测试loss为0
    test_loss = 0.0
    # 正确样本数量为0
    correct = 0
    # 总样本数量为0
    total = 0
    # 在评估模式下不需要计算梯度
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
            # 将数据都送到设备里面
            inputs, labels = inputs.to(device), labels.to(device)
            # 将数据送到模型内
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            # 获取模型预测的最大值
            _, predicted = torch.max(outputs, 1)
            # 计算总样本的数量
            total = total + labels.size(0)
            # 正确样本数累加
            correct = correct + (predicted == labels).sum().item()
    # 计算平均loss
    avg_loss = test_loss / len(test_loader.dataset)
    # 计算准确率
    accuracy = 100.0 * correct / total
    print(f"Test_Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return accuracy


def save_model(model, path):
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    num_epochs = 10
    learning_rate = 0.001
    num_class = 8
    save_path = "best_model.pth"
    # 对模型进行实例化，并送入gpu或cpu中
    model = simpleCnn(num_class).to(device)
    # 指定损失函数为交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 指定优化器为adam
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 使用训练集训练
    train(model, train_loader, criterion, optimizer, num_epochs)
    # 使用测试集进行测试
    evaluate(model, test_loader, criterion)
