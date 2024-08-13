import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from cnn import simple_cnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = simple_cnn(4).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))  # 加载模型参数

# 设置模型为评估模式
model.eval()

# 定义预处理步骤（根据您的模型需要进行相应的预处理）
transform = transforms.Compose([
    # 将数据裁剪为224*224
    transforms.Resize([224, 224]),
    # 将数据转换为Tensor张量，0-1的像素值
    transforms.ToTensor(),
    # 对数据进行标准化
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载并预处理输入图像
input_image = Image.open('input_image.jpg')  # 替换为您的输入图像
input_tensor = transform(input_image).unsqueeze(0)  # 添加批次维度

# 运行推理
with torch.no_grad():
    output = model(input_tensor)

# 处理模型输出（根据您的模型输出进行相应的处理）
# 例如，获取预测结果或后处理操作

# 输出预测结果或进行后续处理