# 用于操作系统相关的功能，如文件路径处理
import os
# 处理和分析数据
import pandas as pd
# 用于发送 HTTP 请求，下载数据
import requests
# 用于文本编码
import tiktoken
# 用于深度学习，尤其是 PyTorch 库
import torch
from torch import nn

# 这行代码判断是否有可用的 GPU（CUDA），如果有则使用 CUDA，否则使用 CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 检查文件 sales_textbook.txt 是否存在，如果不存在，则从指定 URL 下载文本文件并保存到本地
if not os.path.exists('data/sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
    with open('data/sales_textbook.txt', 'w') as f:
        f.write(requests.get(url).text)

# 以 UTF-8 编码读取文件内容，并将其存储在变量 text 中
with open('data/sales_textbook.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 使用 tiktoken 对文本进行编码，将其转换为一系列的整数标记（tokens）
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)
# 将标记转换为 PyTorch 张量，并指定数据类型和设备（CPU 或 GPU）
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)  # put tokenized text into tensor

# 计算分割索引 split_idx，这里将数据的 90% 用于训练，10% 用于验证
split_idx = int(len(tokenized_text) * 0.9)
# 使用切片操作从开始到 split_idx 位置获取训练数据
train_data = tokenized_text[:split_idx]
# 使用切片操作从 split_idx 到结束获取验证数据
validation_data = tokenized_text[split_idx:]

# 每个训练样本将包含 16 个连续的标记
context_length = 16
# 每个训练批次将包含 4 个样本
batch_size = 4

data = train_data
# 使用 torch.randint 生成 batch_size 个随机整数，这些整数范围在 0 到 len(data) - context_length 之间。
# 这样可以确保在提取上下文时不超出数据的边界
idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))

"""
使用列表推导式遍历 idxs 中的每个索引 idx
对于每个索引，提取从 idx 开始的 context_length 个标记，形成一个输入样本
使用 torch.stack 将所有输入样本堆叠成一个张量 x_batch
"""
x_batch = torch.stack([data[idx:idx + context_length] for idx in idxs])

"""
同样使用列表推导式遍历 idxs
对于每个索引 idx，提取从 idx + 1 开始的 context_length 个标记，形成目标样本
使用 torch.stack 将所有目标样本堆叠成一个张量 y_batch
"""
y_batch = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs])

# 每个标记将被映射到一个 64 维的嵌入向量。这种高维表示可以捕捉标记之间的复杂语义关系。
d_model = 64
"""
计算编码后标记的最大值加1
标记索引从 0 开始：在计算机中，索引通常是从 0 开始的。因此，如果你的最大标记值是 max_token_count，那么实际的标记索引范围是从 0 到 max_token_count，共计 max_token_count + 1 个索引
避免索引超出范围：如果不加 1，模型在处理某些标记时可能会因为索引超出范围而导致错误
"""
max_token_count = max(tokenized_text) + 1
"""
nn.Embedding 是一个 PyTorch 中的层，用于创建一个查找表，将标记的索引映射到对应的嵌入向量
max_token_count 是之前计算出的标记数量，表示词汇表的大小。这个值确保嵌入层能够处理所有可能的标记
当输入一个标记的索引时，嵌入层会返回对应的 64 维向量
"""
token_embedding_lookup_table = nn.Embedding(max_token_count, d_model)
x_batch_embedding = token_embedding_lookup_table(x_batch)
y_batch_embedding = token_embedding_lookup_table(y_batch)

import math

"""
用于设置位置嵌入（position embedding），结合了正弦和余弦函数，符合原始 Transformer 论文中的方法
"""
# 创建一个大小为 (context_length, d_model) 的零张量，用于存储位置嵌入
position_encoding_lookup_table = torch.zeros(context_length, d_model)
# 生成一个从 0 到 context_length - 1 的张量，并通过 unsqueeze(1) 将其形状改变为 (context_length, 1)，以便后续计算
position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
# 计算位置编码中的分母项。这个分母是根据模型的维度 d_model 来调整的，确保不同维度的频率适当
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
# 对于偶数索引的维度，使用正弦函数填充位置嵌入，这种方式可以为不同的维度提供不同的周期性信号
position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
# 对于奇数索引的维度，使用余弦函数
position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
# 使用 unsqueeze(0) 增加一个维度，然后通过 expand 方法复制位置嵌入，使其形状变为 (batch_size, context_length, d_model)，以便与输入嵌入相加
position_encoding_lookup_table = position_encoding_lookup_table.unsqueeze(0).expand(batch_size, -1, -1)
"""
将位置嵌入与输入嵌入（x_batch_embedding 和 y_batch_embedding）相加，以确保模型能够利用位置信息
这是因为 Transformer 模型本身不具备处理序列顺序的能力，位置嵌入提供了序列中每个标记的位置信息
"""
x = x_batch_embedding + position_encoding_lookup_table
y = y_batch_embedding + position_encoding_lookup_table

"""
用于定义线性层并计算查询（Q）、键（K）和值（V），是实现自注意力机制的关键步骤
Wq, Wk, Wv 是三个线性变换层，分别用于生成查询、键和值
"""
Wq = nn.Linear(d_model, d_model)
Wk = nn.Linear(d_model, d_model)
Wv = nn.Linear(d_model, d_model)
"""
Q 是通过输入嵌入 x 经过线性变换 Wq 计算得到的查询矩阵
K 是通过输入嵌入 x 经过线性变换 Wk 计算得到的键矩阵
V 是通过输入嵌入 x 经过线性变换 Wv 计算得到的值矩阵
"""
Q = Wq(x)
K = Wk(x)
V = Wv(x)

"""
实现了多头自注意力机制的关键步骤，包括查询（Q）、键（K）、值（V）的重塑和注意力得分的计算
"""
# 定义多头注意力中的头数为 4
num_heads = 4
"""
重塑查询、键和值
将 Q、K 和 V 重塑为形状 (batch_size, context_length, num_heads, d_model // num_heads)，将每个头的维度分离
使用 permute 调整维度顺序，将形状更改为 (batch_size, num_heads, context_length, d_model // num_heads)，以便后续计算
"""
Q = Q.reshape(batch_size, context_length, num_heads, d_model // num_heads).permute(0, 2, 1, 3)
K = K.reshape(batch_size, context_length, num_heads, d_model // num_heads).permute(0, 2, 1, 3)
V = V.reshape(batch_size, context_length, num_heads, d_model // num_heads).permute(0, 2, 1, 3)
"""
计算注意力得分
通过查询矩阵 Q 和键矩阵 K 的转置（最后两个维度交换）进行矩阵乘法
将结果除以 math.sqrt(d_model // num_heads)，以防止随着维度增加而导致的数值过大，从而保持稳定性
"""
output = (Q @ K.transpose(-2, -1)) / math.sqrt(d_model // num_heads)
"""
创建和应用掩码
创建一个上三角掩码，防止模型在计算当前标记的注意力时看到未来的标记。diagonal=1 表示不包括主对角线。
使用 masked_fill 将掩码位置的得分设置为负无穷，确保这些位置在 softmax 计算时不会被考虑
"""
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
output = output.masked_fill(mask, float('-inf'))

from torch.nn import functional as F

"""
计算注意力权重
使用 softmax 函数计算注意力得分，将 output 转换为注意力权重。dim=-1 表示在最后一个维度上进行归一化
"""
attention_score = F.softmax(output, dim=-1)
"""
计算加权值
通过将注意力权重 attention_score 与值矩阵 V 相乘，生成加权后的输出 A
"""
A = attention_score @ V
"""
重塑输出
将 A 的维度顺序调整为 (batch_size, context_length, d_model)
然后重塑为 (batch_size, context_length * num_heads, d_model)，以便与后续线性层兼容
"""
A = A.transpose(1, 2).reshape(batch_size, -1, d_model)
"""
重塑输出
定义一个线性层 Wo，将加权输出 A 通过这个层进行转换，得到最终的 output
"""
Wo = nn.Linear(d_model, d_model)
output = Wo(A)
"""
残差连接
将线性变换后的输出与输入 x 进行相加，形成残差连接。这有助于缓解梯度消失问题，并促进信息流动
"""
output = output + x
"""
应用层归一化
使用层归一化对 output 进行归一化处理，以提高训练的稳定性和加速收敛
"""
layer_norm = nn.LayerNorm(d_model)
layer_norm_output = layer_norm(output)
"""
前馈神经网络（Feedforward Neural Network）
"""
"""
线性变换：将输入 layer_norm_output 通过一个线性层进行变换
这个层将输入维度 d_model 映射到更高的维度 d_model * 4，以增加模型的表达能力
"""
output = nn.Linear(d_model, d_model * 4)(layer_norm_output)
"""
激活函数：对线性变换的输出应用ReLU（Rectified Linear Unit）激活函数
ReLU将负值归零，保留正值，增加非线性，使模型可以学习更复杂的函数
"""
output = nn.ReLU()(output)
"""
再次线性变换：将经过ReLU激活后的输出再次通过一个线性层，将其维度从 d_model * 4 变回 d_model
这一步骤的目的是将数据压缩回原始维度
"""
output = nn.Linear(d_model * 4, d_model)(output)
"""
残差连接：将线性层的输出与 layer_norm_output 相加，形成残差连接
这种方法有助于保持原始输入的信息，有效缓解梯度消失问题，并促进信息流动
"""
output = output + layer_norm_output

"""
模型的最后阶段，将经过处理的输出转换为实际的类别概率
"""
"""
线性变换：
nn.Linear(d_model, max_token_count) 创建一个线性层，该层的输入维度是 d_model，输出维度是 max_token_count
output 作为输入经过这个线性层，意味着每个输入向量会被映射到 max_token_count 维度，这通常对应于模型要预测的词汇表大小
输出的含义：
经过这个线性变换后，output 的每个元素代表了对应于词汇表中每个标记的未归一化得分（logits）
"""
output = nn.Linear(d_model, max_token_count)(output)
"""
F.softmax(output, dim=-1) 对 output 应用 softmax 函数，将未归一化的得分转换为概率分布
dim=-1 表示在最后一个维度上进行归一化，这样每个样本的所有得分会被转换为一个和为 1 的概率分布
"""
logits = F.softmax(output, dim=-1)

if __name__ == '__main__':
    print(output.shape)
    dataFrame = pd.DataFrame(output[0].detach().numpy())
    print(dataFrame.shape)
