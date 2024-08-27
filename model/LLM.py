# 提供与操作系统交互的功能
# 数学库
import math
import os

# 数据分析和处理工具
# 用于发送 HTTP 请求的库
import requests
# 用于文本编码的库
import tiktoken
# 深度学习框架
import torch
# 神经网络模块
from torch import nn
# 激活函数、损失函数
from torch.nn import functional as F

# Hyperparameters
batch_size = 12
context_length = 128
d_model = 512
num_blocks = 12
num_heads = 8
learning_rate = 1e-4
dropout = 0.1
max_iters = 15000
eval_interval = 50
eval_iters = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

# Load training data
# if not os.path.exists('data/sales_textbook.txt'):
#     url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
#     with open('data/sales_textbook.txt', 'w') as f:
#         f.write(requests.get(url).text)

with open('data/scifi.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Using TikToken (Same as GPT3) to tokenize the source text
encoding = tiktoken.get_encoding("o200k_base")
tokenized_text = encoding.encode(text)
max_token_value = max(tokenized_text) + 1  # the maximum value of the tokenized numbers
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)  # put tokenized text into tensor

# Split train and validation
split_idx = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:split_idx]
validation_data = tokenized_text[split_idx:]


class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model // num_heads, bias=False)
        self.Wk = nn.Linear(d_model, d_model // num_heads, bias=False)
        self.Wv = nn.Linear(d_model, d_model // num_heads, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        weights = (q @ k.transpose(-2, -1)) / math.sqrt(d_model // num_heads)
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        output = weights @ v

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Attention() for _ in range(num_heads)])
        self.projection_layer = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        head_outputs = torch.cat(head_outputs, dim=-1)
        out = self.dropout(self.projection_layer(head_outputs))

        return out


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention()
        self.ffn = FeedForwardNetwork()

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self, max_token_value=200018):
        super().__init__()
        self.token_embedding_lookup_table = nn.Embedding(max_token_value, d_model)
        self.transformer_blocks = nn.Sequential(*(
                [TransformerBlock() for _ in range(num_blocks)] +
                [nn.LayerNorm(d_model)]
        ))
        self.model_out_linear_layer = nn.Linear(d_model, max_token_value)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        position_encoding_lookup_table = torch.zeros(context_length, d_model, device=device)
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
        position_embedding = position_encoding_lookup_table[:T, :].to(device)
        x = self.token_embedding_lookup_table(idx) + position_embedding
        x = self.transformer_blocks(x)
        logits = self.model_out_linear_layer(x)

        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens=100):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -context_length:]
            logits, loss = self.forward(idx_crop)
            logits_last_timestep = logits[:, -1, :]
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            idx_next = torch.multinomial(input=probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def get_batch(split: str):
    data = train_data if split == 'train' else validation_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'validation']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out


if __name__ == '__main__':
    model = TransformerLanguageModel()
    model = model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
    tracked_losses = list()
    for step in range(max_iters):
        if step % eval_iters == 0 or step == max_iters - 1:
            losses = estimate_loss()
            tracked_losses.append(losses)
            print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
                  round(losses['validation'].item(), 3))

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), 'model-scifi.pt')

    model.eval()
    start = 'The scifi model'
    start_ids = encoding.encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    y = model.generate(x, max_new_tokens=100)
    print('---------------')
    print(encoding.decode(y[0].tolist()))
    print('---------------')
