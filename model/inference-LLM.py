import tiktoken
import torch

from LLM import TransformerLanguageModel as Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed(TORCH_SEED)

encoding = tiktoken.get_encoding("cl100k_base")

model = Model()
model.load_state_dict(torch.load('model-scifi.pt', weights_only=False))
model.eval()
model.to(device)

start = '奥特曼出生在一个小村庄'
start_ids = encoding.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

with torch.no_grad():
    y = model.generate(x, max_new_tokens=500)
    print('---------------')
    print(encoding.decode(y[0].tolist()))
    print('---------------')
