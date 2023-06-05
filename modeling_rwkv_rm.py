import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class RWKV_RM(nn.Module):
    def __init__(self, rwkv, base_model_trainable=False):
        super().__init__()
        self.rwkv = rwkv
        # self.reward_head = nn.Linear(rwkv.args.n_embd * rwkv.args.n_layer, 1, bias=False)
        self.reward_head = nn.Sequential(
            nn.Linear(rwkv.args.n_embd * rwkv.args.n_layer, 4096),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(2048, 1)
        )
        if not base_model_trainable:
            for param in self.rwkv.parameters():
                param.requires_grad = False

    def forward(self, tokens):
        # The tokens are [batch_size, seq_len], padded with -100
        # but the RWKV model can't run in batch, therefore we need to run it one by one
        results = []
        for i in tqdm(tokens):
            tok = i.ids
            state = self.rwkv(tok, None)
            # the state is [n_layer * 5, n_embd], we need to extract the n_layer * n_embd targeting only the pp values
            state = torch.cat([state[1][4+i] for i in range(0, len(state[1]), 5)]).view(-1, self.rwkv.args.n_embd * self.rwkv.args.n_layer)
            # put state back in dtype=torch.float32
            state = state.to(torch.float32)
            reward = self.reward_head(state)
            results.append(reward)
        
        return torch.cat(results)

    def load_head_state(self, state_dict):
        self.reward_head.load_state_dict(state_dict)