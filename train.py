import torch
import os
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # if '1' then use CUDA kernel for seq mode (much faster)

from rwkv.model import RWKV                         # pip install rwkv
from modeling_rwkv_rm import RWKV_RM
from tokenizers import Tokenizer
from utils import collate_fn

tokenizer = Tokenizer.from_file("data/20B_tokenizer.json")
model_path = "D:/weights/rwkv/RWKV-4-Pile-430M-20220808-8066.pth"
model = RWKV(model=model_path, strategy='cuda fp16')
rwkv_rm = RWKV_RM(model, base_model_trainable=True)

# put these models on the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rwkv_rm.to(device)

from datasets import load_dataset
dataset = load_dataset("yitingxie/rlhf-reward-datasets")


def train(rm, dataset, loss, optimizer, batch_size=64, epoch = 10):
    """
    Train the model on the dataset.
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in range(epoch):
        for i, batch in enumerate(dataloader):
            batch_context, batch_label = collate_fn(batch, tokenizer)
            batch_label = batch_label.to(device)
            reward = rm(batch_context)
            loss_value = loss(reward, batch_label)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Batch {i}: loss {loss_value.item()}")

def eval(rm, dataset, loss, batch_size = 64):
    """
    Evaluate the model on the dataset.
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_loss = 0
    for i, batch in enumerate(dataloader):
        batch_context, batch_label = collate_fn(batch, tokenizer)
        batch_label = batch_label.to(device)
        reward = rm(batch_context)
        loss_value = loss(reward, batch_label)
        total_loss += loss_value.item()
    return total_loss / len(dataloader)

# mse loss
loss = torch.nn.MSELoss()

# These two lines are proof of concept to overfit a very small dataset
optimizer = torch.optim.Adam(rwkv_rm.parameters(), lr=1e-3)
train(rwkv_rm, dataset["train"].select(range(100)), loss, optimizer, batch_size=16, epoch = 10)

# save the model
torch.save(rwkv_rm.reward_head.state_dict(), "rwkv_rm.reward_head.pth")

# load the model
rwkv_rm.reward_head.load_state_dict(torch.load("rwkv_rm.reward_head.pth"))

test_loss = eval(rwkv_rm, dataset["test"].select(range(20)), loss)
print(f"Test loss: {test_loss}")