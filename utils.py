import torch

def pad_tokens(tokens, pad_id=-100):
    max_len = max([len(token) for token in tokens])
    for i in range(len(tokens)):
        tokens[i] = tokens[i] + [pad_id] * (max_len - len(tokens[i]))
    return torch.tensor(tokens)

def collate_fn(batch, tokenizer):
    """
    Given several rows, generate the context and the labels
    batch is a dictionary of prompt, chosen and reject, each a list of strings (batch_size)
    """
    good_examples = [i+"\n"+j for i, j in zip(batch["prompt"], batch["chosen"])]
    bad_examples = [i+"\n"+j for i, j in zip(batch["prompt"], batch["rejected"])]
    labels = [1] * len(good_examples) + [-1] * len(bad_examples)
    batch_tokens = tokenizer.encode_batch(good_examples + bad_examples)
    # batch_labels will be 1 or 0 for cross entropy loss
    batch_labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
    return batch_tokens, batch_labels