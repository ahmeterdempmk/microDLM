"""
The most atomic way to train and run inference for a Diffusion Language Model in PyTorch.
same algorithm as micro_dlm.py but accelerated with tensor operations and batching.
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")


if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# tokenizer
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
MASK = len(uchars) + 1
vocab_size = len(uchars) + 2
print(f"vocab size: {vocab_size}")

# hyperparameters
n_layer = 4
n_embd = 64
block_size = 32
n_head = 4
head_dim = n_embd // n_head
batch_size = 64
num_steps = 2000
learning_rate = 3e-4

# model: Bidirectional Transformer for mask prediction
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.scale

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        q = self.wq(x).view(B, L, n_head, head_dim).transpose(1, 2) # (B, nh, L, hd)
        k = self.wk(x).view(B, L, n_head, head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, n_head, head_dim).transpose(1, 2)
        # BIDIRECTIONAL attention: no causal mask!
        attn = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.wo(out)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = MultiHeadAttention()
        self.norm2 = RMSNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x



class MaskPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.norm_in = RMSNorm(n_embd)
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device)
        x = self.tok_emb(x) + self.pos_emb(pos)
        x = self.norm_in(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

model = MaskPredictor().to(device)
print(f"num params: {sum(p.numel() for p in model.parameters())}")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


def tokenize(doc):
    return [BOS] + [uchars.index(ch) for ch in doc] + [BOS]

#forward process: mask tokens with probability t
def forward_process(x0, t):
    """x0: (B, L) clean tokens, t: scalar noise level. returns noisy tokens and mask."""
    mask_prob = torch.full_like(x0, t, dtype=torch.float)
    # Never mask BOS tokens
    mask_prob[x0 == BOS] = 0.0
    mask = torch.bernoulli(mask_prob).bool()
    # Ensure at least one token is masked per sequence
    for i in range(x0.shape[0]):
        if not mask[i].any():
            non_bos = (x0[i] != BOS).nonzero(as_tuple=True)[0]
            if len(non_bos) > 0:
                idx = non_bos[torch.randint(len(non_bos), (1,))]
                mask[i, idx] = True
    xt = x0.clone()
    xt[mask] = MASK
    return xt, mask


print("training diffusion language model...")
for step in range(num_steps):
    # Sample a batch of documents
    batch_docs = [docs[(step * batch_size + i) % len(docs)] for i in range(batch_size)]
    max_len = min(block_size, max(len(tokenize(d)) for d in batch_docs))

    # Tokenize and pad
    x0_list = []
    for doc in batch_docs:
        toks = tokenize(doc)[:max_len]
        toks = toks + [BOS] * (max_len - len(toks)) # pad with BOS
        x0_list.append(toks)
    x0 = torch.tensor(x0_list, device=device)

    # Sample noise level
    t = random.uniform(0.01, 1.0)

    # Forward process: corrupt the clean sequence
    xt, mask = forward_process(x0, t)

    # Predict all positions
    logits = model(xt) # (B, L, vocab_size)

    # Loss: CE on masked positions only (uniform weighting for stability)
    loss_all = F.cross_entropy(logits.view(-1, vocab_size), x0.view(-1), reduction='none')
    loss_all = loss_all.view(x0.shape)
    loss = (loss_all * mask.float()).sum() / mask.float().sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % 50 == 0 or step == 0:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f} | t={t:.2f}")


print("\n--- inference (diffusion-generated names) ---")
num_steps_inference = 16
temperature = 0.8 # controls sampling randomness (lower = more confident)

model.eval()
with torch.no_grad():
    for sample_idx in range(20):
        # Variable generation length matching typical name lengths
        gen_length = random.randint(3, 8)
        # Start fully masked
        tokens = torch.tensor([[BOS] + [MASK] * gen_length + [BOS]], device=device)
        L = tokens.shape[1]

        for step_i in range(num_steps_inference):
            logits = model(tokens) # (1, L, vocab_size)
            probs = F.softmax(logits[0] / temperature, dim=-1) # (L, vocab_size)

            # For each masked position, sample and record confidence
            mask_positions = (tokens[0] == MASK).nonzero(as_tuple=True)[0]
            if len(mask_positions) == 0:
                break

            # Sample predictions for all masked positions
            pred_probs = probs[mask_positions] # (n_masked, vocab_size)
            predicted = torch.multinomial(pred_probs, 1).squeeze(-1)
            confidences = pred_probs[torch.arange(len(mask_positions)), predicted]

            # Unmask all
            tokens[0, mask_positions] = predicted

            # Re-mask lowest confidence ones based on schedule
            frac_to_keep = (step_i + 1) / num_steps_inference
            n_to_keep = max(1, int(frac_to_keep * len(mask_positions)))

            if step_i < num_steps_inference - 1 and n_to_keep < len(mask_positions):
                _, sorted_idx = confidences.sort(descending=True)
                to_remask = mask_positions[sorted_idx[n_to_keep:]]
                tokens[0, to_remask] = MASK

        # Decode
        name_tokens = tokens[0, 1:-1].tolist()
        name = ''
        for tok in name_tokens:
            if tok == BOS:
                break
            if tok == MASK:
                name += '_'
            elif tok < len(uchars):
                name += uchars[tok]
        print(f"sample {sample_idx+1:2d}: {name}")
