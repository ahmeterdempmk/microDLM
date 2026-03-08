"""
The most atomic way to train and run inference for a Diffusion Language Model in pure, dependency-free Python.
inspired by @karpathy's microgpt.py

special thanks to @alperiox for his valuable thoughts :D
@ahmeterdempmk
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) 


if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars)      # token id for a special Beginning of Sequence (BOS) token
MASK = len(uchars) + 1 # token id for a special MASK token (the diffusion ingredient)
vocab_size = len(uchars) + 2 # total number of unique tokens, +2 for BOS and MASK
print(f"vocab size: {vocab_size}")


class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data + 1e-10), (self,), (1/(self.data + 1e-10),))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


n_layer = 1     # depth of the transformer
n_embd = 16     # width of the network (embedding dimension)
block_size = 32  # maximum sequence length
n_head = 4      # number of attention heads
head_dim = n_embd // n_head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# define the model: a bidirectional transformer that predicts masked tokens
# Unlike GPT's causal (left-to-right) attention, diffusion uses FULL attention. every token sees every other token
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def mask_predictor(token_ids):
    """The core diffusion model: takes a partially masked sequence, predicts all positions."""
    L = len(token_ids)
    # embed all tokens and positions
    xs = []
    for pos, tok in enumerate(token_ids):
        tok_emb = state_dict['wte'][tok]
        pos_emb = state_dict['wpe'][pos]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm(x)
        xs.append(x)

    for li in range(n_layer):
        # Compute Q, K, V for ALL positions at once
        qs = [linear(rmsnorm(xs[i]), state_dict[f'layer{li}.attn_wq']) for i in range(L)]
        ks = [linear(rmsnorm(xs[i]), state_dict[f'layer{li}.attn_wk']) for i in range(L)]
        vs = [linear(rmsnorm(xs[i]), state_dict[f'layer{li}.attn_wv']) for i in range(L)]

        new_xs = []
        for i in range(L):
            x_residual = xs[i]
            x_attn = []
            for h in range(n_head):
                hs = h * head_dim
                q_h = qs[i][hs:hs+head_dim]
                # BIDIRECTIONAL: attend to ALL positions (no causal mask!)
                attn_logits = [
                    sum(q_h[j] * ks[t][hs+j] for j in range(head_dim)) / head_dim**0.5
                    for t in range(L)
                ]
                attn_weights = softmax(attn_logits)
                head_out = [
                    sum(attn_weights[t] * vs[t][hs+j] for t in range(L))
                    for j in range(head_dim)
                ]
                x_attn.extend(head_out)
            x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
            x = [a + b for a, b in zip(x, x_residual)]
            # MLP block
            x_residual = x
            x = rmsnorm(x)
            x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
            x = [xi.relu() for xi in x]
            x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
            x = [a + b for a, b in zip(x, x_residual)]
            new_xs.append(x)
        xs = new_xs

    # Output logits at all positions
    return [linear(xs[i], state_dict['lm_head']) for i in range(L)]

# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m_buf = [0.0] * len(params)
v_buf = [0.0] * len(params)


num_steps = 5000
print("training diffusion language model...")
for step in range(num_steps):

    # Take single document, tokenize it, surround with BOS
    doc = docs[step % len(docs)]
    clean_tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    L = min(block_size, len(clean_tokens))
    clean_tokens = clean_tokens[:L]

    # Forward process: sample noise level t, mask each token with probability t
    t = random.uniform(0.01, 1.0)
    noisy_tokens = []
    masked_positions = []
    for i, tok in enumerate(clean_tokens):
        if tok == BOS:
            noisy_tokens.append(tok) # never mask BOS
        elif random.random() < t:
            noisy_tokens.append(MASK)
            masked_positions.append(i)
        else:
            noisy_tokens.append(tok)

    # If nothing got masked, force-mask at least one position
    if not masked_positions:
        non_bos = [i for i, tok in enumerate(clean_tokens) if tok != BOS]
        if non_bos:
            i = random.choice(non_bos)
            noisy_tokens[i] = MASK
            masked_positions.append(i)

    # Forward pass: predict all positions from partially masked input
    all_logits = mask_predictor(noisy_tokens)

    # cross-entropy only on masked positions (uniform weighting for stability)
    losses = []
    for i in masked_positions:
        probs = softmax(all_logits[i])
        loss_i = -probs[clean_tokens[i]].log()
        losses.append(loss_i)
    loss = (1 / len(masked_positions)) * sum(losses)

    # Backward
    loss.backward()

    # Adam optimizer update
    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
        m_hat = m_buf[i] / (1 - beta1 ** (step + 1))
        v_hat = v_buf[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f} | t={t:.2f} | masked {len(masked_positions)}/{L}", end='\r')


# iterative unmasking 
print("\n\n--- inference (diffusion-generated names) ---")
num_steps_inference = 16
temperature = 0.8

for sample_idx in range(20):
    # Variable generation length matching typical name lengths
    gen_length = random.randint(3, 8)
    # Start fully masked: [BOS, MASK, MASK, ..., MASK, BOS]
    tokens = [BOS] + [MASK] * gen_length + [BOS]
    L = len(tokens)

    for step in range(num_steps_inference):
        # Predict all positions
        all_logits = mask_predictor(tokens)

        # For each masked position, sample a token and record confidence
        predictions = [] # (position, predicted_token, confidence)
        for i in range(L):
            if tokens[i] != MASK:
                continue
            scaled_logits = [val / temperature for val in all_logits[i]]
            probs = softmax(scaled_logits)
            prob_data = [p.data for p in probs]
            predicted = random.choices(range(vocab_size), weights=prob_data)[0]
            confidence = prob_data[predicted]
            predictions.append((i, predicted, confidence))

        if not predictions:
            break

        # Unmask all positions with their predictions
        for pos, tok, _ in predictions:
            tokens[pos] = tok

        # Determine how many to keep unmasked based on schedule
        # Linear schedule: at step s of S, keep fraction (s+1)/S of positions unmasked
        frac_to_keep = (step + 1) / num_steps_inference
        n_to_keep = max(1, int(frac_to_keep * len(predictions)))

        if step < num_steps_inference - 1:
            # re-mask the least confident predictions
            predictions.sort(key=lambda x: x[2], reverse=True)
            to_remask = predictions[n_to_keep:]
            for pos, _, _ in to_remask:
                tokens[pos] = MASK


    name_tokens = tokens[1:-1]
    name = ''
    for tok in name_tokens:
        if tok == BOS:
            break
        if tok == MASK:
            name += '_'
        elif tok < len(uchars):
            name += uchars[tok]
    print(f"sample {sample_idx+1:2d}: {name}")
