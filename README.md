# microDLM

The **most atomic Diffusion Language Model** in pure Python.

No PyTorch.
No NumPy.
No dependencies.
Just Python, math, and vibes.

Inspired by the minimal ML philosophy of Andrej Karpathy.

---

## What is this?

A tiny **diffusion-style language model** that learns to generate names.

Instead of predicting tokens left→right like GPT, the model:

1. Randomly **masks tokens**
2. Learns to **predict the masked tokens**
3. Generates text via **iterative unmasking**

Think of it like:

| Model        | Strategy                |
|--------------|-------------------------|
| GPT          | autoregressive generation |
| BERT         | masked prediction       |
| Diffusion LM | iterative denoising     |

---

## Features

- Pure Python autograd
- Tiny transformer
- Diffusion-style masked training
- Iterative unmasking inference
- Adam optimizer

~300 lines of code.

---

## Training

Run:

```bash
python3 micro_dlm.py
```

The script automatically downloads the dataset from the makemore names corpus.

Example training log:

```
step  123 / 1000 | loss 1.87 | t=0.42 | masked 3/8
```

---

## Inference

Generation starts from a fully masked sequence:

```
[BOS] [MASK] [MASK] ... [MASK] [BOS]
```

The model repeatedly predicts tokens and keeps the most confident ones until a name emerges.

Example output:

```
emma
olivia
noah
ethan
lara
```

---

## Why?

To show how small and understandable a diffusion language model can be.

You can read the entire model in one sitting.