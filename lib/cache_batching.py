#%%
import einops
from dataclasses import dataclass
import torch as t
from torch import Tensor
import torch.nn as nn
import numpy as np
import math
from tqdm.notebook import tqdm
from typing import Tuple, List, Optional, Dict, Callable
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from jaxtyping import Float, Int
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

n_batch = 4
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2", use_fast=True)

# Base prompt
prompt = "She is "
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Perform initial forward pass to generate cache
outputs = model(input_ids, use_cache=True)
cache = outputs.past_key_values

# cache is a tuple of tuples of tensors, so need to loop through to repeat along bacth dimension of tensors.
batched_cache = []
for layer_cache in cache:
    expanded_layer_cache = []
    for tensor in layer_cache:
        expanded_tensor = einops.repeat(tensor, 'batch ... -> (n batch) ...', n=n_batch)
        expanded_layer_cache.append(expanded_tensor)
    batched_cache.append(tuple(expanded_layer_cache))

# Convert new_cache into a tuple of tuples
batched_cache = tuple(batched_cache)

append_list = ['going', 'running', 'under', 'apple']

# Tokens to append
append_ids = t.tensor([[tokenizer.encode(append, add_special_tokens=False)] for append in append_list]).to(device)

# Perform a single forward pass with the batch of new tokens
# Note: Repeat the cache for each new token in the batch
outputs = model(append_ids, batched_cache)
token_ids = outputs.logits[:,:,-1,:].squeeze().argmax(dim=-1)
words = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
print(words)






# %%
