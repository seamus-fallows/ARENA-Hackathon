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
MODEL_NAME = 'distilgpt2'
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# Base prompt
prompt = "She is "
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Perform initial forward pass to generate cache
outputs = model(input_ids, use_cache=True)
cache = outputs.past_key_values

# cache is a tuple of tuples of tensors, so need to loop through to repeat along bacth dimension of tensors.
batched_cache = tuple(
    tuple(einops.repeat(tensor, 'batch ... -> (n batch) ...', n=n_batch) for tensor in layer_cache)
    for layer_cache in cache
)

append_list = ['going', 'running', 'under', 'apple']

# Tokens to append
append_ids = t.tensor([tokenizer.encode(append, add_special_tokens=False) for append in append_list]).unsqueeze(1).to(device)

# Perform a single forward pass with the batch of new tokens
outputs = model(append_ids, batched_cache)
token_ids = outputs.logits[:,:,-1,:].squeeze().argmax(dim=-1)
words = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
print(words)






# %%
