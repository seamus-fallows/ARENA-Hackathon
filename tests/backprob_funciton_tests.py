import sys

sys.path.append("..")
from lib import backprop_functions
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW


def test_tokenize_input():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
