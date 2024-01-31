# %%
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM 
import einops
from typing import Union, Optional, Tuple, Any
from torch import Tensor
from jaxtyping import Int, Float
from typing import List, Dict
from torch import Tensor

def generate_data_tokens(n_data: int, seq_len: int, concept_tokens: Int[Tensor, "token"], vocab_size: int, p_concept: float = 0.5) -> Tuple[Tensor]:
    # Create a boolean mask for the entire vocabulary
    mask = t.ones(vocab_size, dtype=t.bool)
    mask[concept_tokens] = 0

    # Find the indices where mask is True
    available_tokens = t.nonzero(mask).squeeze()

    # Randomly choose concept tokens
    random_concept_token_indices = t.randint(0, concept_tokens.shape[0], (n_data, seq_len))
    concept_tokens_data = concept_tokens[random_concept_token_indices]

    # Randomly choose non-concept tokens
    random_vocab_indices = t.randint(0, available_tokens.shape[0], (n_data, seq_len))
    random_text_tokens = available_tokens[random_vocab_indices]

    # Create a mask for the concept tokens
    concept_token_probs = t.rand(n_data, seq_len, dtype=t.float)
    concept_token_mask = (concept_token_probs < p_concept)

    data = random_text_tokens
    data[concept_token_mask] = concept_tokens_data[concept_token_mask]

    return data, concept_token_mask

#%%
# Test
if __name__ == "__main__":

    llama_token = "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"
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    n_param = 7
    tokenizer = AutoTokenizer.from_pretrained(
                f"meta-llama/Llama-2-{n_param}b-chat-hf", token=llama_token, use_fast=True, add_bos_token=False, add_prefix_space=False, add_special_tokens=False
            )
    tokenizer.pad_token = tokenizer.eos_token

    vocab_size = tokenizer.vocab_size
    numeral_tokens = tokenizer.batch_encode_plus([str(i) for i in range(10)], return_tensors='pt')['input_ids'][:,1].flatten()  
    data_token_ids, labels = generate_data_tokens(n_data = 10, seq_len = 10, concept_tokens = numeral_tokens, vocab_size=vocab_size)
    print(labels)
    print(data_token_ids)

    for i in range(data_token_ids.size(0)):
        sequence = data_token_ids[i].tolist() 
        decoded_sequence = tokenizer.decode(sequence) 
        print(decoded_sequence)
# %%
