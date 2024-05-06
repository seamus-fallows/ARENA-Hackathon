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


def generate_data_tokens(
    n_data: int,
    seq_len: int,
    concept_tokens: Int[Tensor, "token"],
    vocab_size: int = None,
    p_concept: float = 0.5,
    vocab: Int[Tensor, "token"] = None,
) -> Tuple[Tensor]:

    # Randomly choose concept tokens
    random_concept_token_indices = t.randint(
        0, concept_tokens.shape[0], (n_data, seq_len)
    )
    concept_tokens_data = concept_tokens[random_concept_token_indices]

    # Randomly choose non-concept tokens
    if vocab is None:
        mask = t.ones(vocab_size, dtype=t.bool)
        mask[concept_tokens] = 0
        available_tokens = t.nonzero(mask).squeeze()
        random_vocab_indices = t.randint(
            0, available_tokens.shape[0], (n_data, seq_len)
        )
        random_text_tokens = available_tokens[random_vocab_indices]
    else:
        random_vocab_indices = t.randint(0, vocab.shape[0], (n_data, seq_len))
        random_text_tokens = vocab[random_vocab_indices]

    # Create a mask for the concept tokens
    concept_token_probs = t.rand(n_data, seq_len, dtype=t.float)
    concept_token_mask = concept_token_probs < p_concept

    data = random_text_tokens
    data[concept_token_mask] = concept_tokens_data[concept_token_mask]

    return data, concept_token_mask


animals_sentences = [
    "The cat chased the mouse.",
    "A dog barked at the mailman.",
    "The bird sang a sweet song.",
    "A frog leaped into the pond.",
    "The duck swam in the lake.",
    "A bear hibernated in the cave.",
    "The lion roared in the savanna.",
    "A deer grazed in the meadow.",
    "The owl hooted in the night.",
    "A fox prowled in the forest.",
    "The hen laid eggs in the coop.",
    "A bat flew through the sky.",
    "The seal basked on the rock.",
    "A cow grazed in the pasture.",
    "The lamb frolicked in the field.",
    "A crab scuttled on the beach.",
    "The mole dug a tunnel underground.",
    "A hawk soared above the trees.",
    "The shrimp swam in the aquarium.",
    "A snake slithered in the grass.",
    "The goat climbed the steep hill.",
    "A moth fluttered near the light.",
    "The pig rolled in the mud.",
    "A crow perched on the fence.",
    "A wolf howled at the moon.",
    "The ram butted heads with another.",
    "The yak grazed on the mountain.",
    "The ape swung from the branch.",
    "A rat scurried in the alley.",
    "A rabbit hopped in the garden.",
]

animals_list = [
    "cat",
    "mouse",
    "dog",
    "bird",
    "frog",
    "duck",
    "bear",
    "lion",
    "deer",
    "owl",
    "fox",
    "hen",
    "bat",
    "seal",
    "cow",
    "lamb",
    "crab",
    "mole",
    "hawk",
    "shrimp",
    "snake",
    "goat",
    "moth",
    "pig",
    "crow",
    "wolf",
    "ram",
    "yak",
    "ape",
    "rat",
    "rabbit",
]


def generate_animal_data(tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    animal_tokens = tokenizer.batch_encode_plus(
        [" " + animal for animal in animals_list], return_tensors="pt"
    )["input_ids"]
    animal_tokens = animal_tokens[:, -1]
    print(animal_tokens)
    animal_tokens = animal_tokens.flatten()

    sentence_tokens = [tokenizer.encode(sentence) for sentence in animals_sentences]

    animal_mask = [
        [float(token in animal_tokens) for token in sentence]
        for sentence in sentence_tokens
    ]
    return sentence_tokens, animal_mask


# %%
# Test
if __name__ == "__main__":

    llama_token = "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"
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    n_param = 7
    tokenizer = AutoTokenizer.from_pretrained(
        f"meta-llama/Llama-2-{n_param}b-chat-hf",
        token=llama_token,
        use_fast=True,
        add_bos_token=False,
        add_prefix_space=False,
        add_special_tokens=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    vocab_size = tokenizer.vocab_size
    numeral_tokens = tokenizer.batch_encode_plus(
        [str(i) for i in range(10)], return_tensors="pt"
    )["input_ids"][:, 1].flatten()
    data_token_ids, labels = generate_data_tokens(
        n_data=10, seq_len=10, concept_tokens=numeral_tokens, vocab_size=vocab_size
    )
    print(labels)
    print(data_token_ids)

    for i in range(data_token_ids.size(0)):
        sequence = data_token_ids[i].tolist()
        decoded_sequence = tokenizer.decode(sequence)
        print(decoded_sequence)
# %%
if __name__ == "__main__":

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch as t

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    sentence_tokens, animal_mask = generate_animal_data(tokenizer)
    print("sentences with animals:")
    print("##########################")
    for sentence, mask in zip(sentence_tokens, animal_mask):
        cleaned_senence = tokenizer.decode(sentence)
        print(cleaned_senence)
    print("sentences without animals:")
    print("##########################")

    for sentence, mask in zip(sentence_tokens, animal_mask):
        cleaned_senence = [token for (token, mask) in zip(sentence, mask) if mask == 0]
        cleaned_senence = tokenizer.decode(cleaned_senence)
        print(cleaned_senence)
    print("Only animals:")
    print("##########################")
    for sentence, mask in zip(sentence_tokens, animal_mask):
        cleaned_senence = [token for (token, mask) in zip(sentence, mask) if mask == 1]
        cleaned_senence = tokenizer.decode(cleaned_senence)
        print(cleaned_senence)
