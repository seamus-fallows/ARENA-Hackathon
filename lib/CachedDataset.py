# %%
import torch as t
from torch.nn import functional as F
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import einops
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Optional, Tuple, Any
from torch import Tensor
from dataclasses import dataclass, field
from tqdm.notebook import tqdm
from jaxtyping import Int, Float
from typing import List, Dict
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
import datetime

llama_token = "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"


class CachedDataset(Dataset):
    def __init__(
        self,
        model,
        tokenizer,
        token_list,
        activation_list,
        magic_token_string: str = "magic",
        threshhold: float = 0.5,
    ):
        super().__init__()
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.B_SYS, self.E_SYS = "<<SYS>>", "<</SYS>>"

        self.magic_token_ids = tokenizer.encode(magic_token_string)[1]
        self.tokenizer = tokenizer
        self.model = model

        yes_label = tokenizer.encode("1")[-1]
        no_label = tokenizer.encode("0")[-1]

        systtem_prompt = """ Your task is to assess if a given token (word) from a sentence represents a specified concept. Provide a rating based on this assessment:
                            If the token represents the concept, respond with "Rating: 1".
                            If the token does not represent the concept, respond with "Rating: 0".
                            Focus solely on the token and use the sentence for context only. Be confident.
                        """
        systemprompt_ids = self.systemprompt_to_ids(tokenizer, systtem_prompt, delete_first_token=False)
        system_promt_cache = self.get_cache(systemprompt_ids.to(device))

        max_len = max([len(tokens) for tokens in token_list])
        for tokens in token_list:
            tokens += [tokenizer.eos_token_id] * (max_len - len(tokens))

        self.sentence_caches = []

        for sentence in tqdm(token_list):
            sentence_ids = self.sentence_to_ids(sentence)
            sentence_cache = self.get_cache(
                sentence_ids.to(device), prev_cache=system_promt_cache
            )
            # print(sentence_cache[0][0].shape)
            # make a deep copy of the cache
            # sentence_cache = [[layer.clone() for layer in sub_cache] for sub_cache in sentence_cache]
            self.sentence_caches.append(sentence_cache)

        self.datapoint_counter = 0
        self.sentence_counter = 0

        self.datapoint_number_to_sentence_number = dict()

        self.label_list = []
        self.question_end_ids_list = []
        for sentence, activations in tqdm(zip(token_list, activation_list)):
            for token, activation in zip(sentence, activations):
                label = yes_label if activation > threshhold else no_label
                self.label_list.append(label)
                question_end_ids = self.question_end_to_ids(token)
                self.question_end_ids_list.append(question_end_ids)
                self.datapoint_number_to_sentence_number[
                    self.datapoint_counter
                ] = self.sentence_counter
                self.datapoint_counter += 1
            self.sentence_counter += 1

    def systemprompt_to_ids(self, tokenizer, systtem_prompt):
        prompt = self.B_INST + self.B_SYS + systtem_prompt + self.E_SYS + "Sentence: "
        ids = t.tensor(tokenizer.encode(prompt)).unsqueeze(0)
        return ids

    def get_cache(self, ids, prev_cache=None):
        with t.no_grad():
            if prev_cache is None:
                output = self.model(ids, return_dict=True)
            else:
                output = self.model(ids, past_key_values=prev_cache, return_dict=True)
        return output.past_key_values

    def sentence_to_ids(self, sentence, delete_first_token=True):
        post_text = "Concept:"
        post_text_ids = self.tokenizer.encode(post_text)
        if delete_first_token:
            post_text_ids = post_text_ids[1:]
        ids = t.tensor(sentence + post_text_ids).unsqueeze(0)
        return ids

    def question_end_to_ids(self, question_token_ids):
        text_1 = " Token:"
        ids_1 = tokenizer.encode(text_1)[1:]
        text_2 = self.E_INST + "The rating is "
        ids_2 = self.tokenizer.encode(text_2)[1:]
        ids = [self.magic_token_ids] + ids_1 + [question_token_ids] + ids_2
        return t.tensor(ids).unsqueeze(0)

    def __getitem__(self, idx):
        return (
            self.sentence_caches[self.datapoint_number_to_sentence_number[idx]],
            self.question_end_ids_list[idx],
            self.label_list[idx],
        )

    def __len__(self):
        return self.datapoint_counter


class CachedDataloader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, device):
        super().__init__(
            dataset, batch_size, shuffle, collate_fn=self.custom_collate_fn
        )
        self.device = device

    def custom_collate_fn(self, batch):
        # Unzip the batch
        caches, sentence_ids, labels = zip(*batch)

        batched_sentence_ids = t.cat(sentence_ids, dim=0).to(self.device)
        batched_labels = t.tensor(labels).to(self.device)

        batched_caches = tuple(
            [
                tuple(
                    [
                        t.cat([layer[i] for layer in cache], dim=0).to(self.device)
                        for i in range(len(cache[0]))
                    ]
                )
                for cache in zip(*caches)
            ]
        )

        return batched_caches, batched_sentence_ids, batched_labels

# %%
if __name__ == "__main__":
    llama_token = "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"
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    t.cuda.empty_cache()
    n_param = 7
    model = AutoModelForCausalLM.from_pretrained(
        f"meta-llama/Llama-2-{n_param}b-chat-hf", use_auth_token=llama_token
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        ignore_mismatched_sizes=True,
        use_auth_token=llama_token,
    )
    tokenizer.pad_token = tokenizer.eos_token
# %%
if __name__ == "__main__":
    string_list = [
        "Lore ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
        "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
        "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam.",
    ]
    token_list = [tokenizer.encode(string) for string in string_list]
    activation_list = [t.rand(len(tokens)) for tokens in token_list]

# %%
if __name__ == "__main__":
    example_dataset = CachedDataset(model, tokenizer, token_list, activation_list)
    dataloader = CachedDataloader(
        example_dataset, batch_size=50, shuffle=True, device=device
    )

# %%
if __name__ == "__main__":
    probs_on_label = np.array([])
    for sentence_cache, question_end_ids, label in dataloader:
        output = model(
            question_end_ids, past_key_values=sentence_cache, return_dict=True
        )
        output_probs = F.softmax(output.logits, dim=-1)
        prob_on_label = output_probs[0, -1, label].detach().cpu().numpy()
        probs_on_label = np.append(probs_on_label, prob_on_label)

    print(probs_on_label)

# %%
llama_token = "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"
7
model = AutoModelForCausalLM.from_pretrained(
        f"meta-llama/Llama-2-{n_param}b-chat-hf", use_auth_token=llama_token
    ).to(device)
tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        padding=True,
        use_auth_token=llama_token,
        pad_token = tokenizer.eos_token,
    )

# Your input text sequences
text_sequences = ["short sequence", "much longer sequence with many more words. Indeed."]
cached_prompt = "This is the initial prompt."
cached_prompt_tokens = tokenizer(cached_prompt, return_tensors="pt")['input_ids'].to(device)


