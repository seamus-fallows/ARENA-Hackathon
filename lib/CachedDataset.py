# %%

from torch.nn import functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch as t
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple
import numpy as np
from torch import Tensor
from tqdm.notebook import tqdm
from jaxtyping import Int, Float
from typing import List, Dict



llama_token = "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"


class CacheUtil:
    @staticmethod
    def put_cache_on_device(cache: Tuple[Tuple[t.Tensor]], device: str) -> Tuple[Tuple[t.Tensor]]:
        """Move cache to the specified device if not already on it."""
        if cache[0][0].device != device:
            cache = tuple(
                tuple(kv.to(device) for kv in cache_layer) for cache_layer in cache
            )
        return cache

class CachedDataset(Dataset):
    def __init__(
        self,
        model,
        tokenizer,
        token_list: List[List[int]],
        activation_list: List[List[float]],
        magic_token_string: str = "magic",
        threshold: float = 0.5,
        sentence_cache_device: Optional[str] = None,
        system_prompt: List[str] = ["""Your task is to assess if a given token (word) from some text represents a specified concept. Provide a rating based on this assessment:
                            If the token represents the concept, respond with 'Rating: 1'.
                            If the token does not represent the concept, respond with 'Rating: 0'.
                            Focus solely on the token and use the other text for context only. Be confident.""",
                            ""
                            ],
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model.to(model.device)
        self.magic_token_id = tokenizer.encode(magic_token_string, add_special_tokens=False)[-1]
        self.yes_label_id = tokenizer.encode("1", add_special_tokens=False)[-1]
        self.no_label_id = tokenizer.encode("0", add_special_tokens=False)[-1]
        self.system_prompt_ids = self.encode_system_prompt(system_prompt)
        self.sentence_cache_device = sentence_cache_device
        self.threshold = threshold

        # Preprocess token and activation lists
        self.prepare_data(token_list, activation_list)
    
    def setup_prompt_tokens(self):


    def tokens_to_padded_tokens_and_attention_mask(self, token_list: List[List[int]]):
        padded_token_list, attention_masks = [], []
        max_len = max(len(tokens) for tokens in token_list)
        for tokens in token_list:
            tokens = list(tokens)
            attention_mask = [1] * (len(tokens) + self.system_prompt_ids.size(1))
            attention_mask += [0] * (max_len - len(tokens))
            tokens += [self.tokenizer.eos_token_id] * (max_len - len(tokens))
            attention_masks.append(attention_mask)
            padded_token_list.append(tokens)
        return padded_token_list, attention_masks

    def prepare_data(self, token_list: List[List[int]], activation_list: List[List[float]]):
        """Preprocess token and activation lists to prepare caches and other required structures."""
        # pad tokens to the same length
        
        system_prompt_cache = self.get_cache(self.system_prompt_ids.to(self.model.device))
        self.sentence_caches, self.labels, self.question_end_ids_list, self.datapoint_to_sentence_map = [], [], [], {}

        token_list, attention_masks = self.tokens_to_padded_tokens_and_attention_mask(token_list)
        for sentence_idx, (tokens, activations, attention_mask) in enumerate(zip(token_list, activation_list, attention_masks)):
            sentence_cache = self.prepare_sentence_cache(tokens, attention_mask,system_prompt_cache)
            self.prepare_labels_and_questions(tokens, activations, sentence_idx)

    def prepare_sentence_cache(self, tokens: List[int],attention_mask: List[int], system_prompt_cache):
        """Prepare cache for a single sentence."""
        # Convert tokens to model's input format
        sentence_ids = t.tensor(tokens, dtype=t.long).unsqueeze(0)
        ## extend the attention mask to include the system prompt
        attention_mask = t.tensor(attention_mask, dtype=t.long).unsqueeze(0)

        sentence_cache = self.get_cache(sentence_ids.to(self.model.device), system_prompt_cache, attention_mask.to(self.model.device))
        if self.sentence_cache_device:
            sentence_cache = CacheUtil.put_cache_on_device(sentence_cache, self.sentence_cache_device)
        self.sentence_caches.append(sentence_cache)
        return sentence_cache

    def prepare_labels_and_questions(self, tokens: List[int], activations: List[float], sentence_idx: int):
        """Prepare labels and question end IDs for each token in a sentence."""
        for token_idx, (token, activation) in enumerate(zip(tokens, activations)):
            label_id = self.yes_label_id if activation > self.threshold else self.no_label_id
            self.labels.append(label_id)
            question_end_id = self.question_end_to_ids(token)
            self.question_end_ids_list.append(question_end_id)
            datapoint_idx = len(self.labels) - 1  # Current index in flat list
            self.datapoint_to_sentence_map[datapoint_idx] = sentence_idx

    def encode_system_prompt(self, prompt: str) -> t.Tensor:
        """Encode system prompt to tensor."""
        encoded_prompt = self.tokenizer.encode(prompt, add_special_tokens=False)
        return t.tensor(encoded_prompt, dtype=t.long).unsqueeze(0)

    def get_cache(self, ids, prev_cache=None, attention_mask=None) -> Tuple[Tuple[t.Tensor]]:
        """Generate cache for given input IDs."""
        with t.no_grad():
            output = self.model(input_ids=ids, past_key_values=prev_cache, attention_mask=attention_mask, return_dict=True)
        return output.past_key_values


    def question_end_to_ids(self, token_id: int) -> t.Tensor:
        """Encode the end of a question for a given token."""
        question_end_ids = [self.magic_token_id, token_id] + self.tokenizer.encode("Rating: ", add_special_tokens=False)
        return t.tensor(question_end_ids, dtype=t.long).unsqueeze(0)

    def __getitem__(self, idx):
        sentence_cache = self.sentence_caches[self.datapoint_to_sentence_map[idx]]
        question_end_id = self.question_end_ids_list[idx]
        label_id = self.labels[idx]
        return sentence_cache, question_end_id, label_id

    def __len__(self):
        return len(self.labels)

class CachedDataloader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, device='cpu'):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.custom_collate_fn)
        self.device = device

    def custom_collate_fn(self, batch):
        """Custom collation function to handle caching and device placement."""
        caches, question_end_ids, labels = zip(*batch)
        batched_caches = self.batch_caches(caches)
        batched_question_end_ids = t.cat(question_end_ids, dim=0).to(self.device)
        batched_labels = t.tensor(labels, dtype=t.long).to(self.device)
        return batched_caches, batched_question_end_ids, batched_labels

    def batch_caches(self, caches):
        """Batch caches together, moving to the correct device if necessary."""
        # Flatten and combine caches across the batch
        batched_caches = tuple(
            tuple(t.cat([cache_layer[i] for cache_layer in cache], dim=0) for i in range(len(cache[0])))
            for cache in zip(*caches)
        )
        # Ensure the batched cache is on the correct device
        batched_caches = CacheUtil.put_cache_on_device(batched_caches, self.device)
        return batched_caches

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
    token_list = [tokenizer.encode(string)[1:] for string in string_list]
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
    for sentence_cache, question_end_ids, label in tqdm(dataloader):
        output = model(
            question_end_ids, past_key_values=sentence_cache, return_dict=True
        )
        output_probs = F.softmax(output.logits, dim=-1)
        del output
        del sentence_cache
        prob_on_label = output_probs[0, -1, label].detach().cpu().numpy()
        probs_on_label = np.append(probs_on_label, prob_on_label)

    print(probs_on_label)
    print(np.mean(probs_on_label))

# %%
cache, sentence_ids, label = example_dataset[0]
# %%
print(tokenizer.decode(sentence_ids[0]))
# %%
rating_tokens = tokenizer.encode("Rating: ", add_special_tokens=False)
print(tokenizer.decode(rating_tokens))
print(tokenizer.decode(rating_tokens[1:]))
# %%
prompt = dict(
    system_prompt = "Yor task is to rate blah blah blah",
    user_prompt = lambda sentence, concept, token: f"In this sentence: {sentence}:Does the token {token} represent the concept {concept}?"
    ai_answer = "Rating: "
)
# %%
def process_prompt(prompt_obj):
    # Extract the lambda function for user prompts
    user_prompt_func = prompt_obj['user_prompt']
    
    # Generate a sample string from the user prompt function to analyze it

    sample_prompt = prompt_obj['system_prompt'] + user_prompt_func('concept_sample', 'token_sample') + prompt_obj['ai_answer']
    
    # Find the positions of 'token_sample' and 'concept_sample' in the sample prompt
    token_pos = sample_prompt.find('token_sample')
    concept_pos = sample_prompt.find('concept_sample')
    
    # Define the first part: up until the first mentioning of "token" or "concept"
    first_mention_pos = min(token_pos, concept_pos) if token_pos != -1 and concept_pos != -1 else max(token_pos, concept_pos)
    first_part = sample_prompt[:first_mention_pos]
    
    # Define the second function: takes a token and returns the string up until the mention of "concept"
    def until_concept_func(token):
        nonlocal sample_prompt, concept_pos
        return sample_prompt[:concept_pos].replace('token_sample', token)
    
    # Define the third function: takes a concept and returns the rest of the string
    def after_concept_func(concept):
        nonlocal sample_prompt, concept_pos
        return sample_prompt[concept_pos:].replace('concept_sample', concept)
    
    return first_part, until_concept_func, after_concept_func

# Example usage:
prompt = dict(
    system_prompt="Your task is to rate blah blah blah",
    user_prompt=lambda concept, token: f"Does the token {token} represent the concept {concept}?",
    ai_answer="Rating: "
)

first_part, until_concept, after_concept = process_prompt(prompt)

# Testing the functions
print("First part:", first_part)
print("Until concept (with token 'example_token'):", until_concept('example_token'))
print("After concept (with concept 'example_concept'):", after_concept('example_concept'))

# %%
