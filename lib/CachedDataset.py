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
import sys


llama_token = "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"


class CacheUtil:
    @staticmethod
    def put_cache_on_device(
        cache: Tuple[Tuple[t.Tensor]], device: str
    ) -> Tuple[Tuple[t.Tensor]]:
        """Move cache to the specified device if not already on it."""
        if cache[0][0].device != device:
            cache = tuple(
                tuple(kv.to(device) for kv in cache_layer) for cache_layer in cache
            )
        return cache

    @staticmethod
    def spip_first_n_pos_of_cache(cache: Tuple[Tuple[Tensor]], n: int) -> Tuple[Tensor]:
        """Select all but the first n seq positions of the cache."""
        return tuple(
            tuple(kv[:, :, n:, :] for kv in cache_layer) for cache_layer in cache
        )

    @staticmethod
    def add_caches_along_seq_pos(
        cache_1: Tuple[Tuple[Tensor]], cache_2: Tuple[Tuple[Tensor]]
    ) -> Tuple[Tuple[Tensor]]:
        """Add two caches along the sequence position."""
        return tuple(
            tuple(
                t.cat(
                    [kv_1, kv_2], dim=-2
                )  # Concatenating along the sequence dimension
                for kv_1, kv_2 in zip(cache_layer_1, cache_layer_2)
            )
            for cache_layer_1, cache_layer_2 in zip(cache_1, cache_2)
        )

    @staticmethod
    def process_prompt_dict_into_parts(prompt_obj):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        # Extract the lambda function for user prompts
        user_prompt_func = prompt_obj["user_prompt"]

        # Generate a sample string from the user prompt function to analyze it
        # Now including 'sentence_sample' as part of the lambda function call
        sample_prompt = (
            B_INST
            + B_SYS
            + prompt_obj["system_prompt"]
            + E_SYS
            + user_prompt_func("sentence_sample", "concept_sample", "token_sample")
            + E_INST
            + prompt_obj["ai_answer"]
        )

        # Find the positions of 'token_sample', 'concept_sample', and 'sentence_sample' in the sample prompt
        token_pos = sample_prompt.find("token_sample")
        concept_pos = sample_prompt.find("concept_sample")
        sentence_pos = sample_prompt.find("sentence_sample")

        # construct the different parts:
        part_before_sentence = sample_prompt[:sentence_pos]
        part_after_sentence_before_token = sample_prompt[
            sentence_pos + len("sentence_sample") : token_pos
        ]
        part_after_token_before_concept = sample_prompt[
            token_pos + len("token_sample") : concept_pos
        ]
        part_after_concept = (
            sample_prompt[concept_pos + len("concept_sample") :]
            + prompt_obj["ai_answer"]
        )

        return (
            part_before_sentence,
            part_after_sentence_before_token,
            part_after_token_before_concept,
            part_after_concept,
        )

    @staticmethod
    def get_cuda_memory_usage(device="cuda:0"):
        allocated = t.cuda.memory_allocated(device)
        total = t.cuda.get_device_properties(device).total_memory
        return allocated / total


class CachedDataset(Dataset):
    def __init__(
        self,
        model,
        tokenizer,
        token_list: List[List[int]],
        activation_list: List[List[float]],
        prompt_obj: Dict,
        magic_token_string: str = "magic",
        threshold: float = 0.5,
        sentence_cache_device: Optional[str] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model.to(model.device)
        self.magic_token_id = tokenizer.encode(
            magic_token_string, add_special_tokens=False
        )[-1]
        self.sentence_cache_device = sentence_cache_device
        self.threshold = threshold

        self.setup_prompt_tokens(prompt_obj)

        # Preprocess token and activation lists
        self.prepare_data(token_list, activation_list)

    def setup_prompt_tokens(self, prompt_obj):
        (
            part_before_sentence,
            part_after_sentence_before_token,
            part_after_token_before_concept,
            part_after_concept,
        ) = CacheUtil.process_prompt_dict_into_parts(prompt_obj)
        self.ids_before_sentence = self.tokenizer.encode(part_before_sentence)
        self.ids_after_sentence_before_token = self.tokenizer.encode(
            part_after_sentence_before_token, add_special_tokens=False
        )
        self.ids_after_token_before_concept = self.tokenizer.encode(
            part_after_token_before_concept, add_special_tokens=False
        )
        self.ids_after_concept = self.tokenizer.encode(
            part_after_concept, add_special_tokens=False
        )

        self.mask_before_sentence = [1] * len(self.ids_before_sentence)
        self.mask_after_sentence_before_token = [1] * len(
            self.ids_after_sentence_before_token
        )
        self.mask_after_token_before_concept = [1] * len(
            self.ids_after_token_before_concept
        )
        self.mask_after_concept = [1] * len(self.ids_after_concept)

        self.yes_label_id = self.tokenizer.encode(
            prompt_obj["yes_answer"], add_special_tokens=False
        )[-1]
        self.no_label_id = self.tokenizer.encode(
            prompt_obj["no_answer"], add_special_tokens=False
        )[-1]

    def tokens_to_padded_tokens_and_attention_mask(self, token_list: List[List[int]]):
        padded_token_list, attention_masks = [], []
        max_len = max(len(tokens) for tokens in token_list)
        for tokens in token_list:
            tokens = list(tokens)
            attention_mask = [1] * (len(tokens))
            attention_mask += [0] * (max_len - len(tokens))
            tokens += [self.tokenizer.eos_token_id] * (max_len - len(tokens))
            attention_masks.append(attention_mask)
            padded_token_list.append(tokens)
        return padded_token_list, attention_masks

    def prepare_data(
        self, token_list: List[List[int]], activation_list: List[List[float]]
    ):
        """Preprocess token and activation lists to prepare caches and other required structures."""
        # pad tokens to the same length
        self.cache_before_sentence = self.get_cache(
            t.tensor(self.ids_before_sentence).unsqueeze(0).to(self.model.device)
        )

        (
            self.sentence_caches,
            self.sentence_attentions,
            self.labels,
            self.rest_of_prompt,
            self.datapoint_to_sentence_map,
        ) = ([], [], [], [], {})

        token_list, attention_masks = self.tokens_to_padded_tokens_and_attention_mask(
            token_list
        )
        self.sentence_attention_mask_list = attention_masks

        for sentence_idx, (tokens, activations, attention_mask) in enumerate(
            zip(token_list, activation_list, attention_masks)
        ):
            sys.stdout.write(
                f"\GPU: {CacheUtil.get_cuda_memory_usage(self.model.device)*100:.2f}% full, Processing sentence {sentence_idx + 1}/{len(token_list)}\r"
            )
            sys.stdout.flush()  # Ensure the output is displayed immediately
            self.prepare_sentence_cache(tokens, attention_mask)
            self.prepare_labels_and_questions(tokens, activations, sentence_idx)

    def prepare_sentence_cache(self, tokens: List[int], attention_mask: List[int]):
        """Prepare cache for a single sentence."""
        # Convert tokens to model's input format
        sentence_ids = t.tensor(tokens, dtype=t.long).unsqueeze(0)
        ## extend the attention mask to include the system prompt
        attention_mask = (
            t.tensor(self.mask_before_sentence + attention_mask, dtype=t.long)
            .unsqueeze(0)
            .to(self.model.device)
        )

        sentence_cache = self.get_cache(
            sentence_ids.to(self.model.device),
            self.cache_before_sentence,
            attention_mask,
        )
        sentence_cache = CacheUtil.spip_first_n_pos_of_cache(
            sentence_cache, len(self.ids_before_sentence)
        )

        if self.sentence_cache_device:
            sentence_cache = CacheUtil.put_cache_on_device(
                sentence_cache, self.sentence_cache_device
            )

        self.sentence_caches.append(sentence_cache)
        self.sentence_attentions.append(attention_mask)

        return sentence_cache

    def prepare_labels_and_questions(
        self, tokens: List[int], activations: List[float], sentence_idx: int
    ):
        """Prepare labels and question end IDs for each token in a sentence."""
        for token, activation in zip(tokens, activations):
            label_id = (
                self.yes_label_id if activation > self.threshold else self.no_label_id
            )
            self.labels.append(label_id)

            self.rest_of_prompt.append(
                self.ids_after_sentence_before_token
                + [token]
                + self.ids_after_token_before_concept
                + [self.magic_token_id]
                + self.ids_after_concept
            )
            datapoint_idx = len(self.labels) - 1  # Current index in flat list
            self.datapoint_to_sentence_map[datapoint_idx] = sentence_idx

    def get_cache(
        self, ids, prev_cache=None, attention_mask=None
    ) -> Tuple[Tuple[t.Tensor]]:
        """Generate cache for given input IDs."""
        with t.no_grad():
            output = self.model(
                input_ids=ids,
                past_key_values=prev_cache,
                attention_mask=attention_mask,
                return_dict=True,
            )
        return output.past_key_values

    def __getitem__(self, idx):
        sentence_cache = self.sentence_caches[self.datapoint_to_sentence_map[idx]]
        complete_cache = CacheUtil.add_caches_along_seq_pos(
            self.cache_before_sentence, sentence_cache
        )
        question_end_id = t.tensor(self.rest_of_prompt[idx]).unsqueeze(0)
        label_id = self.labels[idx]

        complete_mask = (
            self.mask_before_sentence
            + self.sentence_attention_mask_list[self.datapoint_to_sentence_map[idx]]
            + [1] * len(self.rest_of_prompt[idx])
        )
        complete_mask = (
            t.tensor(complete_mask, dtype=t.long).unsqueeze(0).to(self.model.device)
        )
        return complete_cache, question_end_id, label_id, complete_mask

    def __len__(self):
        return len(self.labels)


class CachedDataloader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, device="cpu"):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.custom_collate_fn,
        )
        self.device = device

    def custom_collate_fn(self, batch):
        """Custom collation function to handle caching and device placement."""
        caches, question_end_ids, labels, attention_masks = zip(*batch)
        batched_caches = self.batch_caches(caches)
        batched_question_end_ids = t.cat(question_end_ids, dim=0).to(self.device)
        batched_attention_masks = t.cat(attention_masks, dim=0).to(self.device)
        batched_labels = t.tensor(labels, dtype=t.long).to(self.device)
        return (
            batched_caches,
            batched_question_end_ids,
            batched_labels,
            batched_attention_masks,
        )

    def batch_caches(self, caches):
        """Batch caches together, moving to the correct device if necessary."""
        # Flatten and combine caches across the batch

        batched_caches = tuple(
            tuple(
                t.cat([cache_layer[i] for cache_layer in cache], dim=0)
                for i in range(len(cache[0]))
            )
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
    prompt = dict(
        system_prompt=""" Your task is to assess if a given token (word) from some text represents a specified concept. Provide a rating based on this assessment:
                            If the token represents the concept, respond with 'Rating: 1'.
                            If the token does not represent the concept, respond with 'Rating: 0'.
                            Focus solely on the token and use the other text for context only. Be confident.
                            """,
        user_prompt=lambda sentence, concept, token: [
            "In general: does the token ",
            token,
            " represent the concept ",
            concept,
            " ?",
        ],
        ai_answer="Rating: ",
        yes_answer="1",
        no_answer="0",
    )
    string_list = [
        "Lore ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
        "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
        "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam.",
    ] * 23
    token_list = [tokenizer.encode(string)[1:] for string in string_list]
    activation_list = [t.rand(len(tokens)) for tokens in token_list]

# %%
if __name__ == "__main__":
    example_dataset = CachedDataset(
        model, tokenizer, token_list, activation_list, prompt
    )
    dataloader = CachedDataloader(
        example_dataset, batch_size=50, shuffle=True, device=device
    )
# %%
if __name__ == "__main__":
    probs_on_label = np.array([])
    for sentence_cache, question_end_ids, label, masks in tqdm(dataloader):
        output = model(
            question_end_ids,
            past_key_values=sentence_cache,
            return_dict=True,
            attention_mask=masks,
        )
        output_probs = F.softmax(output.logits, dim=-1)
        del output
        del sentence_cache
        del question_end_ids
        del masks
        t.cuda.empty_cache()
        prob_on_label = output_probs[0, -1, label].detach().cpu().numpy()
        probs_on_label = np.append(probs_on_label, prob_on_label)

    print(probs_on_label)
    print(np.mean(probs_on_label))
# %%
if __name__ == "__main__":
    import torch

    def estimate_object_gpu_memory_usage(obj, verbose=False):
        total_memory = 0

        def estimate_tensor_memory(tensor):
            return tensor.element_size() * tensor.nelement()

        def recurse_attributes(obj):
            nonlocal total_memory
            if torch.is_tensor(obj) and obj.is_cuda:
                memory = estimate_tensor_memory(obj)
                total_memory += memory
                return memory
            elif hasattr(obj, "__dict__"):
                return sum(
                    recurse_attributes(value) for key, value in obj.__dict__.items()
                )
            elif isinstance(obj, (list, tuple)):
                return sum(recurse_attributes(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(recurse_attributes(value) for key, value in obj.items())
            return 0

        memory = recurse_attributes(obj)

        if verbose:
            print(f"Estimated GPU memory usage: {memory / (1024 ** 2):.2f} MB")

        return memory

    estimate_object_gpu_memory_usage(example_dataset, verbose=True)
    estimate_object_gpu_memory_usage(
        example_dataset.cache_before_sentence, verbose=True
    )
# %%
import inspect
import re


def analyze_function_v2(user_func):
    # Extract the source code of the user function
    src = inspect.getsource(user_func)
    print(src)

    # Find the format string inside the function
    format_string_match = re.search(r'f"(.+)"', src)
    if not format_string_match:
        raise ValueError("No format string found in function.")
    format_string = format_string_match.group(1)

    print(format_string)
    print(format_string_match)


user_prompt = lambda sentence, concept, token: [
    "In general: does the token ",
    token,
    " represent the concept ",
    concept,
    " ?",
]  # Use the updated analyze function
analyze_function_v2(user_prompt)
src = inspect.getsource(user_prompt)
# %%
src
# %%
from inspect import signature, Parameter


def validate_prompt(func):
    # Check if the function has exactly three arguments
    sig = signature(func)
    if len(sig.parameters) != 3:
        raise AssertionError("Function must have exactly three arguments.")

    # Check if the arguments are named correctly and in the correct order
    param_names = list(sig.parameters.keys())
    expected_names = ["sentence", "concept", "token"]
    if param_names != expected_names:
        raise AssertionError(
            f"Function arguments must be named {', '.join(expected_names)} in that order."
        )

    # Check if the list returned alternates between strings and arguments
    # This is a bit trickier because we need to evaluate the function's return value
    # For demonstration, we'll use mock values for sentence, concept, and token
    test_values = {"sentence": -1, "concept": -2, "token": -3}
    result = func(**test_values)
    if not isinstance(result, list) or len(result) < 3:
        raise AssertionError(
            "Function must return a list that alternates between strings and arguments."
        )

    for i, item in enumerate(result):
        if i % 2 == 0 and not isinstance(item, str):
            raise AssertionError("List must alternate between strings and arguments.")
        if i % 2 == 1 and not isinstance(item, int):
            raise AssertionError("List must alternate between strings and arguments.")

    # Check if the concept appears at all in the return list
    if (
        "concept" not in [p.name for p in sig.parameters.values()]
        or test_values["concept"] not in result
    ):
        raise AssertionError("The 'concept' must appear in the return list.")

    return "Function passes all checks."


def insert_replace(original_list, key, insert_list):
    # Create a new list to hold the result
    result = []
    # Iterate through each element in the original list
    for element in original_list:
        # If the element is key, extend the result list with the insert_list
        if element == key:
            if insert_list is None:
                raise ValueError("argument to be inserted is was not provided")
            result.extend(insert_list)
        # Otherwise, just append the element to the result list
        else:
            result.append(element)
    return result


def make_list_to_function(string_list, tokenizer):
    token_list = []
    for item in string_list:
        if isinstance(item, int):
            token_list.append(item)
        elif isinstance(item, str):
            token_list.extend(tokenizer.encode(item, add_special_tokens=False))
        else:
            raise ValueError("List must alternate between strings and arguments.")

    def toeknized_prompt_funciton(sentence=None, concept=None, token=None):
        return_list = insert_replace(token_list, -1, sentence)
        return_list = insert_replace(return_list, -2, concept)
        return_list = insert_replace(return_list, -3, token)

        return return_list

    return toeknized_prompt_funciton


def tokenize_prompt_funciton(prompt_function, tokenizer):

    validate_prompt(prompt_function)

    key_values = {"sentence": -1, "concept": -2, "token": -3}
    string_list = prompt_function(**key_values)

    universal_part = tokenizer.encode(string_list[0], add_special_tokens=False)

    fist_position_of_token_or_concept = min(
        [string_list.index(k) for k in [-2, -3] if k in string_list]
    )

    sentence_dependent_part = string_list[1:fist_position_of_token_or_concept]
    token_dependent_part = string_list[fist_position_of_token_or_concept:]

    return (
        universal_part,
        make_list_to_function(sentence_dependent_part, tokenizer),
        make_list_to_function(token_dependent_part, tokenizer),
    )


def add_syntax_to_prompt_func(prompt_obj):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    # Extract the lambda function for user prompts
    user_prompt_func = prompt_obj["user_prompt"]

    # Generate a sample string from the user prompt function to analyze it
    # Now including 'sentence_sample' as part of the lambda function call

    def syntaxed_prompt_func(sentence, concept, token):
        unsyntacized_sentence = user_prompt_func(sentence, concept, token)
        unsyntacized_sentence[0] = (
            B_INST
            + B_SYS
            + prompt_obj["system_prompt"]
            + E_SYS
            + unsyntacized_sentence[0]
        )
        unsyntacized_sentence[-1] = (
            unsyntacized_sentence[-1] + E_INST + prompt_obj["ai_answer"]
        )
        return unsyntacized_sentence

    return syntaxed_prompt_func


prompt = dict(
    system_prompt=""" Your task is to assess if a given token (word) from some text represents a specified concept. Provide a rating based on this assessment:
                            If the token represents the concept, respond with 'Rating: 1'.
                            If the token does not represent the concept, respond with 'Rating: 0'.
                            Focus solely on the token and use the other text for context only. Be confident.
                            """,
    user_prompt=lambda sentence, concept, token: [
        "In the sentence ",
        sentence,
        " does the token ",
        token,
        " represent the concept ",
        concept,
        " ?",
    ],
    ai_answer="Rating: ",
    yes_answer="1",
    no_answer="0",
)

syntaxed_prompt_func = add_syntax_to_prompt_func(prompt)
universal_part, sentence_dependent_part, token_dependent_part = (
    tokenize_prompt_funciton(syntaxed_prompt_func, tokenizer)
)
# %%

sentence_tokens = tokenizer.encode("The dog had 3 heads", add_special_tokens=False)
concept_tokens = tokenizer.encode("animal", add_special_tokens=False)
token_tokens = tokenizer.encode("dog", add_special_tokens=False)


total_tokens = (
    universal_part
    + sentence_dependent_part(sentence=sentence_tokens)
    + token_dependent_part(token=token_tokens, concept=concept_tokens)
)
print(tokenizer.decode(total_tokens))
# %%
