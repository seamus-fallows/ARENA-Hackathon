import transformer_interaction
from transformers import AutoTokenizer
import torch

token = "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"


class Concept_rater:
    def __init__(self):
        self.model = transformer_interaction.get_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            ignore_mismatched_sizes=True,
            use_auth_token=token,
        )

        self.system_prompt = "Your task is to assess if a given token (word) from a sentence represents a specified concept. Provide a rating based on this assessment:If the token represents the concept, respond with 'Rating: 1'.If the token does not represent the concept, respond with 'Rating: 0'.Focus solely on the token and use the sentence for context only. Be confident."
        self.batch_tokens = None
        self.batch_labels = None

        self.system_prompt_cache = None
        self.batch_cache = None

        self.permutation = None

    def create_permutation(self):
        n, m = self.batch_tokens.shape

        tensor = torch.zeros(n, m)

        for i in range(n):
            tensor[i] = torch.randperm(m) + 1

        self.permutation = tensor

    def load_batch(self, batch_tokens, batch_labels):
        self.batch_tokens = batch_tokens
        self.batch_labels = batch_labels

        self.batch_cache = None
        self.system_prompt_cache = None

        self.create_permutation()

    def get_systemprompt_tokens(self):
        B_INST, E_INST = "[INST]", "[/INST]The rating is "
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        prompt = B_INST + B_SYS + self.system_prompt + E_SYS
        tokens = (
            torch.tensor(self.tokenizer.encode(prompt))
            .unsqueeze(0)
            .to(self.model.device)
        )
        return tokens

    def create_system_prompt_cache(self):
        base_prompt_tokens = self.get_systemprompt_tokens()
        if self.system_prompt_cache is None:
            self.system_prompt_cache = self.model.forward(
                base_prompt_tokens, use_cache=True
            ).past_key_values
        return

    def create_batch_cache(self):
        if self.batch_cache is None:
            self.batch_cache = self.model.forward(
                self.batch_tokens,
                past_key_values=self.system_prompt_cache,
                use_cache=True,
            ).past_key_values
        return

    def get_rating(self, i: int):
        assert i < self.batch_tokens.shape[0]
        assert self.batch_cache is not None
        assert self.system_prompt_cache is not None
