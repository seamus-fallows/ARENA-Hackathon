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
import random
import os
import os


def get_experiment_file_path(experiment_file: str, number: int) -> str:
    current_dir = os.path.dirname(os.path.abspath(experiment_file))
    name_of_current_experiment = os.path.splitext(os.path.basename(experiment_file))[0]
    data_saving_folder = os.path.join(current_dir, "..", "data", "experiment_results")

    # Create folder for the experiment if it does not exist
    experiment_folder = os.path.join(data_saving_folder, name_of_current_experiment)
    os.makedirs(experiment_folder, exist_ok=True)

    # Create a subfolder for the experiment number
    experiment_number_folder = os.path.join(experiment_folder, f"experiment_{number}")
    experiment_new = not os.path.exists(experiment_number_folder)
    os.makedirs(experiment_number_folder, exist_ok=True)

    return experiment_number_folder, experiment_new


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


class TextDataGenerator:
    def __init__(self, sentences=None, words_to_label=None, tokenizer=None, file_path=None):
        if file_path is not None:
            sentence_path = os.path.join(file_path, "sentences.txt")
            word_path = os.path.join(file_path, "labled_words.txt")
            self.sentences = self.load_sentences(sentence_path)
            self.words_to_label = self.load_words(word_path)
        else:
            self.sentences = sentences
            self.words_to_label = words_to_label

        self.tokenizer = tokenizer
        self.appends_eos = len(tokenizer.encode("a")) == 2
        self._check_word_tokenization()

    def load_sentences(self, sentence_path):
        with open(sentence_path, "r") as file:
            sentences = [sentence.strip() for sentence in file.readlines()]
        return sentences

    def load_words(self, word_path):
        with open(word_path, "r") as file:
            words = [word.strip() for word in file.readlines()]
        return words

    def _check_word_tokenization(self):
        word_tokens = [self.tokenizer.encode(word) for word in self.words_to_label]
        word_tokens_spaced = [
            self.tokenizer.encode(" " + word) for word in self.words_to_label
        ]
        if self.appends_eos:
            word_tokens = [word[1:] for word in word_tokens]
            word_tokens_spaced = [word[1:] for word in word_tokens_spaced]
        self.word_tokens = []
        for word, word_spaced in zip(word_tokens, word_tokens_spaced):
            if len(word) != 1 and len(word_spaced) != 1:
                raise ValueError(
                    f"Word tokenization failed for word {self.tokenizer.decode(word)}"
                )
            if len(word) == 1:
                self.word_tokens.append(word[0])
            if len(word_spaced) == 1:
                self.word_tokens.append(word_spaced[0])

    def generate_data(self):
        sentence_tokens = [
            self.tokenizer.encode(sentence) for sentence in self.sentences
        ]
        if self.appends_eos:
            sentence_tokens = [sentence[1:] for sentence in sentence_tokens]

        word_mask = [
            [float(token in self.word_tokens) for token in sentence]
            for sentence in sentence_tokens
        ]
        return sentence_tokens, word_mask

    def print_sentences_with_words(self):
        sentence_tokens, word_mask = self.generate_data()
        print("Sentences with words:")
        print("##########################")
        random_indices = random.sample(
            range(len(sentence_tokens)), min(10, len(sentence_tokens))
        )
        for index in random_indices:
            sentence = sentence_tokens[index]
            cleaned_sentence = self.tokenizer.decode(sentence)
            print(cleaned_sentence)

    def print_sentences_without_words(self):
        sentence_tokens, word_mask = self.generate_data()
        print("Sentences without words:")
        print("##########################")
        random_indices = random.sample(
            range(len(sentence_tokens)), min(10, len(sentence_tokens))
        )
        for index in random_indices:
            sentence = sentence_tokens[index]
            mask = word_mask[index]
            cleaned_sentence = [
                token for (token, mask) in zip(sentence, mask) if mask == 0
            ]
            cleaned_sentence = self.tokenizer.decode(cleaned_sentence)
            print(cleaned_sentence)

    def print_only_words(self):
        sentence_tokens, word_mask = self.generate_data()
        print("Only words:")
        print("##########################")
        random_indices = random.sample(
            range(len(sentence_tokens)), min(10, len(sentence_tokens))
        )
        for index in random_indices:
            sentence = sentence_tokens[index]
            mask = word_mask[index]
            cleaned_sentence = [
                token for (token, mask) in zip(sentence, mask) if mask == 1
            ]
            cleaned_sentence = self.tokenizer.decode(cleaned_sentence)
            print(cleaned_sentence)


# %%
if __name__ == "__main__":
    file_path = "../data/text_data/animals"
    sentence_path = os.path.join(file_path, "sentences.txt")
    word_path = os.path.join(file_path, "labled_words.txt")

    sentences = [sentence.strip() for sentence in open(sentence_path, "r").readlines()]
    animals_list = [animal.strip() for animal in open(word_path, "r").readlines()]

    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
    animal_data_generator = TextDataGenerator(sentences, animals_list, tokenizer)
# %%
# Test
if __name__ == "__main__":

    animal_data_generator.print_sentences_with_words()
    print("\n\n")
    animal_data_generator.print_sentences_without_words()
    print("\n\n")
    animal_data_generator.print_only_words()
    print("\n\n")


# %%
if __name__ == "__main__":
    for token in animal_data_generator.word_tokens:
        print(tokenizer.decode([token]))

# %%
