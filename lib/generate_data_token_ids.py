import torch as t
import os
import random
from transformers import AutoTokenizer

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

class TextDataGenerator:
    def __init__(self, sentences=None, words_to_label=None, tokenizer=None, file_path=None):
        if file_path is not None:
            sentence_path = os.path.join(file_path, "sentences.txt")
            word_path = os.path.join(file_path, "label_tokens.txt")
            self.sentences = self.load_sentences(sentence_path)
            self.words_to_label = self.load_words(word_path)
        else:
            self.sentences = sentences
            self.words_to_label = words_to_label

        self.tokenizer = tokenizer
        self.appends_eos = len(tokenizer.encode("a")) == 2

    def load_sentences(self, sentence_path):
        with open(sentence_path, "r") as file:
            sentences = [sentence.strip() for sentence in file.readlines()]
        return sentences

    def load_words(self, word_path):
        with open(word_path, "r") as file:
            words = [int(word.strip()) for word in file.readlines()]
        return words

    def generate_data(self):
        sentence_tokens = [self.tokenizer.encode(sentence) for sentence in self.sentences]
        if self.appends_eos:
            sentence_tokens = [sentence[1:] for sentence in sentence_tokens]

        word_mask = [
            [float(token in self.words_to_label) for token in sentence]
            for sentence in sentence_tokens
        ]
        return sentence_tokens, word_mask

    def print_sentences_with_words(self):
        sentence_tokens, word_mask = self.generate_data()
        print("Sentences with words:")
        print("##########################")
        random_indices = random.sample(range(len(sentence_tokens)), min(10, len(sentence_tokens)))
        for index in random_indices:
            sentence = sentence_tokens[index]
            cleaned_sentence = self.tokenizer.decode(sentence)
            print(cleaned_sentence)

    def print_sentences_without_words(self):
        sentence_tokens, word_mask = self.generate_data()
        print("Sentences without words:")
        print("##########################")
        random_indices = random.sample(range(len(sentence_tokens)), min(10, len(sentence_tokens)))
        for index in random_indices:
            sentence = sentence_tokens[index]
            mask = word_mask[index]
            cleaned_sentence = [token for (token, mask) in zip(sentence, mask) if mask == 0]
            cleaned_sentence = self.tokenizer.decode(cleaned_sentence)
            print(cleaned_sentence)

    def print_only_words(self):
        sentence_tokens, word_mask = self.generate_data()
        print("Only words:")
        print("##########################")
        random_indices = random.sample(range(len(sentence_tokens)), min(10, len(sentence_tokens)))
        for index in random_indices:
            sentence = sentence_tokens[index]
            mask = word_mask[index]
            cleaned_sentence = [token for (token, mask) in zip(sentence, mask) if mask == 1]
            cleaned_sentence = self.tokenizer.decode(cleaned_sentence)
            print(cleaned_sentence)

