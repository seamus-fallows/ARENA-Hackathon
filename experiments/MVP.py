# %%
import torch as t
from torch.nn import functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
import matplotlib.pyplot as plt
import einops
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Optional, Tuple, Any
from torch import Tensor
from dataclasses import dataclass


t.manual_seed(0)
np.random.seed(0)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False
# Check for GPU availability
device = t.device("cuda" if t.cuda.is_available() else "cpu")
# %%

#create a dataloader class
class Dataset(t.utils.data.Dataset):
    def __init__(self, datapath: str, tokenizer):
    def __getitem__(self, idx):
        pass
    def __len__(self):
        pass
    def plot_item(self, idx):
        pass
    def tokenize_input(self, input_text: Union[str, list[str]], magic_word: str):
        if isinstance(input_text, str):
            input_text = [input_text]
        tokens = self.tokenizer.batch_encode_plus(
            input_text, padding=True, return_tensors="pt"
        ).input_ids
        tokens = t.tensor(tokens)
        magic_ids = self.tokenizer.encode(magic_word)[0]
        magic_token_pos = t.where(tokens == magic_ids)

        return tokens, magic_token_pos

@dataclass   
class Config():
    batch_size: int = 4
    steps: int = 100
    lr: float = 1e-2
    intitialization_std: float = 1
    loss_coeffs: dict = {"accuracy": 1, "kl": 1, "entropy": 1}

@dataclass
class Logs():
    def __init__(self, losses: dict[str, float[Tensor, "1"]], top_tokens: dict[int, dict[str, Any]], specified_tokens: dict[int, dict[str, Any]], config: Config, model_name: str, dataset_name: str, run_date: str):
        self.losses = losses
        self.top_tokens = top_tokens
        self.config = config    
        self.model_name = model_name    
        self.dataset_name = dataset_name    
        self.run_date = run_date 
        self.specified_tokens = specified_tokens

    def save(self, path: str) -> None:
        pass

    def plot_losses(run_data:  dict[str,Any]) -> None:
        pass

    def plot_top_tokens(run_data:  dict[str,Any], n_plotted_tokens = 10) -> None:
        pass

    def plot_loss_tradeoff(run_data:  dict[str,Any], n_plotted_tokens = 10) -> None:
        pass

class Training():
    def __init__(self, config: Config, model, tokenizer):
        self.config = config
        self.device = model.device
        self.model = model
        self.vocab_size = model.config.vocab_size # Need to check this is the same for all models
        self.tokenizer = tokenizer
        self.intialise_random_token_vector()
        self.optimizer = AdamW(self.magic_token_vector, lr=config.lr)
        self.loss_coeffs = config.loss_coeffs

    def intialise_random_token_vector(self) -> None:
        """
        Creates a random vector of length vocab_size, and set it as the magic vecotr
        """
        magic_token_vector = t.empty(self.vocab_size, device=device).normal_(mean=0, std=self.config.intitialization_std)
        magic_token_vector = t.nn.Parameter(magic_token_vector, requires_grad=True)
        self.magic_token_vector = magic_token_vector

    def create_modified_embeddings(self, tokens: int[Tensor, "batch seq_len"], magic_token_pos:  int[Tensor, "2 n_magic_tokens"]) -> float[Tensor, "batch seq_len d_model"]:
        """
        embeds the tokens, creates the embedding of the magic token, and puts it at all places in magic_token_pos
        """

        tokens = tokens.clone().detach().to(device)
        inputs_embeds = self.model.transformer.wte.weight[tokens]
        embedding_matrix = self.model.transformer.wte.weight
        magic_token_embed = einops.einsum(
            embedding_matrix,
            self.magic_token_vector,
            " d_vocab d_model, d_vocab -> d_model ",
        )
        if magic_token_pos != None:
            x_tens, y_tens = magic_token_pos
            for x, y in zip(x_tens, y_tens):
                inputs_embeds[x, y] = magic_token_embed
        return inputs_embeds


    def calculate_losses(self, output_logits: float[Tensor, "batch seq_len d_vocab"], magic_token_vector: float[Tensor, "d_vocab"], magic_token_pos: int[Tensor, "2 n_magic_tokens"], target_tokens: int[Tensor, "batch"]) -> dict[str, float[Tensor, "1"]]:
        """
        calculate all the different losses, and returns a dictionary of them as tensors
        """

        def entropy_from_logits(logits):
            pass

        def KL_div_from_logits(logits_1, logits_2):
            pass

        def cross_ent_from_logits(logits_1, logits_2):
            pass
    
    def make_step(self, tokens: int[Tensor, "batch seq_len"], magic_token_pos:  int[Tensor, "2 n_magic_tokens"]) -> dict[str, float[Tensor, "1"]]:
        """
        takes a batched set of tokens, and a the magic token positions. It then creates the embeddings, runs the forwardpass, calculates the losses, and makes a step
        ot returns the losses

        """
        pass

    def log_top_tokens(self, n_top_tracked_tokens: int, specified_tokens: Optional[int[Tensor, "batch seq_len"]] = None) -> None:
        """
        it tracks the probabilities of the top n_top_tracked_tokens tokens, and logs them.
        """
        magic_porobs = t.nn.functional.softmax(self.magic_token_vector)
        top_tokens = t.argsort(magic_porobs)[-n_top_tracked_tokens:].tolist()

        for id in top_tokens:
            if id not in self.top_token_log.keys():
                self.top_token_log[id] = dict(epochs=[], prob=[])
            self.top_token_log[id]["epochs"].append(self.step)
            self.top_token_log[id]["prob"].append(magic_porobs[id].item())


    def train(self, dataset: Dataset, specified_tokens: Optional[int[Tensor, "batch seq_len"]] = None, wandb_log: bool = False, n_top_tracked_tokens = 10) -> Logs:
        """
        takes a dataset, creates a dataloader, iterates through dataloader, makes training steps and logs losses and top tokens, returns a log object
        """

        logs = Logs()
        pass

    def return_data(self) -> dict[str,Any]:
        pass
