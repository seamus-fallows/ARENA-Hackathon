# %%
import torch as t
from torch.nn import functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
import einops
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Optional, Tuple, Any
from torch import Tensor
from dataclasses import dataclass, field
import torchtyping
from tqdm.notebook import tqdm
from jaxtyping import Int, Float
from typing import List, Dict
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset

t.manual_seed(0)
np.random.seed(0)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False

# Check for GPU availability
device = t.device("cuda" if t.cuda.is_available() else "cpu")


# create a dataset class
class TokenizedDataset(Dataset):
    def __init__(self, datapath: str, tokenizer):
        self.tokenizer = tokenizer
        self.datapath = datapath
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def visualise_item(self, idx):
        pass

    def tokenize_input(
        self, input_text: Union[str, list[str]], magic_word: str
    ) -> Tuple[Int[Tensor, "batch seq_len"], Int[Tensor, "batch seq_len"]]:
        if isinstance(input_text, str):
            input_text = [input_text]

        tokens = self.tokenizer.batch_encode(
            input_text, padding=True, return_tensors="pt"
        ).input_ids

        # assert, that the tokenizer padds with EOS and on the left
        assert (
            self.tokenizer.pad_token == self.tokenizer.eos_token
        ), "Tokenizer does not pad with EOS"
        assert (
            self.tokenizer.padding_side == "left"
        ), "Tokenizer does not pad on the left"

        magic_ids = self.tokenizer.encode(magic_word)[0]
        magic_token_pos = tokens == magic_ids
        return tokens, magic_token_pos


@dataclass
class Config:
    batch_size: int = 4
    epochs: int = 100
    lr: float = 1e-2
    intitialization_std: float = 1
    loss_coeffs: dict = field(
        default_factory=lambda: {"acc": 1.0, "kl": 1.0, "entropy": 1.0}
    )


@dataclass
class Logs:
    def __init__(
        self,
        losses: dict[str, Float[Tensor, "1"]],
        top_tokens: dict[Int, dict[str, Any]],
        specified_tokens: dict[Int, dict[str, Any]],
        config: Config,
        model_name: str,
        dataset_name: str,
        run_date: str,
    ):
        self.losses = losses
        self.top_tokens = top_tokens
        self.config = config
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.run_date = run_date
        self.specified_tokens = specified_tokens

    def save(self, path: str) -> None:
        pass

    def plot_losses(run_data: dict[str, Any]) -> None:
        pass

    def plot_top_tokens(run_data: dict[str, Any], n_plotted_tokens=10) -> None:
        pass

    def plot_loss_tradeoff(run_data: dict[str, Any], n_plotted_tokens=10) -> None:
        pass


class Training:
    def __init__(self, config: Config, model, tokenizer):
        self.config = config
        self.device = model.device
        self.model = model
        self.vocab_size = (
            model.config.vocab_size
        )  # Need to check this is the same for all models
        self.tokenizer = tokenizer
        self.intialise_random_token_vector()
        self.optimizer = AdamW(
            self.magic_token_vector, lr=config.lr
        )  # TODO: consider if it's better to have the optimizer at start of train loop - seems more readable
        self.loss_coeffs = config.loss_coeffs
        self.step = 0
        self.magic_ids = self.tokenizer.encode(config.magic_word)[0]

    def intialise_random_token_vector(self) -> None:
        """
        Creates a random vector of length vocab_size, and set it as the magic vecotr
        """
        magic_token_vector = t.empty(self.vocab_size, device=self.device).normal_(
            mean=0, std=self.config.intitialization_std
        )
        magic_token_vector = t.nn.Parameter(magic_token_vector, requires_grad=True)
        self.magic_token_vector = magic_token_vector

    def create_modified_embeddings(
        self,
        tokens: Int[Tensor, "batch seq_len"],
        magic_token_pos: Int[Tensor, "batch seq_len"],
    ) -> Float[Tensor, "batch seq_len d_model"]:
        """
        embeds the tokens, creates the embedding of the magic token, and puts it at all places in magic_token_pos
        """

        tokens = tokens.to(device)
        inputs_embeds = self.model.transformer.wte.weight[
            tokens
        ]  # TODO; check that it is the rightr way for llama
        embedding_matrix = self.model.transformer.wte.weight
        magic_token_embed = einops.einsum(
            embedding_matrix,
            self.magic_token_vector,
            " d_vocab d_model, d_vocab -> d_model ",
        )

        inputs_embeds[magic_token_pos] = magic_token_embed
        return inputs_embeds

    def calculate_losses(
        self,
        output_logits: Float[Tensor, "batch seq_len d_vocab"],
        magic_token_vector: Float[Tensor, "d_vocab"],
        magic_token_pos: Int[Tensor, "2 n_magic_tokens"],
        target_tokens: Int[Tensor, "batch"],
    ) -> dict[str, Float[Tensor, "1"]]:
        """
        calculate all the different losses, and returns a dictionary of them as tensors
        """

        # Helper functions for calculating losses
        def entropy_from_logits(magic_vector: float[Tensor, "batch d_vocab"]):
            probs = t.softmax(magic_vector, dim=-1)
            log_probs = t.log_softmax(magic_vector, dim=-1)
            return -(probs * log_probs).sum(dim=-1)

        def KL_div_from_logits(
            magic_vector: float[Tensor, "batch d_vocab"],
            prediction_on_magic_pos: float[Tensor, "batch d_vocab"],
        ):
            probs_1 = t.softmax(magic_vector, dim=-1)
            log_probs_2 = t.log_softmax(prediction_on_magic_pos, dim=-1)
            return F.kl_div(log_probs_2, probs_1, reduction="batchmean")

        pass

    def make_step(
        self,
        tokens: Int[Tensor, "batch seq_len"],
        magic_token_pos: Int[Tensor, "2 n_magic_tokens"],
    ) -> dict[str, Float[Tensor, "1"]]:
        """
        takes a batched set of tokens, and a the magic token positions. It then creates the embeddings, runs the forwardpass, calculates the losses, and makes a step
        ot returns the losses

        """
        self.optimizer.zero_grad()
        embeddings = self.create_modified_embeddings(tokens, magic_token_pos)
        output_logits = self.model(inputs_embeds=embeddings).logits
        losses = self.calculate_losses(
            output_logits, self.magic_token_vector, magic_token_pos, tokens
        )
        losses["total_loss"].backward()

    def log_top_tokens(
        self,
        n_top_tracked_tokens: int,
        specified_tokens: Optional[Int[Tensor, "batch seq_len"]] = None,
    ) -> None:
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

    def train(
        self,
        dataset: Dataset,
        specified_tokens: Optional[Int[Tensor, "batch seq_len"]] = None,
        wandb_log: bool = False,
        n_top_tracked_tokens=10,
    ) -> Logs:
        """
        takes a dataset, creates a dataloader, iterates through dataloader, makes training steps and logs losses and top tokens, returns a log object
        """

        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        # iterate through dataloader
        for epoch in tqdm(range(self.config.epochs)):
            for tokens in dataloader:
                magic_token_pos = tokens == self.magic_ids

                losses = self.make_step(tokens, magic_token_pos)
                self.log_top_tokens(n_top_tracked_tokens, specified_tokens)
                self.step += 1

        logs = Logs()
        pass

    def return_data(self) -> dict[str, Any]:
        pass


# %%
