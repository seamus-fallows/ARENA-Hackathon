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
    def __init__(self, text_data: List[Tuple[str]]):
        self.text_data = text_data    

    def __getitem__(self, idx):
        return self.text_data[idx]

    def __len__(self):
        return len(self.text_data)

    def visualise_item(self, idx):
        pass


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

    def plot_losses(self, figsize: Tuple[int] = (10,5)) -> None:
        
        plt.figure(figsize=figsize)
        plt.plot(self.losses["loss"], label="Total Loss")
        plt.plot(self.losses["label_loss"], label="label Loss")
        plt.plot(self.losses["entropy_loss"], label="Entropy Loss")
        plt.plot(self.losses["kl_loss"], label="KL Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def get_plot_keys(self, n_plotted_tokens):
        plot_keys = sorted(
            self.top_tokens.keys(), key=lambda x: self.top_tokens[x]["max_prob"], reverse=True
        )[:n_plotted_tokens]
        plot_keys += self.specified_tokens.keys()
        return plot_keys
    
    def plot_top_tokens(self, tokenizer, n_plotted_tokens: int = 10) -> None:

        plt.figure(figsize=(10, 5))
        # make a color list of the length of n_plots
        colors = plt.cm.rainbow(np.linspace(0, 1, n_plotted_tokens + len(self.specified_tokens)))

        plot_keys = self.get_plot_keys(n_plotted_tokens)
        combined_log = {**self.top_tokens, **self.specified_tokens}

        for token_id, color in zip(plot_keys, colors):
            steps = combined_log[token_id]["steps"]
            probs = combined_log[token_id]["prob"]

            continuous_step_sections = []
            continuous_prob_sections = []

            step_section = []
            prob_section = []

            last_step = -1
            for step, prob in zip(steps, probs):
                if step != last_step + 1:
                    continuous_step_sections.append(step_section)
                    continuous_prob_sections.append(prob_section)
                    step_section = []
                    prob_section = []
                step_section.append(step)
                prob_section.append(prob)
                last_step = step

            for step_section, prob_section in zip(continuous_step_sections, continuous_prob_sections):
                plt.plot(step_section, prob_section, color=color)
            plt.plot([], [], label=tokenizer.decode([token_id], color=color))

        plt.xlabel("Epoch")
        plt.ylabel("Probability")
        plt.legend()
        plt.show()

    def plot_loss_tradeoff(self, tokenizer, n_plotted_tokens: int = 10, figsize: Tuple[int] = (3,3)) -> None:
        plot_keys = self.get_plot_keys(n_plotted_tokens)
        combined_log = {**self.top_tokens, **self.specified_tokens}
        plt.figure(figsize=figsize)
        for id in plot_keys:
            if "label_loss" in combined_log[id].keys():
                plt.scatter(
                    combined_log[id]["label_loss"],
                    combined_log[id]["kl_lambda"],
                    label=tokenizer.decode([id]),
                )
        # legend outside of plot
        plt.plot(self.losses["label_loss"], self.losses["kl_loss"], label="loss tradeoff")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        plt.xlabel("label loss")
        plt.ylabel("kl loss")
        plt.show()


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
        
        self.loss_coeffs = config.loss_coeffs
        self.step = 0
        tokenized_magic_word = self.tokenizer.encode(config.magic_word)
        assert len(tokenized_magic_word) == 1, "Magic word must be a single token"
        self.magic_ids = tokenized_magic_word[0]

    def intialise_random_token_vector(self) -> None:
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
        output_logits: Float[Tensor, "batch_dim seq_len d_vocab"],
        magic_token_vector: Float[Tensor, "d_vocab"],
        magic_token_pos: Int[Tensor, "batch_dim seq_len"],
        target_tokens: Int[Tensor, "batch_dim"],
    ) -> dict[str, Float[Tensor, "1"]]:
        """
        calculate all the different losses, and returns a dictionary of them as tensors
        """

        # Helper functions for calculating losses
        def entropy_from_logits(magic_vector: Float[Tensor, "batch d_vocab"]):
            probs = t.softmax(magic_vector, dim=-1)
            log_probs = t.log_softmax(magic_vector, dim=-1)
            return -(probs * log_probs).sum(dim=-1)

        def KL_div_from_logits(
            magic_vector: Float[Tensor, "d_vocab"],
            prediction_on_magic_pos: Float[Tensor, "n_magic_tokens d_vocab"],
        ):
            probs_1 = t.softmax(magic_vector, dim=-1)
            log_probs_2 = t.log_softmax(prediction_on_magic_pos, dim=-1)
            return F.kl_div(log_probs_2, probs_1, reduction="mean")
    
        final_token_logits = output_logits[:, -1, :]
        label_loss = F.cross_entropy(final_token_logits, target_tokens)

        entropy_loss = entropy_from_logits(magic_token_vector)

        shifted_magic_token_pos = magic_token_pos[:,1:]
        shifted_output_logits = output_logits[:,:-1,:]
        prediction_on_magic_pos = shifted_output_logits[shifted_magic_token_pos]

        kl_loss = KL_div_from_logits(magic_token_vector, prediction_on_magic_pos)

        total_loss = self.loss_coeffs["label"] * label_loss + self.loss_coeffs["kl"] * kl_loss + self.loss_coeffs["entropy"] * entropy_loss

        losses = dict(
            label_loss=label_loss,
            kl_loss=kl_loss,
            entropy_loss=entropy_loss,
            total_loss=total_loss,
        )

        return losses


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
        self.optimizer = AdamW(
            self.magic_token_vector, lr=config.lr
        ) 
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

        return tokens
# %%
test_dataset = [("This is a test", "This is a test"), ("This is a test", "This is a test"), ("This is a test", "This is a test"), ("This is a test", "This is a test")]
dataset = TokenizedDataset(test_dataset)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
for batch in dataloader:
    print(batch)    
# %%
config = Config()
config.loss_coeffs['acc']=0
print(config.loss_coeffs)
# %%
def KL_div_from_logits(
    magic_vector: Float[Tensor, "d_vocab"],
    prediction_on_magic_pos: Float[Tensor, "n_magic_tokens d_vocab"],
):
    probs_1 = t.softmax(magic_vector, dim=-1)
    log_probs_2 = t.log_softmax(prediction_on_magic_pos, dim=-1)
    return F.kl_div(log_probs_2, probs_1, reduction="mean")

magiv_vector = t.rand(20)
prediction_on_magic_pos = t.rand(10, 20)

kl_loss = KL_div_from_logits(magiv_vector, prediction_on_magic_pos)

print(kl_loss.shape)
print(kl_loss)
# %%

setnenses = ["This is a test", "This is a test", "This is a test", "hello test is a test"]
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

tokens = tokenizer.batch_encode_plus(setnenses, padding=True, return_tensors="pt").input_ids

magic_ids = tokenizer.encode(" test")[0]
magic_token_pos = (tokens == magic_ids)
print(magic_token_pos)

example_lotis = t.rand(tokens.shape[0], tokens.shape[1], 20)
magic_token_pos = magic_token_pos[:,1:]
example_lotis = example_lotis[:,:-1,:]
example_lotis[magic_token_pos].shape
# %%
