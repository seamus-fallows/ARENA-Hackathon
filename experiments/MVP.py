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
import datetime


t.manual_seed(0)
np.random.seed(0)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False

# Check for GPU availability
device = t.device("cuda" if t.cuda.is_available() else "cpu")


# create a dataset class
class CustomDataset(Dataset):
    def __init__(self, text_data: List[Tuple[str]], name: str):
        self.text_data = text_data
        self.name = name

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
        default_factory=lambda: {"label": 1.0, "kl": 1.0, "entropy": 1.0},
    )
    magic_word: str = " magic"


@dataclass
class Logs:
    def __init__(
        self,
        losses: dict[str, Float[Tensor, "1"]],
        top_tokens: dict[Int, dict[str, Any]],
        specified_tokens: dict[Int, dict[str, Any]],
        config: Config,
        model_name: str = None,
        dataset_name: str = None,
        run_date: str = None,
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

    def plot_losses(self, tokenizer, figsize: Tuple[int] = (10, 5)) -> None:
        plt.figure(figsize=figsize)
        plt.plot(self.losses["loss"], label="Total Loss")
        plt.plot(self.losses["label_loss"], label="label Loss")
        plt.plot(self.losses["entropy_loss"], label="Entropy Loss")
        plt.plot(self.losses["kl_loss"], label="KL Loss")
        for id in self.specified_tokens.keys():
            steps = self.specified_tokens[id]["steps"]
            total_loss = np.full_like(steps, self.specified_tokens[id]["total_loss"])
            plt.plot(
                steps,
                total_loss,
                label=f"{tokenizer.decode([id])} loss",
                linestyle="--",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def get_plot_keys(self, n_plotted_tokens):
        plot_keys = sorted(
            self.top_tokens.keys(),
            key=lambda x: self.top_tokens[x]["max_prob"],
            reverse=True,
        )[:n_plotted_tokens]
        plot_keys += self.specified_tokens.keys()
        return plot_keys

    def plot_top_tokens(self, tokenizer, n_plotted_tokens: int = 10) -> None:
        plt.figure(figsize=(10, 5))
        # make a color list of the length of n_plots
        colors = plt.cm.rainbow(
            np.linspace(0, 1, n_plotted_tokens + len(self.specified_tokens))
        )

        plot_keys = self.get_plot_keys(n_plotted_tokens)
        combined_log = {**self.top_tokens, **self.specified_tokens}

        # only take unique plot keys without changing order
        plot_keys = list(dict.fromkeys(plot_keys))

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
            continuous_step_sections.append(step_section)
            continuous_prob_sections.append(prob_section)
            linestyle = "--" if token_id in self.specified_tokens.keys() else "-"

            for step_section, prob_section in zip(
                continuous_step_sections, continuous_prob_sections
            ):
                plt.plot(step_section, prob_section, color=color, linestyle=linestyle)
            plt.plot(
                [],
                [],
                label=tokenizer.decode([token_id]),
                color=color,
                linestyle=linestyle,
            )

        plt.xlabel("Epoch")
        plt.ylabel("Probability")
        plt.legend()
        plt.show()

    def plot_loss_tradeoff(
        self, tokenizer, n_plotted_tokens: int = 10, figsize: Tuple[int] = (3, 3)
    ) -> None:
        plot_keys = self.get_plot_keys(n_plotted_tokens)
        plot_keys = list(dict.fromkeys(plot_keys))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(plot_keys)))

        combined_log = {**self.top_tokens, **self.specified_tokens}
        plt.figure(figsize=figsize)
        for id, color in zip(plot_keys, colors):
            if "label_loss" in combined_log[id].keys():
                if id in self.specified_tokens.keys():
                    marker = "*"
                else:
                    marker = "o"
                plt.scatter(
                    combined_log[id]["label_loss"],
                    combined_log[id]["kl_loss"],
                    label=tokenizer.decode([id]),
                    marker=marker,
                    color=color,
                )
        # legend outside of plot
        plt.plot(
            self.losses["label_loss"], self.losses["kl_loss"], label="loss tradeoff"
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        plt.xlabel("label loss")
        # plt.xscale("log")
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
        magic_token_vector: Optional[Float[Tensor, "d_vocab"]] = None,
    ) -> Float[Tensor, "batch seq_len d_model"]:
        """
        embeds the tokens, creates the embedding of the magic token, and puts it at all places in magic_token_pos
        """
        if magic_token_vector is None:
            magic_token_vector = self.magic_token_vector
        tokens = tokens.to(device)
        inputs_embeds = self.model.transformer.wte.weight[
            tokens
        ]  # TODO; check that it is the rightr way for llama
        embedding_matrix = self.model.transformer.wte.weight
        magic_token_embed = einops.einsum(
            embedding_matrix,
            F.softmax(magic_token_vector, dim=0),
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
            return F.kl_div(log_probs_2, probs_1, reduction="batchmean")

        final_token_logits = output_logits[:, -1, :]
        label_loss = F.cross_entropy(final_token_logits, target_tokens)

        entropy_loss = entropy_from_logits(magic_token_vector)

        shifted_magic_token_pos = magic_token_pos[:, 1:]
        shifted_output_logits = output_logits[:, :-1, :]
        prediction_on_magic_pos = shifted_output_logits[shifted_magic_token_pos]

        kl_loss = KL_div_from_logits(magic_token_vector, prediction_on_magic_pos)

        total_loss = (
            self.loss_coeffs["label"] * label_loss
            + self.loss_coeffs["kl"] * kl_loss
            + self.loss_coeffs["entropy"] * entropy_loss
        )

        return total_loss, label_loss, kl_loss, entropy_loss

    def make_step(
        self,
        tokens: Int[Tensor, "batch seq_len"],
        target_tokens: Int[Tensor, "batch seq_len"],
        magic_token_pos: Int[Tensor, "batch seq_len"],
    ) -> dict[str, Float[Tensor, "1"]]:
        """
        takes a batched set of tokens, and a the magic token positions. It then creates the embeddings, runs the forwardpass, calculates the losses, and makes a step
        ot returns the losses

        """
        self.optimizer.zero_grad()
        embeddings = self.create_modified_embeddings(tokens, magic_token_pos)
        output_logits = self.model(inputs_embeds=embeddings).logits
        total_loss, label_loss, kl_loss, entropy_loss = self.calculate_losses(
            output_logits, self.magic_token_vector, magic_token_pos, target_tokens
        )
        total_loss.backward()
        self.optimizer.step()

        loss_log = {
            "loss": total_loss.item(),
            "label_loss": label_loss.item(),
            "kl_loss": kl_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }
        return loss_log

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
                self.top_token_log[id] = dict(steps=[], prob=[])
            self.top_token_log[id]["steps"].append(self.step)
            self.top_token_log[id]["prob"].append(magic_porobs[id].item())

        for id in specified_tokens:
            if id not in self.specified_tokens_log.keys():
                self.specified_tokens_log[id] = dict(steps=[], prob=[])
            self.specified_tokens_log[id]["steps"].append(self.step)
            self.specified_tokens_log[id]["prob"].append(magic_porobs[id].item())

    def add_ratings_to_id_log(self, dataloader, n_top_tracked_tokens: int = 10) -> None:
        top_token_ids = sorted(
            self.top_token_log.keys(),
            key=lambda x: self.top_token_log[x]["max_prob"],
            reverse=True,
        )[:n_top_tracked_tokens]

        logs = [self.top_token_log, self.specified_tokens_log]
        keys_lists = [top_token_ids, self.specified_tokens_log.keys()]

        for log, keys in zip(logs, keys_lists):
            for id in keys:
                onehot_vector = t.ones(self.vocab_size).to(device) * (-1e10)
                onehot_vector[id] = 1e10
                for sentences, targets in dataloader:
                    tokens = self.tokenizer(
                        sentences, return_tensors="pt", padding=True
                    ).input_ids.to(device)
                    target_tokens = self.tokenizer(
                        targets, return_tensors="pt", padding=True
                    ).input_ids.to(device)
                    assert (
                        target_tokens.shape[1] == 1
                    ), "target tokens must be a single token"
                    target_tokens = target_tokens.squeeze(-1)
                    magic_token_pos = tokens == self.magic_ids
                    embeddings = self.create_modified_embeddings(
                        tokens, magic_token_pos, onehot_vector
                    )
                    output_logits = self.model(inputs_embeds=embeddings).logits
                    (
                        total_loss,
                        label_loss,
                        kl_loss,
                        entropy_loss,
                    ) = self.calculate_losses(
                        output_logits, onehot_vector, magic_token_pos, target_tokens
                    )
                    log[id]["total_loss"] = total_loss.item()
                    log[id]["label_loss"] = label_loss.item()
                    log[id]["kl_loss"] = kl_loss.item()
                    log[id]["entropy_loss"] = entropy_loss.item()

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
        self.optimizer = t.optim.AdamW([self.magic_token_vector], lr=config.lr)
        dataloader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        self.loss_log = defaultdict(list)
        self.top_token_log = {}
        self.specified_tokens_log = {}

        # iterate through dataloader
        for epoch in tqdm(range(self.config.epochs)):
            for sentences, targets in dataloader:
                tokens = self.tokenizer(
                    sentences, return_tensors="pt", padding=True
                ).input_ids.to(device)
                target_tokens = self.tokenizer(
                    targets, return_tensors="pt", padding=True
                ).input_ids.to(device)
                assert (
                    target_tokens.shape[1] == 1
                ), "target tokens must be a single token"
                target_tokens = target_tokens.squeeze(-1)

                magic_token_pos = tokens == self.magic_ids

                losses = self.make_step(tokens, target_tokens, magic_token_pos)
                for key, value in losses.items():
                    self.loss_log[key].append(value)

                self.log_top_tokens(n_top_tracked_tokens, specified_tokens)
                self.step += 1

        dataset_name = dataset.name if hasattr(dataset, "name") else None
        model_name = self.model.name if hasattr(self.model, "name") else None
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # add maximum probability to the log of each id
        for id in self.top_token_log.keys():
            self.top_token_log[id]["max_prob"] = max(self.top_token_log[id]["prob"])

        self.add_ratings_to_id_log(dataloader, n_top_tracked_tokens)

        logs = Logs(
            self.loss_log,
            self.top_token_log,
            self.specified_tokens_log,
            self.config,
            model_name,
            dataset_name,
            time,
        )
        return logs


# %%

string_list = [
    ("I live in a European country called magic, and its capital city is", " Paris"),
    ("England won the war against magic in the famous battle of", " England"),
    ("To the east of magic is the country of", " Poland"),
    ("The president of magic is", " Merkel"),
    ("A nice place to visit in magic is", " Munich"),
]
dataset = CustomDataset(string_list, "France example")

config = Config()
config.loss_coeffs = {"label": 1.0, "kl": 0.5, "entropy": 0.5}
config.lr = 0.2
config.batch_size = 5
config.epochs = 100
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


training = Training(config, model, tokenizer)
solution = [tokenizer.encode(" France")[0]]

trainings_logs = training.train(
    dataset, specified_tokens=solution, n_top_tracked_tokens=10
)
trainings_logs.plot_losses(tokenizer)
trainings_logs.plot_top_tokens(tokenizer)
trainings_logs.plot_loss_tradeoff(tokenizer)
# %%
trainings_logs.plot_losses()
trainings_logs.plot_top_tokens(tokenizer)
trainings_logs.plot_loss_tradeoff(tokenizer)

# %%
for id in trainings_logs.top_tokens.keys():
    if hasattr(trainings_logs.top_tokens[id], "label_loss"):
        print(tokenizer.decode([id]))
        print(trainings_logs.top_tokens[id]["label_loss"])


# %%

tokens = t.tensor(tokenizer(["The only thing to magic is"], padding=True).input_ids).to(
    model.device
)
target_tokens = (
    t.tensor(tokenizer([" fear"], padding=True).input_ids).to(model.device).squeeze(-1)
)
magic_token_pos = tokens == (tokenizer.encode(" magic")[0])

test_id = tokenizer.encode(" fear")[0]
test_id_2 = tokenizer.encode(" ?")[0]

onehot_vector = t.zeros(model.config.vocab_size).to(device)
onehot_vector[test_id] = 1

onehot_vector_2 = t.zeros(model.config.vocab_size).to(device)
onehot_vector_2[test_id_2] = 1

embeddings = training.create_modified_embeddings(tokens, magic_token_pos, onehot_vector)
embeddings_2 = training.create_modified_embeddings(
    tokens, magic_token_pos, onehot_vector_2
)

output_logits = model(inputs_embeds=embeddings).logits
output_logits_2 = model(inputs_embeds=embeddings_2).logits

losses_1 = training.calculate_losses(
    output_logits, onehot_vector, magic_token_pos, target_tokens
)
losses_2 = training.calculate_losses(
    output_logits_2, onehot_vector_2, magic_token_pos, target_tokens
)

print(embeddings - embeddings_2)

# %%
print(
    f"total loss: {losses_1[0].item()}, label loss: {losses_1[1].item()}, kl loss: {losses_1[2].item()}, entropy loss: {losses_1[3].item()}"
)
print(
    f"total loss: {losses_2[0].item()}, label loss: {losses_2[1].item()}, kl loss: {losses_2[2].item()}, entropy loss: {losses_2[3].item()}"
)


# %%
def entropy_from_logits(magic_vector: Float[Tensor, "batch d_vocab"]):
    probs = t.softmax(magic_vector, dim=-1)
    log_probs = t.log_softmax(magic_vector, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


print(entropy_from_logits(onehot_vector))
print(entropy_from_logits(onehot_vector_2))
# %%
onehot_vector = t.zeros(3).to(device)
onehot_vector[0] = 1


def entropy_from_logits(magic_vector: Float[Tensor, "batch d_vocab"]):
    probs = t.softmax(magic_vector, dim=-1)
    log_probs = t.log_softmax(magic_vector, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


print(entropy_from_logits(onehot_vector))


# %%
trainings_logs.specified_tokens
# %%
