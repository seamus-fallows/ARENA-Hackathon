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
import torchtyping as tp
from jaxtyping import Float
from tqdm.notebook import tqdm
from jaxtyping import Int
from typing import List, Dict

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
        self.tokenizer = tokenizer
        self.datapath = datapath
        pass
    def __getitem__(self, idx):
        pass
    def __len__(self):
        pass
    def plot_item(self, idx):
        pass

    def tokenize_input(self, input_text: Union[str, list[str]], magic_word: str) -> Tuple[int[Tensor, "batch seq_len"], int[Tensor, "n_magic_tokens 2"]]:
        if isinstance(input_text, str):
            input_text = [input_text]
            
        tokens = self.tokenizer.batch_encode(
            input_text, padding=True, return_tensors="pt"
        ).input_ids

        #assert, that the tokenizer padds with EOS and on the left
        assert self.tokenizer.pad_token == self.tokenizer.eos_token, "Tokenizer does not pad with EOS"
        assert self.tokenizer.padding_side == "left", "Tokenizer does not pad on the left"

        magic_ids = self.tokenizer.encode(magic_word)[0]
        magic_token_pos = t.stack(t.where(tokens == magic_ids), dim=-1) # a tensor of shape (n_magic_tokens, 2)

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

    def create_modified_embeddings(self, tokens: int[Tensor, "batch seq_len"], magic_token_pos:  int[Tensor, "n_magic_tokens 2"]) -> float[Tensor, "batch seq_len d_model"]:
        """
        embeds the tokens, creates the embedding of the magic token, and puts it at all places in magic_token_pos
        """

        tokens = tokens.to(device)
        inputs_embeds = self.model.transformer.wte.weight[tokens] # TODO; check that it is the rightr way for llama
        embedding_matrix = self.model.transformer.wte.weight
        magic_token_embed = einops.einsum(
            embedding_matrix,
            self.magic_token_vector,
            " d_vocab d_model, d_vocab -> d_model ",
        )

        inputs_embeds[magic_token_pos] = magic_token_embed
        return inputs_embeds


    def calculate_losses(self, output_logits: float[Tensor, "batch seq_len d_vocab"], magic_token_vector: float[Tensor, "d_vocab"], magic_token_pos: int[Tensor, "2 n_magic_tokens"], target_tokens: int[Tensor, "batch"]) -> dict[str, float[Tensor, "1"]]:
        """
        calculate all the different losses, and returns a dictionary of them as tensors
        """

        def entropy_from_logits(magic_vector: float[Tensor, "batch d_vocab"]):
            probs = t.softmax(magic_vector, dim=-1)
            log_probs = t.log_softmax(magic_vector, dim=-1)
            return - (probs * log_probs).sum(dim=-1)

        def KL_div_from_logits(magic_vector, prediction_on_magic_pos):
            probs_1 = t.softmax(magic_vector, dim=-1)
            log_probs_2 = t.log_softmax(prediction_on_magic_pos, dim=-1)
            return F.kl_div(log_probs_2, probs_1, reduction='batchmean')

        
        final_token_logits = output_logits[:, -1]
        predicted_logits_for_magic_token = output_logits[0, magic_word_pos-1]
        pass

    
    def make_step(self, tokens: int[Tensor, "batch seq_len"], magic_token_pos:  int[Tensor, "2 n_magic_tokens"]) -> dict[str, float[Tensor, "1"]]:
        """
        takes a batched set of tokens, and a the magic token positions. It then creates the embeddings, runs the forwardpass, calculates the losses, and makes a step
        ot returns the losses

        """
        embeddings = self.create_modified_embeddings(tokens, magic_token_pos)
        output_logits = self.model(inputs_embeds=embeddings).logits
        losses = self.calculate_losses(output_logits, self.magic_token_vector, magic_token_pos, tokens)
        losses["total_loss"].backward()

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
# %%

model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
#config = Config( batch_size=4, steps=100, lr=1e-2, intitialization_std=1, loss_coeffs={"accuracy": 1, "kl": 1, "entropy": 1})
#dataset = Dataset("data/processed/processed_data.txt", tokenizer)

#trainer = Training(config, model, tokenizer)
#logs = trainer.train(dataset)
# %%

tokens = tokenizer.batch_encode_plus(["hello magic I am magic", " magic"], padding=True, return_tensors="pt").input_ids
magic_ids = tokenizer.encode(' magic')[0]
print(tokens)
print(magic_ids)
magic_token_pos = t.stack(t.where(tokens == magic_ids), dim=-1)
print(magic_token_pos)
# %%
def KL_div_from_logits(logits_1, logits_2):
    probs_1 = t.softmax(logits_1, dim=-1)
    log_probs_2 = t.log_softmax(logits_2, dim=-1)
    return F.kl_div(log_probs_2, probs_1, reduction='batchmean')

logits_1 = t.rand(10)
logits_2 = t.rand(4,10)
test = KL_div_from_logits(logits_1, logits_2)
print(test)
# %%
magic_token_vector = t.empty(model.config.vocab_size, device=device).normal_(mean=0, std=1)
tokens = tokens.to(device)
inputs_embeds = model.transformer.wte.weight[tokens] # TODO; check that it is the rightr way for llama
embedding_matrix = model.transformer.wte.weight
magic_token_embed = einops.einsum(
    embedding_matrix,
    magic_token_vector,
    " d_vocab d_model, d_vocab -> d_model ",
)
# %%
print(magic_token_embed)
print(magic_token_pos)
#inputs_embeds[magic_token_pos.to(device)] #= magic_token_embed
# %%

def tokenize_input(self, input_text, magic_word) -> int:
    if isinstance(input_text, str):
        input_text = [input_text]
        
    tokens = self.tokenizer.batch_encode(
        input_text, padding=True, return_tensors="pt"
    ).input_ids

    #assert, that the tokenizer padds with EOS and on the left
    assert self.tokenizer.pad_token == self.tokenizer.eos_token, "Tokenizer does not pad with EOS"
    assert self.tokenizer.padding_side == "left", "Tokenizer does not pad on the left"

    magic_ids = self.tokenizer.encode(magic_word)[0]
    magic_token_pos = t.stack(t.where(tokens == magic_ids), dim=-1) # a tensor of shape (n_magic_tokens, 2)

    return tokens, magic_token_pos

# %%

def function(tend: Tuple[int float]):
    print(8)
# %%
