#%%
import sys
sys.path.append("..")
from lib import Llama_Leaner, generate_data, CachedDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as t
device = t.device("cuda" if t.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("NousResearch/Meta-Llama-3-8B").to(device)

prompt = dict(
    # system_prompt = "Your task is to assess if a given word from some text belongs to a specific category or is an example of a specific concept. Provide a rating based on this assessment:\nIf the word is an example of the category/concept, respond with 'Rating: 1'.\nIf the word is not an example of the category/concept, respond with 'Rating: 0'.\nFocus solely on the word and use the other text for context only. Be confident.",
    system_prompt="Your task is to assess if a given word from some text represents a specified concept. Provide a rating based on this assessment:\nIf the word represents the concept, respond with 'Rating: 1'.\nIf the word does not represent the concept, respond with 'Rating: 0'.\nFocus solely on the word and use the other text for context only. Be confident.",
    # fmt: off
    user_prompt=lambda sentence, concept, token: ["The text is: \"",sentence,"\". From this text, is the word \"",token,"\" an example of the concept \"" ,concept,"\"?"],
    # fmt: on
    ai_answer="Rating: ",
    yes_answer="1",
    no_answer="0",
)

data_token_ids, labels = generate_data.generate_animal_data(tokenizer)

dataset = CachedDataset.CachedDataset(
    model, tokenizer, data_token_ids, labels, prompt, sentence_cache_device=device, p_threshold=0.5
)

#%%

config = Llama_Leaner.Config()
config.magic_word = "magic"
config.loss_coeffs = {"label": 1.0, "kl": 0.2, "entropy": 0.2}
config.lr = 0.2
config.batch_size = 5
config.epochs = 10
dataloader = CachedDataset.CachedDataloader(
    dataset, batch_size=config.batch_size, shuffle=True, device=device
)
training = Llama_Leaner.Training(config, model, tokenizer)
solution = [
    tokenizer.encode("animal")[-1],
    tokenizer.encode("animals")[-1],
    tokenizer.encode(" animal")[-1],
    tokenizer.encode(" animals")[-1],
]

trainings_logs = training.train(
    dataloader, specified_tokens=solution, n_top_tracked_tokens=5
)
trainings_logs.plot_losses(tokenizer)
trainings_logs.plot_top_tokens(tokenizer)
trainings_logs.plot_loss_tradeoff(tokenizer)
trainings_logs.plot_final_token_accuracy(tokenizer)
# %%
