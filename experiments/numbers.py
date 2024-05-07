# %%
import sys

sys.path.append("..")
from lib import Llama_Leaner, generate_data, CachedDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as t

llama_token = "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"
device = t.device("cuda" if t.cuda.is_available() else "cpu")
# %%
experiment_file_path, experiment_new = generate_data.get_experiment_file_path(__file__,1)
# %%

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
if experiment_new:
    model = AutoModelForCausalLM.from_pretrained("NousResearch/Meta-Llama-3-8B").to(device)
# %%

dataset = CachedDataset.CachedDataset(
    model, tokenizer, data_token_ids, labels, prompt, sentence_cache_device=device

# %%

config = Llama_Leaner.Config()
config.magic_word = "magic"
config.loss_coeffs = {"label": 1.0, "kl": 0.2, "entropy": 0.2}
config.lr = 0.1
config.batch_size = 20
config.epochs = 100
dataloader = CachedDataset.CachedDataloader(
    dataset, batch_size=config.batch_size, shuffle=True, device=device
)
training = Llama_Leaner.Training(config, model, tokenizer)
solution = [
    tokenizer.encode("number")[-1],
    tokenizer.encode("numbers")[-1],
    tokenizer.encode("int")[-1],
    tokenizer.encode(" number")[-1],
    tokenizer.encode(" numbers")[-1],
    tokenizer.encode(" int")[-1],
]

# %%
#check wether there is a training_logs file in the experiment folder
logs_file = experiment_file_path + "/training_logs.pkl"
if experiment_new:
    trainings_logs = training.train(
        dataloader, specified_tokens=solution, n_top_tracked_tokens=5
    )
    trainings_logs.save(logs_file)
else:
    trainings_logs = Llama_Leaner.TrainingLogs.load(logs_file)



# %%
trainings_logs.plot_losses(tokenizer)
trainings_logs.plot_top_tokens(tokenizer)
trainings_logs.plot_loss_tradeoff(tokenizer)
trainings_logs.plot_final_token_accuracy(tokenizer)
# %%
