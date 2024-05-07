# %%
import sys

sys.path.append("..")
from lib import Llama_Leaner, generate_data, CachedDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as t

llama_token = "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"
device = t.device("cuda" if t.cuda.is_available() else "cpu")
# %%
experiment_file_path = generate_data.get_experiment_file_path(__file__,1)

# %%

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("NousResearch/Meta-Llama-3-8B").to(device)
# %%
numb_strings_1 = [str(i) for i in range(10)]
numb_strings_2 = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
]
numb_tokens_1 = tokenizer.batch_encode_plus(numb_strings_1, return_tensors="pt")[
    "input_ids"
][:, -1].flatten()
numb_tokens_2 = tokenizer.batch_encode_plus(numb_strings_2, return_tensors="pt")[
    "input_ids"
][:, -1].flatten()
number_tokens = t.cat([numb_tokens_1, numb_tokens_2])
# other_tokens = tokenizer.batch_encode_plus([chr(i) for i in range(97, 123)], return_tensors='pt')['input_ids'][:,-1].flatten()
data_token_ids, labels = generate_data.generate_data_tokens(
    n_data=40,
    seq_len=1,
    concept_tokens=number_tokens,
    vocab_size=tokenizer.vocab_size,
    p_concept=0.5,
)
print(labels)
for i in range(data_token_ids.size(0)):
    sequence = data_token_ids[i].tolist()
    decoded_sequence = tokenizer.decode(sequence)
    print(decoded_sequence)
prompt = dict(
    system_prompt="Your task is to assess if a given token (word) from some text represents a specified concept. Provide a rating based on this assessment:\nIf the token represents the concept, respond with 'Rating: 1'.\nIf the token does not represent the concept, respond with 'Rating: 0'.\nFocus solely on the token and use the other text for context only. Be confident.",
    # fmt: off
    user_prompt=lambda sentence, concept, token: ["Is the word \"",token,"\" an example of a\"" ,concept,"\"?"],
    # fmt: on
    ai_answer="Rating: ",
    yes_answer="1",
    no_answer="0",
)
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
file_to_check = experiment_file_path + "/training_logs.pkl"
if not os.path.isfile(file_to_check):
    trainings_logs = training.train(
        dataloader, specified_tokens=solution, n_top_tracked_tokens=5
    )
    trainings_logs.save(file_to_check)
else:
    trainings_logs = Llama_Leaner.TrainingLogs.load(file_to_check)



# %%
trainings_logs.plot_losses(tokenizer)
trainings_logs.plot_top_tokens(tokenizer)
trainings_logs.plot_loss_tradeoff(tokenizer)
trainings_logs.plot_final_token_accuracy(tokenizer)
# %%
