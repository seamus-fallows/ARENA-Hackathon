# %%
import sys

sys.path.append("..")
from lib import Llama_Leaner, generate_data, CachedDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as t

llama_token = "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"
device = t.device("cuda" if t.cuda.is_available() else "cpu")
# %%
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", use_auth_token=llama_token
)
model = AutoModelForCausalLM.from_pretrained(
    f"meta-llama/Llama-2-7b-chat-hf", use_auth_token=llama_token
).to(device)
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
    system_prompt="You are a question answering assistent, you always follow instructions exactly and always answer questions correctly. Answer every question with either 'Yes' or 'No'.",
    # fmt: off
    user_prompt=lambda sentence, concept, token: [". Is the word ",token,"an example of a" ,concept,"? Think carefully about your answer and be confident in your response. Answer with either 'Yes' or 'No'."],
    # fmt: on
    ai_answer="Answer: ",
    yes_answer="Yes",
    no_answer="No",
)
dataset = CachedDataset.CachedDataset(
    model, tokenizer, data_token_ids, labels, prompt, sentence_cache_device=device
)
# %%
config = Llama_Leaner.Config()
config.magic_word = "magic"
config.loss_coeffs = {"label": 1.0, "kl": 0.2, "entropy": 0.2}
config.lr = 0.1
config.batch_size = 20
config.epochs = 500
dataloader = CachedDataset.CachedDataloader(
    dataset, batch_size=config.batch_size, shuffle=True, device=device
)
training = Llama_Leaner.Training(config, model, tokenizer)
solution = [
    tokenizer.encode("number")[-1],
    tokenizer.encode("numbers")[-1],
    tokenizer.encode("int")[-1],
]
trainings_logs = training.train(
    dataloader, specified_tokens=solution, n_top_tracked_tokens=5
)
trainings_logs.plot_losses(tokenizer)
trainings_logs.plot_top_tokens(tokenizer)
trainings_logs.plot_loss_tradeoff(tokenizer)
trainings_logs.plot_final_token_accuracy(tokenizer)
# %%
trainings_logs.plot_losses(tokenizer)
trainings_logs.plot_top_tokens(tokenizer)
trainings_logs.plot_loss_tradeoff(tokenizer)
trainings_logs.plot_final_token_accuracy(tokenizer)
# %%
len(dataloader)
# %%
tokenizer.encode(" Token:")
# %%
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

probs_on_label = np.array([])
random_embedding = t.rand((2, 5, model.config.hidden_size)).to(device)
for sentence_cache, question_end_ids, label in tqdm(dataloader):
    output = model(question_end_ids, past_key_values=sentence_cache, return_dict=True)
    output_probs = F.softmax(output.logits, dim=-1)
    prob_on_label = output_probs[0, -1, label].detach().cpu().numpy()
    probs_on_label = np.append(probs_on_label, prob_on_label)
    print(question_end_ids.shape)
    print(sentence_cache[0][0].shape)
    print(len(sentence_cache[0]))
    print(len(sentence_cache))
    break
# print(probs_on_label)
# print(np.mean(probs_on_label))
# %%
random_embedding = t.rand((2, 5, model.config.hidden_size)).to(device)
output = model(
    inputs_embeds=random_embedding, past_key_values=sentence_cache, return_dict=True
)
output.logits.shape
# %%
type(sentence_cache)
# %%
random_embedding = t.rand((2, 5, model.config.hidden_size)).to(device)
# %%
output = model(
    inputs_embeds=random_embedding, return_dict=True, past_key_values=sentence_cache
)
# %%
# %%
