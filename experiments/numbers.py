# %%
import sys

sys.path.append("..")
from lib import Llama_Leaner, generate_data, CachedDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as t

llama_token = "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"
device = t.device("cuda" if t.cuda.is_available() else "cpu")
# %%
experiment_file_path, experiment_new = generate_data.get_experiment_file_path(
    __file__, 1
)
# %%

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
if experiment_new:
    model = AutoModelForCausalLM.from_pretrained("NousResearch/Meta-Llama-3-8B").to(
        device
    )
# %%
if experiment_new:
    prompt = dict(
        system_prompt="Your task is to assess if a given token (word) from some text represents a specified concept. Provide a rating based on this assessment:\nIf the token represents the concept, respond with 'Rating: 1'.\nIf the token does not represent the concept, respond with 'Rating: 0'.\nFocus solely on the token and use the other text for context only. Be confident.",
        # fmt: off
        user_prompt=lambda sentence, concept, token: ["Is the word \"",token,"\" an example of a\"" ,concept,"\"?"],
        # fmt: on
        ai_answer="Rating: ",
        yes_answer="1",
        no_answer="0",
    )
    data_path = "../data/text_data/numbers"
    datagenerator = generate_data.TextDataGenerator(
        file_path=data_path, tokenizer=tokenizer
    )
    data_token_ids, labels = datagenerator.generate_data()
    dataset = CachedDataset.CachedDataset(
        model,
        tokenizer,
        data_token_ids,
        labels,
        prompt,
        sentence_cache_device="cpu",
        p_threshold=0.5,
    )

# %%
if experiment_new:
    config = Llama_Leaner.Config()
    config.magic_word = "magic"
    config.loss_coeffs = {"label": 1.0, "kl": 0.2, "entropy": 0.2}
    config.lr = 0.1
    config.batch_size = 20
    config.epochs = 10
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
# check wether there is a training_logs file in the experiment folder

logs_file = experiment_file_path + "/training_logs.pkl"
if experiment_new:
    trainings_logs = training.train(
        dataloader, specified_tokens=solution, n_top_tracked_tokens=5
    )
    trainings_logs.save(logs_file)
else:
    trainings_logs = Llama_Leaner.Logs.load(logs_file)


# %%
trainings_logs.plot_losses(tokenizer, saving_folder_path=experiment_file_path)
trainings_logs.plot_top_tokens(tokenizer, saving_folder_path=experiment_file_path)
trainings_logs.plot_loss_tradeoff(tokenizer, saving_folder_path=experiment_file_path)
trainings_logs.plot_final_token_accuracy(
    tokenizer, saving_folder_path=experiment_file_path
)
# %%
