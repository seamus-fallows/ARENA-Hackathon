# %%
import sys

sys.path.append("..")
from lib import Llama_Leaner, generate_data_token_ids, CachedDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as t

llama_token = "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"
device = t.device("cuda" if t.cuda.is_available() else "cpu")
# %%
experiment_file_path, experiment_new = generate_data_token_ids.get_experiment_file_path(
    __file__, 4
)
print(experiment_new)
# %%
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
if experiment_new:
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct"
    ).to(device)
# %%
if experiment_new:
    prompt = dict(
    # system_prompt = "Your task is to assess if a given word from some text belongs to a specific category or is an example of a specific concept. Provide a rating based on this assessment:\nIf the word is an example of the category/concept, respond with 'Rating: 1'.\nIf the word is not an example of the category/concept, respond with 'Rating: 0'.\nFocus solely on the word and use the other text for context only. Be confident.",
    system_prompt="""Your task is to assess if a given word from some text is an example of or represents particular concept. You will be given the information in the following format:
Text: <text>
Word: <word>
Concept: <concept>
You should ask yourself "is <word> an example of <concept>?". If the word could be considered an example of or representing the concept, reply with “Answer: Yes”. If the word is not an example of the concept, reply with “Answer: No”. Consider the semantic and syntactic aspects of the word, as well as its context and usage in the sentence.
Here are some examples of the task:

Text: Rex barked at the postman,
Word: Rex,
Concept: dog,
Answer: Yes

Text: The sky is blue,
Word: blue,
Concept: color,
Answer: Yes

Text: The sky is blue,
Word: blue,
Concept: palindrome,
Answer: No

Text: Jetzt steh ich hier ich armer Tor und bin so klug als wie zuvor,
Word: zuvor,
Concept: German,
Answer: Yes""",
    # fmt: off
    user_prompt=lambda sentence, concept, token: ["Text: ",sentence,"\nWord: ",token,"\nConcept:" ,concept,"\n"],
    # fmt: on
    ai_answer="Answer:",
    yes_answer=" Yes",
    no_answer=" No",
)

    data_path = "../data/text_data/chinese"
    datagenerator = generate_data_token_ids.TextDataGenerator(
        file_path=data_path, tokenizer=tokenizer
    )
    data_token_ids, labels = datagenerator.generate_data()
    dataset = CachedDataset.CachedDataset(
        model,
        tokenizer,
        data_token_ids,
        labels,
        prompt,
        sentence_cache_device="cuda",
        p_threshold=0.5,
    )

# %%
if experiment_new:
    config = Llama_Leaner.Config()
    config.magic_word = "magic"
    config.loss_coeffs = {"label": 1.0, "kl": 0.2, "entropy": 0.2}
    config.lr = 0.1
    config.batch_size = 10
    config.epochs = 80
    dataloader = CachedDataset.CachedDataloader(
        dataset, batch_size=config.batch_size, shuffle=True, device=device
    )
    training = Llama_Leaner.Training(config, model, tokenizer)
    solution = [
        tokenizer.encode("Chinese")[-1],
        tokenizer.encode("chinese")[-1],
        tokenizer.encode(" Chinese")[-1],
        tokenizer.encode(" chinese")[-1],
    ]
# %%
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
