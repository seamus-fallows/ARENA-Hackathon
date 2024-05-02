import sys
sys.path.append("..")
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch as t
from lib import Llama_Leaner, generate_data, CachedDataset
import seaborn as sns

def cachedDataset_test(sentences, concept, model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer.pad_token = tokenizer.eos_token

    prompt_dict = dict(
        system_prompt="Please answer the following question",
        user_prompt=lambda sentence, concept, token: ["In the sentence: >", sentence,"< is >",token,"< a " ,concept,"? "],
        ai_answer="Answer: ",
        yes_answer="Yes",
        no_answer="No",
    )

    prompt_function = CachedDataset.PromptUtil.add_syntax_to_prompt_func(prompt_dict)
    concept_id = t.tensor(tokenizer.encode(concept))

    data = tokenizer.batch_encode_plus(sentences, return_tensors="pt", padding=True).input_ids

    

