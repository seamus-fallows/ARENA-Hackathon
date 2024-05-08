#%%
import sys
sys.path.append("..")
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch as t
from lib import Llama_Leaner, generate_data, CachedDataset
import seaborn as sns
from tqdm import tqdm

def cachedDataset_test(sentences, concept, model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer.pad_token = tokenizer.eos_token

    # create comparison logits by passing full prompts through the model
    prompt_dict = dict(
        system_prompt="Please answer the following question",
        user_prompt=lambda sentence, concept, token: ["In the sentence: >", sentence,"< is >",token,"< a " ,concept,"? "],
        ai_answer="Answer: ",
        yes_answer="Yes",
        no_answer="No",
    )

    prompt_function = CachedDataset.PromptUtil.add_syntax_to_prompt_func(prompt_dict)
    concept_id = t.tensor(tokenizer.encode(concept, add_special_tokens=False))

    data = tokenizer.batch_encode_plus(sentences, return_tensors="pt", padding=True).input_ids
    print(data)
    comparison_logit_set = []
    for sentence in data:
        for token in sentence:
            
            prompt = prompt_function(sentence, concept_id, token)
            # make all tensors in the prompt list to string, without trying to decode strings
            str_prompt = [tokenizer.decode(p) if isinstance(p, t.Tensor) else p for p in prompt]
            token_prompt =[tokenizer.encode(p, add_special_tokens=False) if isinstance(p, str) else p for p in str_prompt]
            print(token_prompt)
            token_prompt = [item for sublist in token_prompt for item in sublist]
            attention_mask = t.ones(len(token_prompt), device=device)
            attention_mask[token_prompt == tokenizer.pad_token_id] = 0
            print("".join(str_prompt))
            print("##################")
            with t.no_grad():
                output_comparison = model(
                        input_ids=t.tensor(token_prompt, device=device).unsqueeze(0),
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
            
            comparison_logit_set.append(output_comparison.logits[0, -1, :])

    # create logits through caching process used in Llama_Leaner

    #Create dummy labels required for CachedDataset
    labels = t.zeros_like(data, device=device)

    config = Llama_Leaner.Config()
    config.magic_word = "magic"
    config.batch_size = 1
    config.epochs = 1


    dataset = CachedDataset.CachedDataset(
    model, tokenizer, data, labels, prompt_dict, sentence_cache_device=device
    )
    dataloader = CachedDataset.CachedDataloader(
    dataset, batch_size=config.batch_size, shuffle=True, device=device
    )
    magical_token_vector = t.ones(model.config.vocab_size, device=device) * (- t.inf)
    print("Vocabulary size:", model.config.vocab_size)
    print("Concept ID:", concept_id)
    magical_token_vector[concept_id] = 1.0
    training = Llama_Leaner.Training(config, model, tokenizer, test_mode=True)
    training.magic_token_vector = magical_token_vector

    logs = training.train(dataloader, n_top_tracked_tokens=1, specified_tokens=[concept_id])
    logit_set = []
    print(len(logs.losses["output_logits"]))
    for logits in logs.losses["output_logits"]:
        logit_set.append(logits[0,-1, :])
    print(comparison_logit_set[0].shape)
    print(logit_set[0].shape)
    num_tensors1 = len(logit_set)
    num_tensors2 = len(comparison_logit_set)

    distances = t.zeros((num_tensors1, num_tensors2))
    for i, tensor1 in tqdm(enumerate(logit_set)):
        for j, tensor2 in enumerate(comparison_logit_set):
            distance = t.mean((tensor1.to(device) - tensor2.to(device))**2).detach().cpu().numpy()
            distances[i, j] = float(distance)
    print(distances.shape)
    #take minimum along one dimension
    min_distances = t.min(distances, dim=0)
    #take max along final dimension
    max_distance = t.max(min_distances.values)

    # test if max_distance is zero with tolerance
    assert max_distance < 1e-6
    print("Test passed")
    return True 
#%%
if __name__ == "__main__":
    sentences = ["the dog sat in the deep deep fog","the cat sat on the mat"]
    concept = "animal"
    model_id = "distilgpt2"
    device = "cuda" if t.cuda.is_available() else "cpu"
    cachedDataset_test(sentences, concept, model_id, device)



#%%
if __name__ == "__main__":
    sentences = ["the dog sat in the deep deep fog","the cat sat on the mat"]
    concept = " animal"
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    device = "cuda" if t.cuda.is_available() else "cpu"
    cachedDataset_test(sentences, concept, model_id, device)



    


# %%