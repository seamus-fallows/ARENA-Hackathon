#%%
"""
This script demonstrates the usage of the Meta-Llama-3-8B-Instruct model from the transformers library. 
The model is used to assess if a given word from a text represents a specified concept in the context of the text. 
The script generates prompts and evaluates the model's response for each word in a given sentence.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
).to("cuda")
#%%
concept = "colour"
sentence = "The quick brown fox jumps over the lazy dog."
for word in tokenizer.tokenize(sentence, add_special_tokens=False):
    if word[0] == "Ġ":
        word = word[1:]
    messages = [
        {"role": "system", "content": "Your task is to assess if a given word from some text represents a specified concept in the context of the text. This could be a concept related to the meaning of the word, or its structural usage in the text. Provide a rating based on this assessment:\nIf the word represents the concept, respond with 'Rating: 1'.\nIf the word does not represent the concept, respond with 'Rating: 0'.\nThink carefully about if the concept applies to the word in the context of the text. Be confident."},
        {"role": "user", "content": f"The text is: \"{sentence}\". From this text, is the word \"{word}\" an example of \"{concept}\"?"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    print('####################################')
    print('Full prompt\n')
    print(tokenizer.decode(input_ids[0]))
    print('############################')

    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    tokenizer.convert_tokens_to_ids("1"),
    tokenizer.convert_tokens_to_ids("0")
    ]

    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        eos_token_id=terminators,
        max_new_tokens=10,
        do_sample=True,
        temperature=0.01,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    print(f"Word asked about: {word}")
    print(tokenizer.decode(response))

# %%
print(tokenizer.tokenize(("lizard","reptile","mammal")))
# %%
