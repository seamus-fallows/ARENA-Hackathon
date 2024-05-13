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
concept = " Chinese"
sentence = "他一到，我们就开始 The cat sat in the hat."
token_ids = tokenizer.encode(sentence, add_special_tokens=False)
tokens = [tokenizer.decode(token) for token in token_ids]
print(tokens)

#%%
for word in tokens:
    if word[0] == "Ġ":
        word = word[1:]
    messages = [
        {"role": "system", "content": """Your task is to assess if a given word from some text is an example of or represents particular concept. You will be given the information in the following format:
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
Answer: Yes"""}
        ,
        {"role": "user", "content": f"""Text: {sentence}
Word: {word}
Concept:{concept}
Answer:"""}
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
print(tokenizer.encode("到"))
print(tokenizer.encode("我们"))

# %%
