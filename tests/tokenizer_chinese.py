from transformers import AutoTokenizer

text = "你会说中文吗？我在想你今天会不会来 现在大地变得无形空虚，黑暗 笼罩着深渊的表面"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokens = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
for token in tokenizer.decode(tokens[0]):
    print(len(token))