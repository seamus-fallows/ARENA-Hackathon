#%%
from transformers import AutoTokenizer
import re

def is_chinese(text):
    # Regular expression to detect Chinese characters
    return re.search("[\u4e00-\u9fff]", text)

def extract_chinese_text(filename):
    chinese_text = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if is_chinese(line):
                chinese_text.append(line.strip())
    return chinese_text

# Example usage
filename = '/root/ARENA-Hackathon/data/text_data/chinese/sentences.txt'
chinese_text = extract_chinese_text(filename)
print(chinese_text)
#%%
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
chinese_sentence_tokens = tokenizer.batch_encode_plus(chinese_text, add_special_tokens=False)['input_ids']
# set of unique tokens
unique_tokens = set([token for sentence in chinese_sentence_tokens for token in sentence])
# save unique tokens to file 
for token in unique_tokens:
    print(tokenizer.decode(token))
with open('/root/ARENA-Hackathon/data/text_data/chinese/label_tokens.txt', 'w') as file:
    for token in unique_tokens:
        file.write(f"{token}\n")

# %%
