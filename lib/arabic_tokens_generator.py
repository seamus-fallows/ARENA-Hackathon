#%%
from transformers import AutoTokenizer
import re
# Example usage
input_text = """
لطالما حلمت بالسفر إلى دول مختلفة حول العالم.
هوايتي المفضلة هي قراءة كتب لمؤلفين يتصدرون قوائم الأكثر مبيعاً
يجب أن يكون الجميع نباتيين """

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenized_text = tokenizer.encode(input_text, add_special_tokens=False)
unique_tokens = set(tokenized_text)

# open text file
with open("/root/ARENA-Hackathon/data/text_data/arabic/labled_words.txt", "w", encoding="utf-8") as file:
    for token in unique_tokens:
        token_string = tokenizer.decode(token,add_prefix_space=False)
        filtered_token_string = re.sub(r'[^\w]', '', token_string)
        if filtered_token_string:  # write only non-empty strings
            file.write(filtered_token_string + "\n")



# %%
