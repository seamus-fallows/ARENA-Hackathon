#%%
from transformers import AutoTokenizer

def check_repeated_tokens(sentence, tokenizer):
    tokens = tokenizer.tokenize(sentence)
    token_set = set()
    for token in tokens:
        if token in token_set:
            return True
        token_set.add(token)
    return False

def process_file(filepath, tokenizer):
    with open(filepath, 'r', encoding='utf-8') as file:
        sentences = file.readlines()

    modified = False
    new_sentences = []
    for sentence in sentences:
        if check_repeated_tokens(sentence.strip(), tokenizer):
            new_sentences.append(sentence)
        else:
            modified = True

    if modified:
        with open(filepath, 'w', encoding='utf-8') as file:
            file.writelines(new_sentences)
        print("Some sentences have been removed from the file.")
    else:
        print("No sentences were removed.")

if __name__ == "__main__":
    tokenizer_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    file_path = "/root/ARENA-Hackathon/data/text_data/repeated_words/sentences.txt"
    process_file(file_path, tokenizer)
#%%
def find_repeated_tokens(sentence, tokenizer):
    tokens = tokenizer.tokenize(sentence)
    token_counts = {}
    for token in tokens:
        if token in token_counts:
            token_counts[token] += 1
        else:
            token_counts[token] = 1
    
    # Collect tokens that appear more than once
    repeated_tokens = ['[' + token + ']' for token, count in token_counts.items() if count > 1]
    return repeated_tokens

def print_repeated_tokens(filepath, tokenizer):
    with open(filepath, 'r', encoding='utf-8') as file:
        sentences = file.readlines()

    for sentence in sentences:
        sentence = sentence.strip()
        repeated_tokens = find_repeated_tokens(sentence, tokenizer)
        if repeated_tokens:
            print(f"Sentence: {sentence}")
            print("Repeated Tokens:", ', '.join(repeated_tokens))
            print()  # Add a blank line for better separation

if __name__ == "__main__":
    tokenizer_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    file_path = "/root/ARENA-Hackathon/data/text_data/repeated_words/sentences.txt"
    print_repeated_tokens(file_path, tokenizer)
# %%
def find_repeated_tokens(sentence, tokenizer):
    tokens = tokenizer.tokenize(sentence)
    token_counts = {}
    for token in tokens:
        if token in token_counts:
            token_counts[token] += 1
        else:
            token_counts[token] = 1
    # Collect tokens that appear more than once
    repeated_tokens = {token for token, count in token_counts.items() if count > 1}
    return repeated_tokens

def clean_token(token):
    # Remove the 'Ġ' character and other undesired characters if necessary
    if token.startswith('Ġ'):
        return token[1:]
    return token

def collect_all_repeated_tokens(sentences, tokenizer):
    all_repeated_tokens = set()
    for sentence in sentences:
        repeated_tokens = find_repeated_tokens(sentence, tokenizer)
        cleaned_tokens = {clean_token(token) for token in repeated_tokens}
        all_repeated_tokens.update(cleaned_tokens)
    return all_repeated_tokens

def save_tokens_to_file(filepath, tokens):
    with open(filepath, 'w', encoding='utf-8') as file:
        for token in sorted(tokens):
            file.write(token + '\n')

def process_sentences_and_save_tokens(input_filepath, output_filepath, tokenizer):
    with open(input_filepath, 'r', encoding='utf-8') as file:
        sentences = file.readlines()

    repeated_tokens = collect_all_repeated_tokens(sentences, tokenizer)
    save_tokens_to_file(output_filepath, repeated_tokens)

if __name__ == "__main__":
    tokenizer_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    input_file_path = "/root/ARENA-Hackathon/data/text_data/repeated_words/sentences.txt"
    output_file_path = "/root/ARENA-Hackathon/data/text_data/repeated_words/labelled_words.txt"
    process_sentences_and_save_tokens(input_file_path, output_file_path, tokenizer)
# %%
