from transformers import AutoTokenizer

def check_tokenization(file_path):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    # Read the file with the words
    try:
        with open(file_path, 'r') as file:
            words = file.read().splitlines()
    except Exception as e:
        print(f"Failed to read the file: {e}")
        return

    # Check tokenization of each word
    all_single_token = True
    for word in words:
        word_plus_space = ' ' + word
        tokens = tokenizer.tokenize(word_plus_space)
        if len(tokens) != 1:
            print(f"Word '{word_plus_space}' is tokenized into multiple tokens: {tokens}")
            all_single_token = False

    if all_single_token:
        print("All words are tokenized as single tokens.")
    else:
        print("Some words are not tokenized as single tokens.")

if __name__ == "__main__":
    # Path to the file
    file_path = "/root/ARENA-Hackathon/data/text_data/animals/labled_words.txt"
    check_tokenization(file_path)
