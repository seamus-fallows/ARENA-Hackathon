from transformers import AutoTokenizer, AutoModelForCausalLM
token="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"

tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", use_auth_token=token, ignore_mismatched_sizes=True
        )
model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", use_auth_token=token
        )