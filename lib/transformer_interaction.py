def create_prompt(systtem_prompt, user_promt, answer=""):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    prompt = B_INST + B_SYS + systtem_prompt + E_SYS + user_promt + E_INST + answer

    return prompt
