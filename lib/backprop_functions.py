import torch as t
import matplotlib.pyplot as plt
import einops
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from transformers import AdamW
import numpy as np


def tokenize_input(tokenizer, input_text: [str, list[str]], magic_word: str):
    if isinstance(input_text, str):
        input_text = [input_text]
    tokens = tokenizer.batch_encode_plus(
        input_text, padding=True, return_tensors="pt"
    ).input_ids
    tokens = t.tensor(tokens)
    magic_ids = tokenizer.encode(magic_word)[0]
    magic_token_pos = t.where(tokens == magic_ids)

    return tokens, magic_token_pos


def create_modified_embeddings(tokens, magic_token_pos, magic_token_vector, model):
    device = model.transformer.wte.weight.device
    tokens = tokens.clone().detach().to(device)
    inputs_embeds = model.transformer.wte.weight[tokens]
    embedding_matrix = model.transformer.wte.weight
    magic_token_embed = einops.einsum(
        embedding_matrix,
        magic_token_vector.to(device),
        " d_vocab d_model, d_vocab -> d_model ",
    )
    if magic_token_pos != None:
        x_tens, y_tens = magic_token_pos
        for x, y in zip(x_tens, y_tens):
            inputs_embeds[x, y] = magic_token_embed
    return inputs_embeds


def intialise_random_token_vector(model):
    """
    Returns a random unit-norm vector of length vocab_size
    """
    device = model.transformer.wte.weight.device
    vocab_size = model.config.vocab_size
    magic_token_vector = t.rand(vocab_size, device=device)
    magic_token_vector /= magic_token_vector.sum()
    magic_token_vector = t.nn.Parameter(magic_token_vector, requires_grad=True)

    return magic_token_vector


def KL_div(logits, target_logit_tensor):
    """
    KL divergence between two logits
    """
    n_comp = target_logit_tensor.shape[0]
    kl_divs = ((logits.repeat(n_comp, 1) - target_logit_tensor) * (logits.exp())).sum(1)

    return kl_divs.mean()


def Loss_function(
    logits,
    target_tokens,
    magic_token_vector,
    magic_word_pos,
    accuracy_lambda=1.0,
    l1_lambda=0.01,
    kl_lambda=0.01,
):
    """
    Loss function for the magic token vector
    """
    accuracy_loss = accuracy_lambda * t.nn.functional.cross_entropy(
        logits[:, -1, :], target_tokens, reduction="mean"
    )
    l1_loss = l1_lambda * t.norm(magic_token_vector, 1)
    logits_on_magic_pos = logits[magic_word_pos[0], magic_word_pos[1] - 1, :]
    logprobs_on_magic_pos = logits_on_magic_pos.log_softmax(1)

    loss_kl = kl_lambda * KL_div(magic_token_vector, logprobs_on_magic_pos)
    loss = accuracy_loss + l1_loss + loss_kl
    return loss, accuracy_loss, l1_loss, loss_kl


def train_token_vector(
    model,
    tokens,
    magic_word_pos,
    target_token_ids,
    magic_token_vector,
    lr=0.01,
    epochs=500,
    accuracy_lambda=1.0,
    l1_lambda=0.01,
    kl_lambda=0.01,
    logging_ids=[],
    n_top_log=10,
):
    """
    Perform gradient descent on the magic_token_vector which loss function given by cross-entopy
    between predicted last token and target_token
    """
    loss_values = []
    accuracy_loss_values = []
    l1_loss_values = []
    kl_loss_values = []
    device = model.transformer.wte.weight.device

    if isinstance(target_token_ids, int):
        target_token_ids = [target_token_ids]

    target_tokens = t.zeros(len(target_token_ids), model.config.vocab_size).to(device)
    for i, id in enumerate(target_token_ids):
        target_tokens[i, id] = 1.0

    optimizer = AdamW([magic_token_vector], lr=lr)
    ids_logit_logs = {id: [] for id in logging_ids}

    with tqdm(total=epochs, desc="Training Progress") as pbar:
        for epoch in range(epochs):
            optimizer.zero_grad()
            embeddings = create_modified_embeddings(
                tokens, magic_word_pos, magic_token_vector, model
            )
            outputs = model.forward(inputs_embeds=embeddings)
            logits = outputs.logits

            loss, accuracy_loss, l1_loss, kl_loss = Loss_function(
                logits,
                target_tokens,
                magic_token_vector,
                magic_word_pos,
                accuracy_lambda=accuracy_lambda,
                l1_lambda=l1_lambda,
                kl_lambda=kl_lambda,
            )

            loss.backward()
            optimizer.step()

            # magic_token_vector /= magic_token_vector.norm()
            # keep all entries positive

            loss_values.append(loss.item())
            accuracy_loss_values.append(accuracy_loss.item())
            l1_loss_values.append(l1_loss.item())
            kl_loss_values.append(kl_loss.item())
            pbar.set_postfix(
                {
                    "Loss": loss.item(),
                    "Accuracy Loss": accuracy_loss.item(),
                    "L1 Loss": l1_loss.item(),
                    "KL Loss": kl_loss.item(),
                }
            )
            pbar.update(1)
            top_tokens = t.argsort(magic_token_vector)[-n_top_log:].tolist()

            for id in logging_ids + top_tokens:
                if id not in ids_logit_logs.keys():
                    ids_logit_logs[id] = []
                ids_logit_logs[id].append((epoch, magic_token_vector[id].item()))

    return (
        dict(
            loss=loss_values,
            accuracy_loss=accuracy_loss_values,
            l1_loss=l1_loss_values,
            kl_loss=kl_loss_values,
        ),
        ids_logit_logs,
    )


def plot_loss(loss_dict):
    """
    Plot the loss values from the training
    """
    plt.figure(figsize=(10, 5))
    plt.plot(loss_dict["loss"], label="Total Loss")
    plt.plot(loss_dict["accuracy_loss"], label="Accuracy Loss")
    plt.plot(loss_dict["l1_loss"], label="L1 Loss")
    plt.plot(loss_dict["kl_loss"], label="KL Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_best_tokens(ids_logit_logs, tokenizer, n_plots=5):
    """
    Plot the logit values for the top_n tokens
    """
    plt.figure(figsize=(10, 5))
    # make a color list of the length of n_plots
    colors = plt.cm.rainbow(np.linspace(0, 1, n_plots))
    plot_dict = {}
    max_probs = []
    for word in ids_logit_logs.keys():
        steps = []
        probs = []

        for step, prob in ids_logit_logs[word]:
            steps.append(step)
            probs.append(prob)

        max_prob = max(probs)
        max_probs.append(max_prob)
        plot_dict[word] = (steps, probs, max_prob)

    plot_keys = sorted(plot_dict.keys(), key=lambda x: plot_dict[x][2], reverse=True)[
        :n_plots
    ]
    for word, color in zip(plot_keys, colors):
        steps, probs, max_prob = plot_dict[word]
        step_sections = []
        prob_sections = []

        step_section = []
        prob_section = []

        last_step = -1
        for step, prob in ids_logit_logs[word]:
            if step != last_step + 1:
                step_sections.append(step_section)
                prob_sections.append(prob_section)
                step_section = []
                prob_section = []
            step_section.append(step)
            prob_section.append(prob)
            last_step = step

        for step_section, prob_section in zip(step_sections, prob_sections):
            plt.plot(step_section, prob_section, color=color)
        plt.plot([], [], label=tokenizer.decode([word], color=color))

        plt.plot(steps, probs, color=color, label=tokenizer.decode([word]))

    plt.legend()
    plt.show()
