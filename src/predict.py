from log import (
    log,
    important,
    pretty_format,
    set_status_state,
    ProgressStatus,
    LogTypes,
)
import argparse
import common
from model import Model
from config import Configuration
from prepare_data import (
    SpecialCharacters,
    vocab_size,
    split_to_characters,
    words_to_lookup,
    set_substeps,
)
import torch
from torch import nn, optim
import os
import numpy as np
import math


def predict(
    model,
    device,
    config,
    input_text,
    split_to_posts=False,
    max_predict_chars=400,
    temperature=1.0,
):
    important("Predicting")
    vocab = common.load_json_file(os.path.join(common.cache_dir, "characters.json"))
    words = vocab["words"]
    word_lookup = words_to_lookup(words)
    char_to_index = vocab["char_to_index"]
    index_to_char = vocab["index_to_char"]

    def logits_to_index(t):
        arr = torch.nn.functional.softmax(t / temperature, dim=-1).detach().cpu().numpy()
        # arr = arr - np.min(arr)
        # arr = np.power(arr, 1.0 / randomess)
        # normalized = np.divide(arr, np.sum(arr))
        # for i, n in enumerate(normalized):
        #     normalized[i] = n if not math.isnan(n) else 0.0001
        # normalized = np.divide(normalized, np.sum(normalized))
        return np.random.choice([i for i in range(vocab_size)], p=arr)

    def is_new_post(i):
        return index_to_char.get(str(i), "#") == SpecialCharacters.NEW_POST.value

    def index_to_char_fun(i):
        char = index_to_char.get(str(i), "#")
        if char == SpecialCharacters.NEW_POST.value:
            if split_to_posts:
                return SpecialCharacters.NEW_POST
            else:
                return "\n\nPostaus:\n"
        elif char == "\n":
            return " "
        else:
            return char

    all_input = [char_to_index[c] for c in split_to_characters(word_lookup, input_text)]
    input_prev = all_input[config.context_length :]
    input = torch.from_numpy(np.array(input_prev, np.int32)).unsqueeze(0).to(
        device
    ), torch.from_numpy(
        np.array(
            [
                [
                    [
                        all_input[-1 - i - ii] if i + ii < len(all_input) else 0
                        for ii in range(0, config.context_length)
                    ]
                    for i in range(len(input_prev))
                ]
            ],
            np.int32,
        )
    ).to(
        device
    )
    model.eval()
    state_h, state_c = model.init_state(1)

    max_character_amount = max_predict_chars
    min_character_amount = max_predict_chars - 200
    set_status_state(ProgressStatus(max_character_amount))
    with torch.no_grad():
        pred_indices = all_input
        pred, (state_h, state_c) = model(
            input, (state_h.to(device), state_c.to(device))
        )

        pred = pred[:, -1:, :]
        pred_indices.append(logits_to_index(pred[0, 0, :]))

        for i in range(max_character_amount):
            log("predicting...", repeating_status=True)
            state_c = state_c[:, -1:, :]
            state_h = state_h[:, -1:, :]
            pred, (state_h, state_c) = model(
                (
                    torch.from_numpy(np.array([[pred_indices[-1]]], np.int32)).to(
                        device
                    ),
                    torch.from_numpy(
                        np.array([[pred_indices[-config.context_length :]]])
                    ).to(device),
                ),
                (state_h.contiguous().to(device), state_c.contiguous().to(device)),
            )

            predicted_index = logits_to_index(pred[0, 0, :])
            if i >= min_character_amount and is_new_post(predicted_index):
                break
            pred_indices.append(predicted_index)

    mapped = [index_to_char_fun(i) for i in pred_indices]
    if not split_to_posts:
        return "".join(mapped)
    else:
        posts = []
        current_post = []
        for c in mapped:
            if c == SpecialCharacters.NEW_POST:
                posts.append("".join(current_post))
                current_post = []
            else:
                current_post.append(c)
        posts.append("".join(current_post))
        return posts


def initialize_for_predict():
    device = "cpu"
    important("VAUVADESCENT")
    important("Parsing args")
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    log(pretty_format(args.__dict__))
    config = Configuration(args)
    model = Model(config).to(device)
    if os.path.isfile("../trained_model"):
        trained_model = torch.load(
            "../trained_model", map_location=torch.device(device)
        )
        model.load_state_dict(trained_model["model"])
    return model, device, config


if __name__ == "__main__":
    model, device, config = initialize_for_predict()
    model.eval()
    input_text = (
        "Auttakaa! Mitä teen, kun mun naapurit ei lopeta ees yöllä ja en saa nukuttua"
    )
    log(predict(model, device, config, input_text), multiline=True, type=LogTypes.DATA)
