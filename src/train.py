import argparse
from log import (
    log,
    important,
    warning,
    pretty_format,
    set_status_state,
    ProgressStatus,
    LogTypes,
)
from dataset import Dataset
from model import Model
from discriminator import Discriminator
from config import Configuration
# from database_memory import update as update_memory
import common
from prepare_data import (
    SpecialCharacters,
    vocab_size,
    split_to_characters,
    words_to_lookup,
    set_substeps,
)
from plot import plot_simple_array
from predict import predict

import torch
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torch import nn, optim
from torch.utils.data import DataLoader
from statistics import mean, median
from itertools import tee
import os
import numpy as np
import math
import json

def log10(n):
    if n is None:
        return float('NaN')
    n = abs(n)
    if n <= 0:
        return float('NaN')
    return math.log10(n)

def round_log10(n):
    return round(log10(n), 2)

device = "cuda"

important("VAUVADESCENT")
important("Parsing args")

parser = argparse.ArgumentParser()
parser.add_argument("--max-epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--max-lr", type=float, default=0.001)
parser.add_argument("--save-margin", type=float, default=0.1)

args = parser.parse_args()
log(pretty_format(args.__dict__))
config = Configuration(args)

state = {
    "lr_step": -1,
    "loss_history": [],
    "loss_history_real": [],
    "loss_history_discriminator": [],
}
if os.path.isfile("../state.json"):
    with open("../state.json", "r") as f:
        state = json.load(f)

dataset = Dataset(config)
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
)
model = Model(config).to(device)
discriminator = Discriminator(config).to(device)
model.train()

momentum = 0.9
momentum_max = 0.5
momentum_d = 0.9
criterion = nn.CrossEntropyLoss(reduction="mean")
lr_start_div_factor = 10
optimizer = optim.Adam(
    model.parameters(), lr=config.max_lr / lr_start_div_factor, betas=(momentum, momentum)
)
optimizer_d = optim.Adam(
    discriminator.parameters(),
    lr=config.max_lr / lr_start_div_factor,
    betas=(momentum_d, 0.9),
)
total_steps = 100_000 * config.max_length_of_topic
pct_start = 0.00001
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=config.max_lr,
    div_factor=lr_start_div_factor,
    total_steps=total_steps,
    pct_start=pct_start,
    three_phase=True,
    anneal_strategy="linear",
    base_momentum=momentum,
    max_momentum=momentum_max,
    final_div_factor=1e1000,
)
scheduler_d = torch.optim.lr_scheduler.OneCycleLR(
    optimizer_d,
    max_lr=config.max_lr,
    div_factor=lr_start_div_factor,
    total_steps=total_steps,
    pct_start=pct_start,
    three_phase=False,
    anneal_strategy="linear",
    base_momentum=momentum_d,
    max_momentum=momentum_d,
    final_div_factor=1e1,
)

reset_optimizer = False
previous_model_loss = None
if os.path.isfile("../trained_model"):
    trained_model = torch.load("../trained_model", map_location=torch.device(device))

    previous_model_loss = trained_model["model_loss"]
    important(f"Previous model loss: {round(log10(previous_model_loss), 2)}") 
    model.load_state_dict(trained_model["model"])
    if not reset_optimizer:
        optimizer.load_state_dict(trained_model["model_optimizer"])
        scheduler.load_state_dict(trained_model["model_scheduler"])

    # discriminator.load_state_dict(trained_model["discriminator"])
    # if not reset_optimizer:
    #     optimizer_d.load_state_dict(trained_model["discriminator_optimizer"])
    #     scheduler_d.load_state_dict(trained_model["discriminator_scheduler"])

# print(model)

def pack_loss_history(loss_history):
    new_list = []
    for i in range(0, (len(loss_history) // 2 * 2), 2):
        current = loss_history[i]
        nxt = loss_history[i + 1]

        if current[1] <= nxt[1]:
            new_list.append([(current[0] + nxt[0]) / 2, current[1] + nxt[1]])
        else:
            new_list.append(current)
            new_list.append(nxt)

    return new_list


important("Starting training")
input_text = (
    "Auttakaa! Mitä teen, kun mun naapurit ei lopeta ees yöllä ja en saa nukuttua"
)
loss_history = state["loss_history"]
loss_history_real = state["loss_history_real"]
loss_history_discriminator = state["loss_history_discriminator"]
set_status_state(ProgressStatus(args.max_epochs))
grad_mean = 0
last_epoch_loss = []
has_printed_grad = False
for epoch in range(args.max_epochs):
    # last_epoch_loss = last_epoch_loss[-100_000:]
    dataset.load_batches()
    state_h, state_c = model.init_state(config.batch_size)
    state_h, state_c = state_h.to(device), state_c.to(device)
    # d_state_h, d_state_c = discriminator.init_state(config.batch_size)
    # d_state_h, d_state_c = d_state_h.to(device), d_state_c.to(device)

    epoch_losses = []
    if epoch >= 10:
        loss_diff = mean((last_epoch_loss)[-10_000:]) - (previous_model_loss or 999_999)
        if loss_diff <= 0:
            warning("Model is now better, stopping")
            break

    set_substeps(len(dataloader))
    for batch, (x, c, y) in enumerate(dataloader):
        loss_div = 1
        x = x.to(device)
        y = y.to(device)
        c = c.to(device)

        optimizer.zero_grad()
        # optimizer_d.zero_grad()
        # disc_real, (d_state_h, d_state_c) = discriminator(
        #     (y, c), (d_state_h, d_state_c)
        # )
        # real_labes = torch.ones((y.shape[0], y.shape[1], 1)).to(device)
        # disc_real_loss = criterion(disc_real, real_labes)
        # d_state_h = d_state_h.detach()
        # d_state_c = d_state_c.detach()
        # (disc_real_loss / loss_div).backward()
        # # optimizer_d.step()
        # # optimizer_d.zero_grad()
        # y_pred, (_) = model((x, c), (state_h, state_c))
        # fake_labels = torch.zeros((y.shape[0], y.shape[1], 1)).to(device)
        # disc_fake, (_) = discriminator((y_pred, c), (d_state_h, d_state_c))
        # fake_factor = 0.3 / max(disc_real_loss.item(), 0.3)
        # # fake_factor = 1.0
        # disc_fake_loss = criterion(disc_fake, fake_labels) * (
        #     1.0 / max(disc_real_loss.item(), 1.0)
        # )
        # ((disc_fake_loss * fake_factor) / loss_div).backward()
        # optimizer_d.step()

        y_pred, (state_h, state_c) = model((x, c), (state_h, state_c))
        # y_pred = torch.nn.functional.softmax(y_pred, dim=-1)
        # disc_pred, (_) = discriminator((y_pred, c), (d_state_h, d_state_c))
        real_labes = torch.ones((y.shape[0], y.shape[1], 1)).to(device)
        # loss_d = criterion(disc_pred, real_labes)
        # loss_d_factor = 0.5 / max(disc_real_loss.item(), 0.5)

        # pos_weight CBELOSS tai tää
        # loss_r = sigmoid_focal_loss(y_pred, y, reduction="mean")
        # loss_r = sigmoid_focal_loss(y_pred, y, reduction="sum") + criterion(positive_y_pred, torch.ones(positive_y_pred.shape, device="cuda"))
        
        # positive_y_pred = torch.sum(y_pred * y, dim=-1)
        # loss_r = criterion(positive_y_pred, torch.ones(positive_y_pred.shape, device="cuda"))

        loss_r = criterion(y_pred.view(-1, vocab_size), y.view(-1, vocab_size))

        # loss_d_scaled = loss_d * loss_d_factor * loss_r * 0.1
        loss = loss_r # + loss_d_scaled
        # loss = loss_d_scaled if loss_r.item() < 10.0 else loss_r + loss_d_scaled
        # loss = loss_d_scaled

        loss_history.append([loss.item(), 1])
        loss_history_real.append([loss_r.item(), 1])
        # loss_history_discriminator.append([loss_d_scaled.item(), 1])

        state_h = state_h.detach()
        state_c = state_c.detach()
        (loss / loss_div).backward()
        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

        # if ((batch + 1) % 4 == 0):
        if not has_printed_grad and batch > 20:
            has_printed_grad = True
            for name, param in model.named_parameters():
                log(f"{name}: {round_log10(torch.mean(torch.abs(param.grad)))}, {round_log10(torch.max(param.grad) - torch.min(param.grad))}", LogTypes.WARNING)
        optimizer.step()
        scheduler.step()
        
        loss_diff1 = mean(([previous_model_loss or 999_999] + last_epoch_loss)[-10_000:])
        loss_diff =  loss_diff1 - (previous_model_loss or 999_999) * (1 + config.save_margin)
        isBetter = loss_diff < 0
        log(
            f"{epoch}:{batch} - loss: {round(log10(loss.item()), 2)}, lr: {round(log10(scheduler.get_last_lr()[-1]), 3)}, better: {isBetter} ({round(round_log10(loss_diff1) - round_log10(previous_model_loss), 2)})",
            repeating_status=True,
            substep=True,
        )
        last_epoch_loss.append(loss.item())
        epoch_losses.append(round(math.log10(loss.item() + 1e-11), 2))
    # update_memory()
    log(
        "",
        repeating_status=True,
    )
    state["lr_step"] += 1
    # scheduler_d.step()
    state["loss_history"] = loss_history
    state["loss_history_real"] = loss_history_real
    state["loss_history_discriminator"] = loss_history_discriminator
    with open("../state.json", "w") as f:
        json.dump(state, f)
    # torch.save(model.state_dict(), "../trained_model")
    loss_history = pack_loss_history(loss_history)
    loss_history_real = pack_loss_history(loss_history_real)
    loss_history_discriminator = pack_loss_history(loss_history_discriminator)
    # plot_simple_array([round(math.log10(x[0]), 2) for x in loss_history], "../loss_history.png")
    # log(predict(model, input_text), multiline=True, type=LogTypes.DATA)
    model.train()

last_epoch_loss = mean(last_epoch_loss[-10_000:])

if previous_model_loss is None or last_epoch_loss <= previous_model_loss * (1 + config.save_margin):
    important("Saving model")
    trained_model = {
        "model_loss": last_epoch_loss,

        "model": model.state_dict(),
        "model_optimizer": optimizer.state_dict(),
        "model_scheduler": scheduler.state_dict(),
        # "discriminator": discriminator.state_dict(),
        # "discriminator_optimizer": optimizer_d.state_dict(),
        # "discriminator_scheduler": scheduler_d.state_dict(),
    }
    torch.save(trained_model, "../trained_model")
else:
    warning(f"New model worse, skipping save {round_log10(previous_model_loss)} < {round_log10(last_epoch_loss)}")
plot_simple_array(
    [
        [round(log10(x[0]), 2) for x in loss_history]
    ],
    "../loss_history.png",
)
# model.db_memory.save_db()
log(predict(model, device, config, input_text), multiline=True, type=LogTypes.DATA)
important("Done")