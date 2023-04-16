import argparse
from log import (
    log,
    important,
    pretty_format,
    set_status_state,
    ProgressStatus,
    LogTypes,
)
from dataset import Dataset
from model import Model
from discriminator import Discriminator
from config import Configuration
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
from torch import nn, optim
from torch.utils.data import DataLoader
from statistics import mean, median
from itertools import tee
import os
import numpy as np
import math
import json

device = "cpu"

important("VAUVADESCENT")
important("Parsing args")

parser = argparse.ArgumentParser()
parser.add_argument("--max-epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--max-lr", type=float, default=0.001)

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

momentum = 0.1
momentum_d = 0.4
criterion = nn.BCELoss(reduction="mean")
lr_start_div_factor = 10
optimizer = optim.Adam(
    model.parameters(), lr=config.max_lr / lr_start_div_factor, betas=(momentum, 0.9)
)
optimizer_d = optim.Adam(
    discriminator.parameters(),
    lr=config.max_lr / lr_start_div_factor,
    betas=(momentum_d, 0.9),
)
total_steps = 10_000
pct_start = 0.004
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=config.max_lr,
    div_factor=lr_start_div_factor,
    total_steps=total_steps,
    pct_start=pct_start,
    three_phase=False,
    anneal_strategy="linear",
    base_momentum=momentum,
    max_momentum=momentum,
    final_div_factor=1e9,
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
    final_div_factor=1e9,
)

if os.path.isfile("../trained_model"):
    trained_model = torch.load("../trained_model", map_location=torch.device(device))
    model.load_state_dict(trained_model["model"])
    optimizer.load_state_dict(trained_model["model_optimizer"])
    scheduler.load_state_dict(trained_model["model_scheduler"])
    discriminator.load_state_dict(trained_model["discriminator"])
    optimizer_d.load_state_dict(trained_model["discriminator_optimizer"])
    scheduler_d.load_state_dict(trained_model["discriminator_scheduler"])


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
for epoch in range(args.max_epochs):
    dataset.load_batches()
    state_h, state_c = model.init_state(config.sequence_length)
    state_h, state_c = state_h.to(device), state_c.to(device)
    d_state_h, d_state_c = discriminator.init_state(config.sequence_length)
    epoch_losses = []

    set_substeps(len(dataloader))
    for batch, (x, c, y) in enumerate(dataloader):
        optimizer_d.zero_grad()
        disc_real, (d_state_h, d_state_c) = discriminator(
            (y.to(device), c), (d_state_h, d_state_c)
        )
        real_labes = torch.ones((y.shape[0], y.shape[1], 1))
        disc_real_loss = criterion(disc_real, real_labes)
        d_state_h = d_state_h.detach()
        d_state_c = d_state_c.detach()
        disc_real_loss.backward()
        # optimizer_d.step()
        # optimizer_d.zero_grad()
        y_pred, (_) = model((x.to(device), c.to(device)), (state_h, state_c))
        fake_labels = torch.zeros((y.shape[0], y.shape[1], 1))
        disc_fake, (_) = discriminator((y_pred, c), (d_state_h, d_state_c))
        disc_fake_loss = criterion(disc_fake, fake_labels) * (
            1.0 / max(disc_real_loss.item(), 1.0)
        )
        disc_fake_loss.backward()
        optimizer_d.step()

        optimizer.zero_grad()
        y_pred, (state_h, state_c) = model(
            (x.to(device), c.to(device)), (state_h, state_c)
        )
        disc_pred, (_) = discriminator((y_pred, c), (d_state_h, d_state_c))
        real_labes = torch.ones((y.shape[0], y.shape[1], 1))
        loss_d = criterion(disc_pred, real_labes)
        loss_d_factor = 1.0 / max(disc_real_loss.item(), 1.0)
        loss_d_scaled = loss_d * loss_d_factor
        y = y.to(device)
        loss_r = vocab_size * criterion(y_pred, y)
        loss = loss_d_scaled + loss_r
        # loss = loss_d_scaled if loss_r.item() < 10.0 else loss_r + loss_d_scaled
        # loss = loss_d_scaled

        loss_history.append([loss.item(), 1])
        loss_history_real.append([loss_r.item(), 1])
        loss_history_discriminator.append([loss_d_scaled.item(), 1])

        state_h = state_h.detach()
        state_c = state_c.detach()
        loss.backward()
        optimizer.step()
        log(
            f"{epoch}:{batch} - loss: {round(loss.item(), 2)} ({round(loss.item() - loss_d_scaled.item(), 2)} + {round(loss_d_scaled.item(), 2)}) - loss real: {round(disc_real_loss.item(), 2)}, - loss fake: {round(disc_fake_loss.item(), 2)}, lr: {round(math.log10(scheduler.get_last_lr()[0]), 3)}",
            repeating_status=True,
            substep=True,
        )
        epoch_losses.append(round(math.log10(loss.item() + 1e-6), 2))
    log(
        "",
        repeating_status=True,
    )
    state["lr_step"] += 1
    scheduler_d.step()
    scheduler.step()
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

trained_model = {
    "model": model.state_dict(),
    "model_optimizer": optimizer.state_dict(),
    "model_scheduler": scheduler.state_dict(),
    "discriminator": discriminator.state_dict(),
    "discriminator_optimizer": optimizer_d.state_dict(),
    "discriminator_scheduler": scheduler_d.state_dict(),
}
torch.save(trained_model, "../trained_model")
plot_simple_array(
    [
        [round(math.log10(x[0] + 0.01), 2) for x in loss_history],
        [round(math.log10(x[0] + 0.01), 2) for x in loss_history_real],
        [round(math.log10(x[0] + 0.01), 2) for x in loss_history_discriminator],
    ],
    "../loss_history.png",
)
log(predict(model, device, config, input_text), multiline=True, type=LogTypes.DATA)
important("Done"),
