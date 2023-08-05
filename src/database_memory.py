import torch
from torch import nn
import numpy as np
import os

# import sqlite3
# con = sqlite3.connect("../trained.db")

import log

max_batches = 64 # could be from config
features = 16
slots = 10_000
short_slots = 2
slot_size = 512

short_memory = np.zeros((
    max_batches,
    short_slots,
    slot_size,
))

memory = np.zeros((
    slots,
    slot_size,
))

update_memory = np.copy(memory)

def update():
    global memory, update_memory, short_memory
    memory = np.copy(update_memory)
    short_memory = np.zeros((
        max_batches,
        short_slots,
        slot_size,
    ))

if os.path.isfile("./memory"):
    memory = np.load("./memory", allow_pickle=True)
    log.log(f"memory range {np.max(memory) - np.min(memory)}", type=log.LogTypes.DATA)
    log.log(f"memory mean {np.mean(memory)}", type=log.LogTypes.DATA)
    log.log(f"memory var {np.var(memory)}", type=log.LogTypes.DATA)
    update_memory = np.copy(memory)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_uniform_(m.weight.data, 0.1)
    elif classname.find("Linear") != -1:
        nn.init.xavier_uniform_(m.weight.data, 0.1)
    elif classname.find("LSTM") != -1:
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param.data, 1.0)
            elif "weight_hh" in name:
                torch.nn.init.xavier_uniform_(param.data, 1.0)
            elif "bias" in name:
                param.data.fill_(0)

class FastAttention(nn.Module):
    def __init__(self, config, in_channels):
        super(FastAttention, self).__init__()
        self.config = config

        self.main = nn.Sequential(
            nn.Linear(in_channels, features),
            nn.Softmax(dim=1),
            nn.Linear(features, in_channels)
        )

        self.final = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(True),
            nn.LayerNorm(in_channels),
        )

        self.apply(weights_init)

    def forward(self, inputs):
        return self.final(torch.cat((inputs, inputs * self.main(inputs)), dim=1))
    

class MemoryAccess(nn.Module):
    def __init__(self, config, in_channels, available_slots, read_slots=3, use_short=False):
        super(MemoryAccess, self).__init__()
        self.config = config
        self.read_slots = read_slots
        self.available_slots = available_slots
        self.use_short = use_short

        self.read_weights = nn.Sequential(
            FastAttention(config, in_channels),
            nn.Linear(in_channels, self.read_slots * available_slots),
            nn.Softmax(dim=1)
        )

        self.update_weights = nn.Sequential(
            FastAttention(config, in_channels),
            nn.Linear(in_channels, self.read_slots),
            nn.Sigmoid()
        )

        self.update_memory = nn.Sequential(
            FastAttention(config, in_channels + slot_size),
            nn.Linear(in_channels + slot_size, slot_size),
            nn.ReLU(True)
        )

        self.apply_memory = nn.Sequential(
            FastAttention(config, slot_size * 2),
            nn.Linear(slot_size * 2, slot_size),
            nn.ReLU(True)
        )

        self.update_weights_random = nn.Sequential(
            FastAttention(config, in_channels),
            nn.Linear(in_channels, self.read_slots),
            nn.Sigmoid()
        )

        self.update_memory_random = nn.Sequential(
            FastAttention(config, in_channels + slot_size),
            nn.Linear(in_channels + slot_size, slot_size),
            nn.ReLU(True)
        )

        self.apply_memory_random = nn.Sequential(
            FastAttention(config, slot_size * 2),
            nn.Linear(slot_size * 2, slot_size),
            nn.ReLU(True)
        )

        if self.use_short:
            self.update_weights_short = nn.Sequential(
                FastAttention(config, in_channels),
                nn.Linear(in_channels, short_slots),
                nn.Sigmoid()
            )

            self.update_short_memory = nn.Sequential(
                FastAttention(config, in_channels + slot_size),
                nn.Linear(in_channels + slot_size, slot_size),
                nn.ReLU(True)
            )

            self.apply_short_memory = nn.Sequential(
                FastAttention(config, slot_size * 2),
                nn.Linear(slot_size * 2, slot_size),
                nn.ReLU(True)
            )

        self.apply(weights_init)

    def load_memory(self, indices):
        slots = []
        for i_batch in range(len(indices)):
            batch_slots = []
            for _, i in enumerate(indices[i_batch]):
                batch_slots.append(torch.tensor(memory[i], dtype=torch.float, device="cuda", requires_grad=False))
            slots.append(torch.stack(batch_slots, dim=0))
        return torch.stack(slots, dim=0).reshape((len(indices), -1, slot_size)).to("cuda")

    def load_short_memory(self, batches):
        slots = []
        for i_batch in range(batches):
            batch_slots = []
            for i in range(short_slots):
                batch_slots.append(torch.tensor(short_memory[i], dtype=torch.float, device="cuda", requires_grad=False))
            slots.append(torch.stack(batch_slots, dim=0))
        return torch.stack(slots, dim=0).reshape((batches, -1, slot_size)).to("cuda")

    def select_weights(self, indices, weights):
        w = []
        for i_batch in range(len(indices)):
            batch_w = []
            for i_slot, _i in enumerate(indices[i_batch]):
                batch_w.append(weights[i_batch, i_slot])
            w.append(torch.stack(batch_w, dim=0))
        return torch.stack(w, dim=0).reshape((len(indices), -1)).to("cuda")

    def forward(self, inputs):
        global memory, update_memory
        batch_size = inputs.shape[0]
        read = nn.functional.softmax(self.read_weights(inputs).reshape((batch_size, -1, self.available_slots)), dim=2)
        read_weights, read_indices = torch.max(read, dim=2)
        update_weights = self.update_weights(inputs)
        update_weights_random = self.update_weights_random(inputs)
        indices = read_indices.cpu().numpy()
        read_memory = []

        random_indices = torch.randint(0, self.available_slots, read.shape[:-1]).cpu().numpy()
        update_factor = 0.1

        m = torch.zeros((batch_size, slot_size, ), device="cuda", requires_grad=False)
        mem = self.load_memory(indices)
        uw = update_weights
        rw = self.select_weights(read_indices, read_weights)

        mem_random = self.load_memory(random_indices)
        uw_random = self.select_weights(random_indices, update_weights_random)
        rw_random = self.select_weights(random_indices, read_weights)

        for slot in range(0, self.read_slots):
            r = mem[:,slot]
            r = self.update_memory(torch.cat((r, inputs), dim=1)) * uw[:, slot].view((-1, 1)) + r * (torch.ones(uw[:, slot].shape, device="cuda") - uw[:, slot]).view((-1, 1))

            update = r.detach().cpu().numpy()
            for i_batch in range(len(indices)):
                i = indices[i_batch, slot]
                for update_i, update_value in enumerate(update[i_batch]):
                    update_memory[i][update_i] = update_factor * update_value + (1.0 - update_factor) * update_memory[i][update_i]

            m = m + self.apply_memory(torch.cat((r, m), dim=1)) * rw[:, slot].view((-1, 1))

            r = mem_random[:,slot]
            r = self.update_memory_random(torch.cat((r, inputs), dim=1)) * uw_random[:, slot].view((-1, 1)) + r * (torch.ones(uw_random[:, slot].shape, device="cuda") - uw_random[:, slot]).view((-1, 1))

            update = r.detach().cpu().numpy()
            for i_batch in range(len(indices)):
                i = random_indices[i_batch, slot]
                for update_i, update_value in enumerate(update[i_batch]):
                    update_memory[i][update_i] = update_factor * update_value + (1.0 - update_factor) * update_memory[i][update_i]

            m = m + self.apply_memory_random(torch.cat((r, m), dim=1)) * rw_random[:, slot].view((-1, 1))

        if self.use_short:
            update_weights_short = self.update_weights_short(inputs)
            mem = self.load_short_memory(len(indices))
            uw = update_weights_short
            for slot in range(0, short_slots):
                r = mem[:,slot]
                r = self.update_memory(torch.cat((r, inputs), dim=1)) * uw[:, slot].view((-1, 1)) + r * (torch.ones(uw[:, slot].shape, device="cuda") - uw[:, slot]).view((-1, 1))

                update = r.detach().cpu().numpy()
                for i_batch in range(len(indices)):
                    for update_i, update_value in enumerate(update[i_batch]):
                        short_memory[i_batch][slot][update_i] = update_factor * update_value + (1.0 - update_factor) * short_memory[i_batch][slot][update_i]

                m = m + self.apply_memory(torch.cat((r, m), dim=1))

        # return torch.stack(read_memory, dim=0)
        return m


class DatabaseMemory(nn.Module):
    def __init__(self, config, in_channels, read_slots, use_short=False):
        super(DatabaseMemory, self).__init__()
        self.read_slots = read_slots
        self.in_channels = in_channels
        self.config = config
        self.slots = slots

        batch_size = 16

        self.memory_access = MemoryAccess(config, in_channels, self.slots, self.read_slots, use_short=use_short)

        self.main_logic = nn.Sequential(
            FastAttention(config, slot_size + in_channels),
            nn.Linear(slot_size + in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.ReLU(True),
        )

        self.apply(weights_init)

    def forward(self, inputs):
        og_shape = inputs.shape
        inputs = inputs.reshape((-1, self.in_channels))
        m = self.memory_access(inputs)
        return self.main_logic(torch.cat((inputs, m), dim=1)).reshape(og_shape)

    def save_db(self):
        global memory
        memory.dump("./memory")
