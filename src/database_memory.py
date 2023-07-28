import torch
from torch import nn
import numpy as np
import os

# import sqlite3
# con = sqlite3.connect("../trained.db")

import log
from prepare_data import vocab_size

max_batches = 16
features = 16
slots = 10_000
short_slots = 3
slot_size = 256

short_memory = np.zeros((
    max_batches,
    short_slots,
    slot_size,
))

memory = np.zeros((
    max_batches,
    slots,
    slot_size,
))

update_memory = np.zeros((
    max_batches,
    slots,
    slot_size,
))

def update():
    global memory, update_memory, short_memory
    memory = np.copy(update_memory)
    short_memory = np.zeros((
        max_batches,
        short_slots,
        slot_size,
    ))

if os.path.isfile("../memory"):
    memory = np.load("../memory", allow_pickle=True)

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
            nn.Linear(in_channels, in_channels),
            nn.Softmax(dim=1),
            nn.Linear(in_channels, in_channels)
        )

        self.apply(weights_init)

    def forward(self, inputs):
        return inputs * self.main(inputs)
    

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
            nn.Tanh()
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

    def forward(self, inputs):
        global memory, update_memory
        batch_size = inputs.shape[0]
        read = self.read_weights(inputs).reshape((batch_size, -1, self.available_slots))
        read_weights, read_indices = torch.max(read, dim=2)
        update_weights = self.update_weights(inputs)
        indices = read_indices.cpu().numpy()
        read_memory = []

        for i_batch in range(len(indices)):
            m = torch.zeros((slot_size, ), device="cuda")
            for i_slot, i in enumerate(indices[i_batch]):
                r = torch.tensor(memory[i_batch][i], dtype=torch.float, device="cuda")
                r = update_weights[i_batch, i_slot] * self.update_memory(torch.cat((r, inputs[i_batch]), dim=0).unsqueeze(dim=0)).squeeze(dim=0) + (1.0 - update_weights[i_batch, i_slot]) * r
                #update start
                if update_weights[i_batch, i_slot] > 0.5:
                    update = r.detach().cpu().numpy()
                    for update_i, update_value in enumerate(update):
                        update_memory[i_batch][i][update_i] = 0.5 * update_value + 0.5 * update_memory[i_batch][i][update_i]
                #update end
                m = read_weights[i_batch, i_slot] * self.apply_memory(torch.cat((r, m), dim=0).unsqueeze(dim=0)).squeeze(dim=0)
            read_memory.append(torch.tanh(m))

        if self.use_short:
            update_weights_short = self.update_weights_short(inputs)
            for i_batch in range(len(indices)):
                m = read_memory[i_batch]
                for i in range(short_slots):
                    r = torch.tensor(short_memory[i_batch][i], dtype=torch.float, device="cuda")
                    r = update_weights_short[i_batch, i] * self.update_short_memory(torch.cat((r, inputs[i_batch]), dim=0).unsqueeze(dim=0)).squeeze(dim=0) + (1.0 - update_weights_short[i_batch, i]) * r
                    #update start
                    if update_weights_short[i_batch, i] > 0.5:
                        update = r.detach().cpu().numpy()
                        for update_i, update_value in enumerate(update):
                            short_memory[i_batch][i][update_i] = update_value
                    #update end
                    m = read_weights[i_batch, i_slot] * self.apply_memory(torch.cat((r, m), dim=0).unsqueeze(dim=0)).squeeze(dim=0)
                read_memory[i_batch] = torch.tanh(m)

        return torch.stack(read_memory, dim=0)


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

            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.ReLU(True)
        )

        self.apply(weights_init)

    def forward(self, inputs):
        og_shape = inputs.shape
        inputs = inputs.reshape((-1, self.in_channels))
        m = self.memory_access(inputs)
        return self.main_logic(torch.cat((inputs, m), dim=1)).reshape(og_shape)

    def save_db(self):
        global memory
        memory.dump("../memory")
