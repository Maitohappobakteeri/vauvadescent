import torch
from torch import nn

import log
from prepare_data import vocab_size
from easy_lstm import EasyLSTM


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_uniform_(m.weight.data, 0.5)
    elif classname.find("Linear") != -1:
        nn.init.xavier_uniform_(m.weight.data, 0.5)
    elif classname.find("LSTM") != -1:
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param.data, 1.0)
            elif "weight_hh" in name:
                torch.nn.init.xavier_uniform_(param.data, 1.0)
            elif "bias" in name:
                param.data.fill_(0)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding_dim = 256
        self.lstm_size = 16
        self.num_layers = 2

        self.jx_lstm = 16

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)

        self.pre_lstm = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.lstm_size * 32),
            nn.ReLU(True),
            nn.LayerNorm(self.lstm_size * 32),
            #
            nn.Linear(self.lstm_size * 32, self.lstm_size * 64),
            nn.ReLU(True),
            nn.LayerNorm(self.lstm_size * 64),
            #
            nn.Linear(self.lstm_size * 64, self.lstm_size * self.jx_lstm),
            nn.Tanh(),
        )

        # self.lstm = nn.LSTM(
        #     input_size=self.lstm_size * self.jx_lstm,
        #     hidden_size=self.lstm_size * self.jx_lstm,
        #     num_layers=self.num_layers,
        #     dropout=0.0,
        # )

        self.lstm = EasyLSTM(4, False)

        self.attention = nn.MultiheadAttention(self.embedding_dim, 4)

        self.context_layer = nn.Sequential(
            nn.Conv1d(
                config.context_length, self.lstm_size * 2, 8, stride=2, padding=2
            ),
            nn.ReLU(True),
            nn.BatchNorm1d(self.lstm_size * 2),
            #
            nn.Conv1d(self.lstm_size * 2, self.lstm_size * 4, 8, stride=2, padding=2),
            nn.ReLU(True),
            nn.BatchNorm1d(self.lstm_size * 4),
            #
            nn.Conv1d(self.lstm_size * 4, self.lstm_size * 8, 8, stride=2, padding=2),
            nn.ReLU(True),
            nn.BatchNorm1d(self.lstm_size * 8),
            #
            nn.Conv1d(self.lstm_size * 8, self.lstm_size * 16, 8, stride=2, padding=2),
            nn.ReLU(True),
            nn.BatchNorm1d(self.lstm_size * 16),
            #
            nn.Conv1d(self.lstm_size * 16, self.embedding_dim, 14),
            nn.Tanh(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.lstm_size * self.jx_lstm * 2, self.lstm_size * 32),
            nn.ReLU(True),
            nn.LayerNorm(self.lstm_size * 32),
            #
            nn.Linear(self.lstm_size * 32, self.lstm_size * 64),
            nn.ReLU(True),
            nn.LayerNorm(self.lstm_size * 64),
            #
            nn.Linear(self.lstm_size * 64, vocab_size),
            nn.Sigmoid(),
        )

        self.apply(weights_init)

    def forward(self, inputs, prev_state):
        x, c = inputs

        e = self.embedding(x)

        c = torch.flatten(c, end_dim=1)
        c = self.embedding(c)
        (c, _) = self.attention(c, c, c, need_weights=False)
        c = self.context_layer(c)
        c = torch.flatten(c)
        c = torch.unflatten(c, 0, (-1, x.shape[1], self.embedding_dim))

        s = torch.cat((e, c), dim=2)
        s = self.pre_lstm(s)
        output, state = self.lstm(s, prev_state)
        logits = self.fc(torch.cat((s, output), dim=2))
        logits = torch.divide(logits, torch.add(torch.max(logits), 1e-6))

        return logits, state

    def init_state(self, sequence_length):
        return self.lstm.init_state(sequence_length)
