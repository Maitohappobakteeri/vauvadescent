import torch
from torch import nn

import log
from prepare_data import vocab_size


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_normal_(m.weight.data, 1.0)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight.data, 1.0)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding_dim = 16
        self.lstm_size = 16
        self.num_layers = 1

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)

        self.pre_lstm = nn.Sequential(
            nn.Linear(self.lstm_size * 16, self.lstm_size * 32),
            nn.ReLU(True),
            #
            nn.Linear(self.lstm_size * 32, self.lstm_size * 16),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim * 16,
            hidden_size=self.lstm_size * 16,
            num_layers=self.num_layers,
            dropout=0.0,
        )

        self.context_layer = nn.Sequential(
            nn.Conv1d(
                config.context_length, self.lstm_size * 4, 8, stride=2, padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.lstm_size * 4),
            #
            nn.Conv1d(self.lstm_size * 4, self.lstm_size * 6, 8, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(self.lstm_size * 6),
            #
            nn.Conv1d(self.lstm_size * 6, self.lstm_size * 8, 1),
            nn.ReLU(),
            nn.BatchNorm1d(self.lstm_size * 8),
            #
            nn.Conv1d(self.lstm_size * 8, self.lstm_size * 16, 1),
            nn.ReLU(),
            nn.BatchNorm1d(self.lstm_size * 16),
            #
            nn.Conv1d(self.lstm_size * 16, self.lstm_size * 15, 2),
            nn.Tanh(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.lstm_size * 16, self.lstm_size * 32),
            nn.ReLU(),
            nn.Dropout(0.01),
            #
            nn.Linear(self.lstm_size * 32, self.lstm_size * 64),
            nn.ReLU(),
            nn.Dropout(0.01),
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
        c = self.context_layer(c)
        c = torch.flatten(c)
        c = torch.unflatten(c, 0, (-1, x.shape[1], self.lstm_size * 15))

        s = torch.cat((e, c), dim=2)
        s = self.pre_lstm(s)
        output, state = self.lstm(s, prev_state)
        logits = self.fc(output)
        logits = torch.divide(logits, torch.add(torch.max(logits), 1e-6))

        return logits, state

    def init_state(self, sequence_length):
        return (
            torch.zeros(self.num_layers, sequence_length, self.lstm_size * 16),
            torch.zeros(self.num_layers, sequence_length, self.lstm_size * 16),
        )
