import torch
from torch import nn

import log
from prepare_data import vocab_size


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_normal_(m.weight.data, 0.5)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight.data, 1.0)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding_dim = 128
        self.lstm_size = 128
        self.num_layers = 6

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim  * 2,
            hidden_size=self.lstm_size * 2,
            num_layers=self.num_layers,
            dropout=0.2,
        )

        self.context_layer = nn.Sequential(
            nn.Conv1d(
                config.context_length, self.lstm_size // 3, 16, stride=4, padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.lstm_size // 3),
            nn.Conv1d(
                self.lstm_size // 3, self.lstm_size // 2, 16, stride=4, padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.lstm_size // 2),
            nn.Conv1d(self.lstm_size // 2, self.lstm_size, 5),
            nn.BatchNorm1d(self.lstm_size),
            nn.Sigmoid(),
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear(self.lstm_size * 2, self.lstm_size * 3),
            nn.Tanh(),
            nn.Dropout(0.05),
            nn.Linear(self.lstm_size * 3, self.lstm_size * 4),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(self.lstm_size * 4, self.lstm_size * 8),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(self.lstm_size * 8, vocab_size),
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
        c = torch.unflatten(c, 0, (-1, x.shape[1], self.lstm_size))

        s = torch.cat((e, c), dim=2)
        output, state = self.lstm(s, prev_state)
        logits = self.fc(output)

        return logits, state

    def init_state(self, sequence_length):
        return (
            torch.zeros(self.num_layers, sequence_length, self.lstm_size * 2),
            torch.zeros(self.num_layers, sequence_length, self.lstm_size * 2),
        )
