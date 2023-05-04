import torch
from torch import nn
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


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.embedding_dim = 16
        self.lstm_size = 16
        self.num_layers = 12
        self.jx_lstm = 16

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)

        self.preparation_layer = nn.Sequential(
            nn.Linear(vocab_size, self.lstm_size), nn.Tanh()
        )

        self.pre_lstm = nn.Sequential(
            nn.Linear(self.lstm_size * 16, self.lstm_size * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            #
            nn.Linear(self.lstm_size * 32, self.lstm_size * 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            #
            nn.Linear(self.lstm_size * 64, self.lstm_size * self.jx_lstm),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(
            input_size=self.lstm_size * self.jx_lstm,
            hidden_size=self.lstm_size * self.jx_lstm,
            num_layers=self.num_layers,
            dropout=0.3 / self.num_layers,
        )

        self.fc = nn.Sequential(
            nn.Linear(self.lstm_size * self.jx_lstm, self.lstm_size * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            #
            nn.Linear(self.lstm_size * 32, self.lstm_size * 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            #
            nn.Linear(self.lstm_size * 64, 1),
            nn.Sigmoid(),
        )

        self.context_layer = nn.Sequential(
            nn.Conv1d(
                config.context_length, self.lstm_size * 4, 8, stride=2, padding=2
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(self.lstm_size * 4),
            #
            nn.Conv1d(self.lstm_size * 4, self.lstm_size * 6, 8, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(self.lstm_size * 6),
            #
            nn.Conv1d(self.lstm_size * 6, self.lstm_size * 8, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(self.lstm_size * 8),
            #
            nn.Conv1d(self.lstm_size * 8, self.lstm_size * 16, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(self.lstm_size * 16),
            #
            nn.Conv1d(self.lstm_size * 16, self.lstm_size * 15, 2),
            nn.Tanh(),
        )

        self.apply(weights_init)

    def forward(self, inputs, prev_state):
        input, context = inputs
        c = torch.flatten(context, end_dim=1)
        c = self.embedding(c)
        c = self.context_layer(c)
        c = torch.flatten(c)
        c = torch.unflatten(c, 0, (-1, input.shape[1], self.lstm_size * 15))
        input = self.preparation_layer(input)
        s = torch.cat((input, c), dim=2)
        s = self.pre_lstm(s)
        output, state = self.lstm(s, prev_state)
        return self.fc(output), state

    def init_state(self, sequence_length):
        return (
            torch.full(
                (
                    self.num_layers,
                    sequence_length,
                    self.lstm_size * self.jx_lstm,
                ),
                1.0 / (self.lstm_size * self.jx_lstm),
            ),
            torch.full(
                (
                    self.num_layers,
                    sequence_length,
                    self.lstm_size * self.jx_lstm,
                ),
                1.0 / (self.lstm_size * self.jx_lstm),
            ),
        )
