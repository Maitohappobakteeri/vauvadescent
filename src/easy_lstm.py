import torch
from torch import nn

import log
from prepare_data import vocab_size


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


class EasyLSTM(nn.Module):
    def __init__(self, num_layers, discriminator=False):
        super(EasyLSTM, self).__init__()
        self.lstm_size = 16
        self.num_layers = 1
        self.num_layers_group = num_layers
        self.dropout = 0.2
        self.jx_lstm = 8

        self.lstm_layers = []
        for i in range(num_layers):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=self.lstm_size * self.jx_lstm,
                    hidden_size=self.lstm_size * self.jx_lstm,
                    num_layers=self.num_layers,
                    dropout=0.0,
                )
            )

        self.inbetween_layers = []
        self.leak_layers = []
        for i in range(num_layers - 1):
            self.inbetween_layers.append(
                nn.Sequential(
                    nn.Linear(
                        self.lstm_size * self.jx_lstm, self.lstm_size * self.jx_lstm * 2
                    ),
                    nn.ReLU(True)
                    if not discriminator
                    else nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(self.dropout),
                    nn.Linear(
                        self.lstm_size * self.jx_lstm * 2,
                        self.lstm_size * self.jx_lstm * 4,
                    ),
                    nn.ReLU(True)
                    if not discriminator
                    else nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(self.dropout),
                    nn.Linear(
                        self.lstm_size * self.jx_lstm * 4, self.lstm_size * self.jx_lstm
                    ),
                    nn.Tanh(),
                    nn.Dropout(self.dropout),
                )
            )

            self.leak_layers.append(
                nn.Linear(
                    self.lstm_size * self.jx_lstm * 2, self.lstm_size * self.jx_lstm
                )
            )

        self.original_dropout = nn.Dropout(0.5)
        self.mix_layer = nn.Linear(
            self.lstm_size * self.jx_lstm * 2, self.lstm_size * self.jx_lstm
        )

        self.apply(weights_init)

    def forward(self, inputs, prev_state):
        state_h, state_c = prev_state
        state_h = state_h.split(self.num_layers)
        state_c = state_c.split(self.num_layers)

        new_state = []
        inputs_og = self.original_dropout(inputs)
        for i in range(self.num_layers_group):
            if i > 0:
                inputs = self.inbetween_layers[i - 1](inputs)
                inputs_og = self.mix_layer(torch.cat((inputs, inputs_og), dim=2))
            inputs, state = self.lstm_layers[i](inputs, (state_h[i], state_c[i]))
            new_state.append(state)

        inputs = self.mix_layer(torch.cat((inputs, inputs_og), dim=2))

        return inputs, (
            torch.cat([state[0] for state in new_state]),
            torch.cat([state[1] for state in new_state]),
        )

    def init_state(self, sequence_length):
        return (
            torch.zeros(
                self.num_layers_group * self.num_layers,
                sequence_length,
                self.lstm_size * self.jx_lstm,
            ),
            torch.zeros(
                self.num_layers_group * self.num_layers,
                sequence_length,
                self.lstm_size * self.jx_lstm,
            ),
        )
