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
    def __init__(self, config, num_layers, discriminator=False):
        super(EasyLSTM, self).__init__()
        self.config = config
        self.lstm_size = 16
        self.num_layers = 2
        self.num_layers_group = num_layers
        self.jx_lstm = 16

        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=self.lstm_size * self.jx_lstm,
                    hidden_size=self.lstm_size * self.jx_lstm,
                    num_layers=self.num_layers,
                    dropout=0.0,
                    batch_first=True
                )
            )

        self.leak_layers = nn.ModuleList()
        for i in range(num_layers):
            self.leak_layers.append(
                nn.Sequential(
                    nn.Linear(
                        self.lstm_size * self.jx_lstm * 2, self.lstm_size * self.jx_lstm
                    ),

                    nn.ReLU(True)
                    if not discriminator
                    else nn.LeakyReLU(0.2, inplace=True),

                    nn.LayerNorm(self.lstm_size * self.jx_lstm)
                )
            )

        self.apply(weights_init)

    def forward(self, inputs, prev_state):
        state_h, state_c = prev_state
        state_h = state_h.split(self.num_layers)
        state_c = state_c.split(self.num_layers)
        new_state1 = []
        new_state2 = []
        for i in range(self.num_layers_group):
            new_inputs, state = self.lstm_layers[i](inputs, (state_h[i], state_c[i]))
            inputs = self.leak_layers[i](torch.cat((inputs, new_inputs), dim=2))
            new_state1.append(state[0])
            new_state2.append(state[1])

        return inputs, (
            torch.cat([state for state in new_state1]),
            torch.cat([state for state in new_state2]),
        )

    def init_state(self, sequence_length):
        return (
            torch.full(
                (
                    self.num_layers_group * self.num_layers,
                    sequence_length,
                    self.lstm_size * self.jx_lstm,
                ),
                1.0 / (self.lstm_size * self.jx_lstm),
            ),
            torch.full(
                (
                    self.num_layers_group * self.num_layers,
                    sequence_length,
                    self.lstm_size * self.jx_lstm,
                ),
                1.0 / (self.lstm_size * self.jx_lstm),
            ),
        )
