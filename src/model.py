import torch
from torch import nn

import log
from prepare_data import vocab_size
from easy_lstm import EasyLSTM
from database_memory import DatabaseMemory


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

        self.jx_lstm = 32

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)

        # self.lstm = nn.LSTM(
        #     input_size=self.lstm_size * self.jx_lstm,
        #     hidden_size=self.lstm_size * self.jx_lstm,
        #     num_layers=self.num_layers,
        #     dropout=0.0,
        # )

        self.db_memory = DatabaseMemory(config, self.lstm_size * 32, 1, use_short=True)
        # self.lstm = EasyLSTM(config, 4, False)

        self.context_layer = nn.Sequential(
            nn.Conv1d(
                config.context_length, self.lstm_size * 2, 8, stride=2, padding=2
            ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(self.lstm_size * 2),
            #
            nn.Conv1d(self.lstm_size * 2, self.lstm_size * 4, 8, stride=2, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(self.lstm_size * 4),
            #
            nn.Conv1d(self.lstm_size * 4, self.lstm_size * 8, 8, stride=2, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(self.lstm_size * 8),
            #
            nn.Conv1d(self.lstm_size * 8, self.lstm_size * 16, 8, stride=2, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(self.lstm_size * 16),
            #
            nn.Conv1d(self.lstm_size * 16, self.lstm_size * 32, 14),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(self.lstm_size * 32),
            # nn.Tanh(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.lstm_size * 64 + self.embedding_dim, vocab_size),
        )

        self.apply(weights_init)

    def forward(self, inputs, prev_state):
        x, c = inputs

        e = self.embedding(x)

        c = torch.flatten(c, end_dim=1)
        c = self.embedding(c)
        c = self.context_layer(c)
        
        c = c.reshape((x.shape[0], -1, self.lstm_size * 32))
        m_output = c
        m_output = torch.split(m_output, 1, dim=1)
        c_list = []
        for cc in m_output:
            cc = cc.reshape((-1, self.lstm_size * 32))
            memory_output = self.db_memory(cc)
            c_list.append(memory_output.reshape((x.shape[0], 1, self.lstm_size * 32)))
        m_output = torch.cat(c_list, dim=1)

        c = c.view((-1, x.shape[1], self.lstm_size * 32))
        s = torch.cat((e, c, m_output), dim=2)
        logits = nn.functional.softmax(self.fc(s), dim=2)
        # logits = torch.divide(logits, torch.add(torch.max(logits), 1e-11))

        return logits, prev_state

    def init_state(self, sequence_length):
        # return self.lstm.init_state(sequence_length)
        return torch.zeros((1,1,1)), torch.zeros((1,1,1))
