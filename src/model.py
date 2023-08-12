import torch
from torch import nn

import log
from prepare_data import vocab_size
from easy_lstm import EasyLSTM
from database_memory import DatabaseMemory


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Embedding") != -1:
        nn.init.xavier_normal_(m.weight.data, 1.0)
    elif classname.find("Conv") != -1:
        nn.init.xavier_normal_(m.weight.data, 1.0)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight.data, 1.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("LayerNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    # elif classname.find("LSTM") != -1:
    #     for name, param in m.named_parameters():
    #         if "weight_ih" in name:
    #             torch.nn.init.xavier_normal_(param.data, 1.0)
    #         elif "weight_hh" in name:
    #             torch.nn.init.xavier_normal_(param.data, 1.0)
    #         elif "bias" in name:
    #             param.data.fill_(0)


class Attention(nn.Module):
    def __init__(self, config, in_channels):
        super(Attention, self).__init__()
        self.config = config
        self.attention = nn.MultiheadAttention(in_channels, 1)
        self.feedforward = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * 2, bias=False),
            nn.Linear(in_channels * 2, in_channels, bias=False),
        )
        self.norm = nn.LayerNorm(in_channels)
    
    def forward(self, input, key=None, value=None):
        if key is None:
            key = input
        if value is None:
            value = input
        a0, _ = self.attention(input, key, value, need_weights=False)
        a1 = self.feedforward(a0 + input)
        return self.norm(a0 + a1)
    
class AttentionNoForward(nn.Module):
    def __init__(self, config, in_channels):
        super(AttentionNoForward, self).__init__()
        self.config = config
        self.attention = nn.MultiheadAttention(in_channels, 1)
        self.norm = nn.LayerNorm(in_channels)
    
    def forward(self, input, key=None, value=None):
        if key is None:
            key = input
        if value is None:
            value = input
        a0, _ = self.attention(input, key, value, need_weights=False)
        return self.norm(a0 + input)

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding_dim = 256
        self.lstm_size = 16
        self.num_layers = 2
        self.config = config

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.pos_embedding = nn.Embedding(config.context_length + 1, self.embedding_dim)

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=self.num_layers,
            dropout=0.0,
            batch_first=True
        )

        self.attention = Attention(config, self.embedding_dim)

        self.context_layer = nn.Sequential(
            nn.Conv1d(
                config.short_context_length, self.lstm_size * 2, 8, stride=2, padding=2, bias=False
            ),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(self.lstm_size * 2),
            #
            nn.Conv1d(self.lstm_size * 2, self.lstm_size * 4, 8, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(self.lstm_size * 4),
            #
            nn.Conv1d(self.lstm_size * 4, self.lstm_size * 8, 8, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(self.lstm_size * 8),
            #
            nn.Conv1d(self.lstm_size * 8, self.lstm_size * 16, 8, 2, 2, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(self.lstm_size * 16),
            #
            nn.Conv1d(self.lstm_size * 16, self.embedding_dim, 14, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(self.embedding_dim),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, vocab_size * 2, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.LayerNorm(vocab_size * 2),
            nn.Linear(vocab_size * 2, vocab_size, bias=False),

        )

        self.attention_no_forward = AttentionNoForward(config, self.embedding_dim)
        self.final_attention = Attention(config, self.embedding_dim)
        self.final_norm = nn.LayerNorm(self.embedding_dim)

        self.apply(weights_init)

    def forward(self, inputs, prev_state):
        x, c = inputs

        e = self.embedding(x)
        
        c = torch.flatten(c, end_dim=1)
        c = self.embedding(c)
        positions = torch.arange(1, 1 + self.config.context_length).view((1, -1)).repeat(x.shape[0] * x.shape[1], 1).to(self.config.device)
        c = c0 = c + self.pos_embedding(positions) 
        c_shape = c.shape

        # mask zero!!!!
        c = self.attention(c.view((x.shape[0] * x.shape[1], -1, self.embedding_dim))).view(c_shape)
        c_a = c
        c = c[:, :self.config.short_context_length, :]
        c = self.context_layer(c)
        
        # c = c.reshape((x.shape[0], -1, self.lstm_size * 32))
        # m_output = c
        # m_output = torch.split(m_output, 1, dim=1)
        # c_list = []
        # for cc in m_output:
        #     cc = cc.reshape((-1, self.lstm_size * 32))
        #     memory_output = self.db_memory(cc)
        #     memory_output = self.db_memory2(memory_output)
        #     c_list.append(memory_output.reshape((x.shape[0], 1, self.lstm_size * 32)))
        # c = torch.cat(c_list, dim=1)

        c = c.view((-1, x.shape[1], self.embedding_dim))
        s = s0 = e + c
        s, state = self.lstm(s, prev_state)
        s1 = s
        # logits = nn.functional.softmax(self.fc(s / temperature), dim=-1)
        # return self.fc(s), state
        # logits = torch.divide(logits, torch.add(torch.max(logits), 1e-11))
        output_embedding = s.reshape((x.shape[0] * x.shape[1], 1, self.embedding_dim))
        output_embedding = torch.cat((output_embedding, c0.view((x.shape[0] * x.shape[1], -1, self.embedding_dim))[:, :-1, :]), dim=1)

        positions = torch.arange(0, self.config.context_length).view((1, -1)).repeat(x.shape[0] * x.shape[1], 1).to(self.config.device)
        output_embedding = output_embedding + self.pos_embedding(positions) 

        c1 = self.attention_no_forward(c0.view((x.shape[0] * x.shape[1], -1, self.embedding_dim)))
        output_embedding = self.final_attention(output_embedding, c_a, c1).view((x.shape[0], x.shape[1], -1, self.embedding_dim))
        output_embedding = torch.sum(output_embedding.transpose(2, 3), dim=-1)
        output_embedding = self.final_norm(output_embedding)
        output_embedding = output_embedding.view((x.shape[0], x.shape[1], self.embedding_dim))
        # s = s + s0 + s1 + output_embedding
        s = torch.cat((s0, s1, output_embedding), dim=-1)

        logits = self.fc(s)

        return logits, state

    def init_state(self, sequence_length):
        # return self.lstm.init_state(sequence_length)
        # return torch.zeros((1,1,1)), torch.zeros((1,1,1))
        return (
            torch.zeros(
                (
                    self.num_layers,
                    sequence_length,
                    self.embedding_dim,
                )
            ),
            torch.zeros(
                (
                    self.num_layers,
                    sequence_length,
                    self.embedding_dim,
                )
            ),
        )
