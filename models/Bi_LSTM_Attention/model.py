import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_hidden, num_layers, num_classes):
        super(BiLSTM_Attention, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, num_layers, bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, num_classes)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        hidden = final_state.view(-1, self.n_hidden * 2, self.num_layers)
        # attn_weights : [batch_size, n_step]
        attention_weight = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attention_weight, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        # context[batch_size, n_hidden * num_directions(=2)]
        return context, soft_attn_weights.data.numpy()

    def forward(self, X):
        # input : [batch_size, len_seq, embedding_dim]
        input = self.embedding(X)
        # input : [len_seq, batch_size, embedding_dim]
        input = input.permute(1, 0, 2)
        # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        hidden_state = torch.zeros(self.num_layers * 2, len(X), self.n_hidden)
        # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.zeros(self.num_layers * 2, len(X), self.n_hidden)
        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2)  # output : [batch_size, len_seq, n_hidden]
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention
