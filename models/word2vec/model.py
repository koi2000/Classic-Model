import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)

    def forward(self, input_labels, pos_labels, neg_labels):
        input_embedding = self.in_embed(input_labels)
        pos_embedding = self.in_embed(pos_labels)
        neg_embedding = self.in_embed(neg_labels)

        input_embedding = input_embedding.unsqueeze(2)
        pos_dot = torch.bmm(pos_embedding, input_embedding)
        pos_dot = pos_dot.squeeze(2)

        neg_dot = torch.bmm(neg_embedding, -input_embedding)
        neg_dot = neg_dot.squeeze(2)

        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = log_pos + log_neg
        return -loss

    def input_embedding(self):
        return self.in_embed.weight.detach().numpy()
