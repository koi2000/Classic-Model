import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout=0.5):
        super(TextCNN, self).__init__()
        Ci = 1
        Co = kernel_num
        self.embed = nn.Embedding(embed_num, embed_dim)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (f, embed_dim), padding=(2, 0)) for f in kernel_sizes])
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(Co * len(kernel_sizes), class_num)

    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc(x)
        return logit
