import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = Field(tokenize=tokenize_de,
            init_token="<sos>",
            eos_token="<eos>",
            lower=True)

TRG = Field(tokenize=tokenize_en,
            init_token="<sos>",
            eos_token="<eos>",
            lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device
)

for i, it in enumerate(iter(train_iterator)):
    if i > 10:
        break
    src = it.src
    trg = it.trg
    print(src.shape, trg.shape)

batch_idx = 0
data = next(iter(train_iterator))
for idx in data.src[:, batch_idx].cpu().numpy():
    print(SRC.vocab.itos[idx], end=' ')

print()
for idx in data.trg[:, batch_idx].cpu().numpy():
    print(TRG.vocab.itos[idx], end=' ')
