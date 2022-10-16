import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import random
import math
import time

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# conda install -c conda-forge spacy-model-en_core_web_sm
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

# 定义分词方法
def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

# 定义分词方法
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

train_data, valid_data, test_data = Multi30k.splits(exts=(".de", ".en"),
                                                    fields=(SRC, TRG))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

BATCH_SIZE = 128
# 大概能做到将长度相似的文本放在一起，避免某一个batch出现大量长短不一的现象
# 该部分也能做到自动的将源语言句子padding到统一的长度
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),
                                                                      batch_size=BATCH_SIZE,
                                                                      device=device)
