import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from torch import optim
from tqdm import tqdm

from config import Config
from dataset import WordEmbeddingDataSet
from model import EmbeddingModel
from handledata import processPoetry, processWord
from utils import adjust_learning_rate

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
# ----超参数-----------------------------------------
# 窗口大小
C = Config.C
# 负采样样本倍数
K = Config.K
# 训练轮数
epochs = Config.epochs
MAX_VOCAB_SIZE = Config.MAX_POETRY_VOCAB_SIZE
EMBEDDING_SIZE = Config.EMBEDDING_SIZE
batch_size = Config.batch_size
lr = Config.lr
momentum = Config.momentum
# ---------------------------------------------------
mp = np.load(Config.npData,allow_pickle=True)
# text = mp["text"]
# word2idx = mp["word2idx"]
# idx2word = mp['idx2word']
# word_freqs = mp['word_freqs']
# word_counts = mp['word_counts']

text = mp[0]
word2idx = mp[1]
idx2word = mp[2]
word_freqs = mp[3]
word_counts = mp[4]

dataset = WordEmbeddingDataSet(text, word2idx, idx2word, word_freqs, word_counts, C, K)
dataloader = tud.DataLoader(dataset, batch_size, shuffle=True)


def train():
    if Config.use_gpu:
        Config.device = torch.device("cuda")
    else:
        Config.device = torch.device("cpu")
    device = Config.device

    model = EmbeddingModel(len(word_freqs), EMBEDDING_SIZE)

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    # 加载上次使用的模型
    # weight = torch.load("./checkpoint1/embedding-100000.th")
    # model.load_state_dict(weight)
    # 转移到相应计算设备上
    model.to(device)

    for e in (range(epochs)):
        adjust_learning_rate(optimizer, e, lr)
        for i, (input_labels, pos_labels, neg_labels) in tqdm(enumerate(dataloader)):
            input_labels = input_labels.long().to(device)
            pos_labels = pos_labels.long().to(device)
            neg_labels = neg_labels.long().to(device)

            optimizer.zero_grad()
            loss = model(input_labels, pos_labels, neg_labels).mean()
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                print('epoch', e, 'iteration', i, loss.item())
            if i % 10000 == 0:
                torch.save(model.state_dict(), "./checkpoint2/embedding-{}.th".format(i))
    embedding_weights = model.input_embeddings()
    torch.save(model.state_dict(), "./checkpoint2/embedding-{}.th".format("final"))


if __name__ == '__main__':
    train()
