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

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
# ----超参数-----------------------------------------
# 窗口大小
C = 3
# 负采样样本倍数
K = 15
# 训练轮数
epochs = 1
MAX_VOCAB_SIZE = 100000
EMBEDDING_SIZE = 1000
batch_size = 32
lr = 0.2
momentum = 0.9
# ---------------------------------------------------

with open(Config.data, encoding='utf-8') as f:
    text = f.read()  # 得到文本内容
text = text.replace('，', '')
text = text.replace('。', '')
text = list(text)

vocab_dict = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))  # 得到单词字典表，key是单词，value是次数
vocab_dict['<UNK>'] = len(text) - np.sum(list(vocab_dict.values()))  # 把不常用的单词都编码为"<UNK>"
idx2word = [word for word in vocab_dict.keys()]
word2idx = {word: i for i, word in enumerate(idx2word)}
word_counts = np.array([count for count in vocab_dict.values()], dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3. / 4.)

dataset = WordEmbeddingDataSet(text, word2idx, idx2word, word_freqs, word_counts, C, K)
dataloader = tud.DataLoader(dataset, batch_size, shuffle=True)


def adjust_learning_rate(optimizer, epoch, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    if Config.use_gpu:
        Config.device = torch.device("cuda")
    else:
        Config.device = torch.device("cpu")
    device = Config.device

    model = EmbeddingModel(len(word_freqs), EMBEDDING_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=lr,momentum=momentum)
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
            if i % 50000 == 0:
                torch.save(model.state_dict(), "./checkpoint/embedding-{}.th".format(i))
    embedding_weights = model.input_embeddings()
    torch.save(model.state_dict(), "./checkpoint/embedding-{}.th".format("final"))


if __name__ == '__main__':
    train()
