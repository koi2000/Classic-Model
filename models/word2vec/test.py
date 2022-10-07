import pandas as pd
import numpy as np
from collections import Counter

import scipy
import torch
from config import Config
from model import EmbeddingModel
from handledata import processPoetry, processWord

# ----超参数-----------------------------------------
# 窗口大小
C = Config.C
# 负采样样本倍数
K = Config.K
# 训练轮数
epochs = Config.epochs
MAX_VOCAB_SIZE = Config.MAX_WORD_VOCAB_SIZE
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


def find_nearest(embedding_weights, word):
    index = word2idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx2word[i] for i in cos_dis.argsort()[:10]]


def test():
    model = EmbeddingModel(len(word_freqs), Config.EMBEDDING_SIZE)
    weight = torch.load("./checkpoint2/embedding-50000.th")
    model.load_state_dict(weight)
    embedding_weights = model.input_embedding()
    for word in ["喜",'怒',"哀","乐"]:
        print(word, find_nearest(embedding_weights, word))
    # for word in ["one", 'second',"computer"]:
    #     print(word, find_nearest(embedding_weights, word))


if __name__ == '__main__':
    test()
