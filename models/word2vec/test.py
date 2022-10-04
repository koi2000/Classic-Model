import pandas as pd
import numpy as np
from collections import Counter

import scipy
import torch

# ----超参数-----------------------------------------
# 窗口大小

from models.word2vec.model import EmbeddingModel

C = 3
# 负采样样本倍数
K = 15
# 训练轮数
epochs = 1
MAX_VOCAB_SIZE = 10000
EMBEDDING_SIZE = 100
batch_size = 32
lr = 0.2
momentum = 0.9
# ---------------------------------------------------

with open('宋_4.txt', encoding='utf-8') as f:
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


def find_nearest(embedding_weights, word):
    index = word2idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx2word[i] for i in cos_dis.argsort()[:10]]


def test():
    model = EmbeddingModel(len(word_freqs), 100)
    weight = torch.load("./checkpoint/embedding-10000.th")
    model.load_state_dict(weight)
    embedding_weights = model.input_embedding()
    for word in ["田", "地", "人"]:
        print(word, find_nearest(embedding_weights, word))


if __name__ == '__main__':
    test()
