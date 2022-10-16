import numpy as np
import torch
from cvxopt import matrix, solvers
import numpy as np
from gensim.similarities import WmdSimilarity
from gensim.models.word2vec import Word2Vec
from models.word2vec.model import EmbeddingModel


class Config(object):
    npData = "./song.npz.npy"


embedding_weights = None
word2idx = None


def loading_embedding():
    global embedding_weights, word2idx
    mp = np.load(Config.npData, allow_pickle=True)
    text = mp[0]
    word2idx = mp[1]
    idx2word = mp[2]
    word_freqs = mp[3]
    word_counts = mp[4]
    model = EmbeddingModel(len(word_freqs), 500)
    weight = torch.load("./embedding-50000.th")
    model.load_state_dict(weight)
    embedding_weights = model.input_embedding()


# 计算两个向量的欧式距离
def get_word_embedding_distance(emb1, emb2):
    if (len(emb1) != len(emb2)):
        print('error input,x and y is not in the same space')
        return
    result1 = 0.0
    for i in range(len(emb1)):
        result1 += (emb1[i] - emb2[i]) * (emb1[i] - emb2[i])
    distance = result1 ** 0.5
    return distance


# 把list转成map，key为list中的元素，value为在原句中出现的次数
def word_count(words):
    word_map = {}
    for word in words:
        if word in word_map.keys():
            word_map[word] += 1.0
        else:
            word_map[word] = 1.0
    return word_map


def WMD(sen1, sen2):
    # words1 = sen1.split()
    # words2 = sen2.split()
    words1 = list(sen1)
    words2 = list(sen2)

    word_map1 = word_count(words1)
    word_map2 = word_count(words2)

    word_embs1 = [embedding_weights[word2idx[word]] for word in word_map1.keys()]
    word_embs2 = [embedding_weights[word2idx[word]] for word in word_map2.keys()]

    len1 = len(word_embs1)
    len2 = len(word_embs2)
    c_ij = []
    for i in range(len1):
        for j in range(len2):
            c_ij.append(get_word_embedding_distance(word_embs1[i], word_embs2[j]))

    # 设计A矩阵
    a = []
    #   句子1对应到句子2部分
    for i in range(len1):
        line_ele = []
        for ii in range(i * len2):
            line_ele.append(0.0)
        for j in range(len2):
            line_ele.append(1.0)
        for ii in range((len1 - i - 1) * len2):
            line_ele.append(0.0)
        a.append(line_ele)

    #   句子2对应到句子1
    for i in range(len2):
        line_ele = []
        for ii in range(len1):
            for jj in range(i):
                line_ele.append(0.0)
            line_ele.append(1.0)
            for jj in range(0, len2 - i - 1):
                line_ele.append(0.0)
        a.append(line_ele)

    # 获得出入量之和 这部分逻辑是跟A的设计连在一起的
    b = [ele / len(words1) for ele in list(word_map1.values())] + \
        [ele / len(words2) for ele in list(word_map2.values())]

    # 列出线性规划问题
    A_matrix = matrix(a).trans()
    b_matrix = matrix(b)
    c_matrix = matrix(c_ij)
    num_of_T = len(c_ij)
    G = matrix(-np.eye(num_of_T))
    h = matrix(np.zeros(num_of_T))

    # 求解器求解，注意这里必须指定solver，否则会报错，蛋疼
    sol = solvers.lp(c_matrix, G, h, A=A_matrix, b=b_matrix, solver='glpk')
    return sol['primal objective']


if __name__ == '__main__':
    sent2 = "漫卷诗书喜欲狂"
    sent1 = "飞流直下三千尺"
    sent3 = "疑"
    print(WMD(sent1, sent2))
    print(WMD(sent1, sent3))
