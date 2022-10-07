import pathlib
from collections import Counter

import pandas as pd
import numpy as np
from wordcloud import WordCloud, ImageColorGenerator  # , STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image
from jiayan import PMIEntropyLexiconConstructor  # 常用的古汉语分词器
import jieba  # 常用的中文分词器
import hanlp
import torch

from config import Config


def csv2txt(txtPath, df):
    with open(txtPath, 'a', encoding='utf-8') as f:
        for i in df['内容'].values:
            f.write(i)


def processData():
    for csv in pathlib.Path(Config.dataPath).glob('*.csv'):
        name = Config.dataPath + "/" + csv.name
        df = pd.read_csv(name, encoding='utf-8')
        csv2txt(Config.poetrydata, df)


def processPoetryWithJiayan():
    with open(Config.poetrydata, encoding='utf-8') as f:
        text = f.read()
    # constructor = PMIEntropyLexiconConstructor()
    # lexicon = constructor.construct_lexicon(Config.poetrydata)
    # df = lexicon
    # constructor.save(lexicon, './data/jiayan.csv')
    df = pd.read_csv("./data/jiayan.csv", encoding='gbk')

    df = df.loc[:, ["Word", "Frequency"]]
    hot = df.head(Config.MAX_POETRY_VOCAB_SIZE)
    vocab_dict = dict()
    for row in range(hot.shape[0]):
        vocab_dict[df.loc[row,'Word']] = df.loc[row,'Frequency']
    vocab_dict['<UNK>'] = df['Frequency'].sum() - hot['Frequency'].sum()

    idx2word = [word for word in vocab_dict.keys()]
    word2idx = {word: i for i, word in enumerate(idx2word)}
    word_counts = np.array([count for count in vocab_dict.values()], dtype=np.float32)

    word_freqs = word_counts / np.sum(word_counts)
    word_freqs = word_freqs ** (3. / 4.)
    mp = {}
    mp["text"] = text
    mp["word2idx"] = word2idx
    mp['idx2word'] = idx2word
    mp['word_freqs'] = word_freqs,
    mp['word_counts'] = word_counts
    # np.save("./data/song.npz", mp)
    np.save("./data/song.npz", np.asanyarray([text, word2idx, idx2word, word_freqs, word_counts]))
    return text, word2idx, idx2word, word_freqs, word_counts


def processPoetry():
    with open(Config.poetrydata, encoding='utf-8') as f:
        text = f.read()
    text = text.replace('，', '')
    text = text.replace('。', '')

    vocab_dict = dict(Counter(text).most_common(Config.MAX_POETRY_VOCAB_SIZE - 1))
    vocab_dict['<UNK>'] = len(text) - np.sum(list(vocab_dict.values()))
    idx2word = [word for word in vocab_dict.keys()]
    word2idx = {word: i for i, word in enumerate(idx2word)}
    word_counts = np.array([count for count in vocab_dict.values()], dtype=np.float32)
    word_freqs = word_counts / np.sum(word_counts)
    word_freqs = word_freqs ** (3. / 4.)
    mp = {}
    mp["text"] = text
    mp["word2idx"] = word2idx
    mp['idx2word'] = idx2word
    mp['word_freqs'] = word_freqs,
    mp['word_counts'] = word_counts
    # np.save("./data/song.npz", mp)
    np.save("./data/song.npz", [text, word2idx, idx2word, word_freqs, word_counts])
    # torch.save(mp,'./data/song.npz')
    return text, word2idx, idx2word, word_freqs, word_counts


def processWord():
    with open(Config.worddata) as f:
        text = f.read()

    text = text.lower().split()
    vocab_dict = dict(Counter(text).most_common(Config.MAX_WORD_VOCAB_SIZE - 1))
    vocab_dict['<UNK>'] = len(text) - np.sum(list(vocab_dict.values()))
    idx2word = [word for word in vocab_dict.keys()]
    word2idx = {word: i for i, word in enumerate(idx2word)}
    word_counts = np.array([count for count in vocab_dict.values()], dtype=np.float32)
    word_freqs = word_counts / np.sum(word_counts)
    word_freqs = word_freqs ** (3. / 4.)
    return text, word2idx, idx2word, word_freqs, word_counts


if __name__ == '__main__':
    # processData()
    processPoetryWithJiayan()
