import pathlib

import pandas as pd
import numpy as np
from wordcloud import WordCloud, ImageColorGenerator  # , STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image
from jiayan import PMIEntropyLexiconConstructor  # 常用的古汉语分词器
import jieba  # 常用的中文分词器

from models.word2vec.config import Config


def csv2txt(txtPath, df):
    with open(txtPath, 'a', encoding='utf-8') as f:
        for i in df['内容'].values:
            f.write(i)


def processData():
    for csv in pathlib.Path(Config.dataPath).glob('*.csv'):
        name = Config.dataPath + "/" + csv.name
        df = pd.read_csv(name, encoding='utf-8')
        csv2txt(Config.data, df)


if __name__ == '__main__':
    processData()
