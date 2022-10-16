from gensim.similarities import WmdSimilarity
import jieba
from gensim.models import Word2Vec

corpus = []
documents = []
# 停用词载入
stopwords = []
# stopword = open('data/stopword.txt', 'r', encoding='utf-8')
# for line in stopword:
#     stopwords.append(line.strip())

# 加载模型
model = Word2Vec.load("checkpoint/song.model")
# 已经分好词并且已经去掉停用词的训练集文件
# f = open(r'./data/song.txt_cut.txt', 'r', encoding='utf-8')
# lines = f.readlines()
# # 建立语料库list文件（list中是已经分词后的）
# for each in lines:
#     text = list(each.replace('\n', '').split(' '))
#     # pri
#     nt(text)
#     corpus.append(text)
# print(len(corpus))l
#
# 未分词的原始train文件
# 建立相对应的原始语料库语句list文件（未分词）
with open(r'./data/song.txt', 'r', encoding='utf-8') as f_1:
    f_1.readline()
    for test_line in f_1:
        # print(test_line)
        test_line = test_line.replace("\n", "")
        documents.append(test_line)

num_best = 10
mp = {}
for sen in documents[:10000]:
    distance = model.wv.wmdistance("漫卷诗书喜欲狂", sen)
    mp[sen] = distance

after = sorted(mp.items(), key=lambda item: item[1])

print(after[:10])
