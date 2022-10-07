import torch
import torch.utils.data as tud

class WordEmbeddingDataSet(tud.Dataset):
    def __init__(self, text, word2idx, idx2word, word_freqs, word_counts, C, K):
        '''
        text: 文本数据
        word2idx: 字到id的映射
        idx2word: id到字的映射
        word_freq: 单词出现的频率
        '''
        super(WordEmbeddingDataSet, self).__init__()
        self.C = C
        self.K = K
        # 存储每一个单词的index
        self.text_encoded = [word2idx.get(word, word2idx['<UNK>']) for word in text]
        # self.text_encoded = [word2idx.get(word) for word in text]
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):
        center_word = self.text_encoded[idx]
        # 返回中心词周围若干个词的index
        pos_indices = list(range(idx - self.C, idx)) + list(range(idx, idx + self.C))
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        # 拿到实际的index
        pos_words = self.text_encoded[pos_indices]
        # 随机选取若干负样本
        neg_words = torch.multinomial(self.word_freqs, self.K * pos_words.shape[0], True)
        return center_word, pos_words, neg_words