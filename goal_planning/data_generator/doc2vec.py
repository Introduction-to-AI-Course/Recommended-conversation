#!/usr/bin/python
# -*- coding: UTF-8 -*-
from gensim import corpora, models, similarities
from collections import defaultdict
import jieba
import re
jieba.load_userdict("../origin_data/user_dict.txt")
#添加自定义分词词典
class Doc2Vec(object):
    def __init__(self, documents):
        self.stop_words = self.stop_words()
        self.word_dict, self.texts = self.dictionary_generator(documents)
        self.UNK = self.word_dict["unk"]

    def stop_words(self):
        stop_words = list()
        with open("../origin_data/stop_words.txt", "r") as f:
            for line in f.readlines():
                stop_words.append(line.replace("\n", ""))
        return stop_words
        #停用词表

    def text_generator(self, documents):
        texts_idx = list()
        for doc in documents:
            doc_idx = list()
            for line in doc:
                line_idx = list()
                line = self.remove_punctuation(line)
                words = ' '.join(jieba.cut(line)).split(' ')
                for word in words:
                    line_idx.append(self.word_dict.get(word, self.UNK))
                doc_idx.append(line_idx)
            texts_idx.append(doc_idx)
        return texts_idx

    def dictionary_generator(self, documents):
        documents = [self.remove_punctuation(line) for doc in documents for line in doc]
        #先去除文档每一行的标点符号
        texts = list()
        for line in documents:
            text= list()
            words = ' '.join(jieba.cut(line)).split(' ')
            #对每一行分词
            for word in words:
                # if word not in self.stop_words:
                text.append(word)
            #行列表
            texts.append(text)
            #包含行列表的文章列表
        frequency = defaultdict(int)
        #创建统计字典
        for text in texts:
            for word in text:
                frequency[word] += 1
        #计算每个单词的频率
        docs = [[word for word in text if frequency[word] > 5] for text in texts]
        #对每行挑出在全文中频率大于5的词
        dictionary = corpora.Dictionary(docs)
        #利用筛选完的docs列表创建字典对象,为每个分词生成独立的编号Key
        corpus = [dictionary.doc2bow(text) for text in docs]
        #对每行生成这样的格式[（分词1的ID，分词在该行出现的频率）]，实际上这是词袋模型
        word_dict = dict()
        # word_dict["pad"] = 0
        for k, v in dictionary.items():
            word_dict[v] = k
        #生成字典，key:分词原词，value:分词编号
        word_dict["unk"] = len(word_dict)
        #分词个数
        # for k, v in word_dict.items():
        #     print(k, v)
        print(len(word_dict))
        return word_dict, texts

    def remove_punctuation(self, line):#正则处理，去除标点符号
        line = re.sub("\[\d*\]", "", line)
        return re.sub('[^\u4e00-\u9fa5^a-z^A-Z^0-9]', '', line)

    
