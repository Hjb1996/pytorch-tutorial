# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     1.word-embedding
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2020/4/6
   Description :   wordembedding
==================================================
"""

"""
链接:https://pan.baidu.com/s/1tFeK3mXuVXEy3EMarfeWvg 密码:v2z5
在这一份notebook中，我们会（尽可能）尝试复现论文Distributed Representations of Words and Phrases and their Compositionality中训练词向量的方法. 我们会实现Skip-gram模型，并且使用论文中noice contrastive sampling的目标函数。
这篇论文有很多模型实现的细节，这些细节对于词向量的好坏至关重要。我们虽然无法完全复现论文中的实验结果，主要是由于计算资源等各种细节原因，但是我们还是可以大致展示如何训练词向量。
以下是一些我们没有实现的细节

"""
__author__ = 'songdongdong'

import torch
import  torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
import torch.nn.parameter as Paramerter

from collections import Counter
import numpy as np
import random
import math

import  pandas as pd
import  scipy
import  sklearn
from sklearn.metrics.pairwise import  cosine_similarity

import utils


USE_CUDA  = torch.cuda.is_available()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

if USE_CUDA:
    torch.cuda.manual_seed(0)

# 设定一些超参数

K = 100  # number of negative samples
C = 3  # nearby words threshold
NUM_EPOCHS = 2  # The number of epochs of training
MAX_VOCAB_SIZE = 30000  # the vocabulary size
BATCH_SIZE = 128  # the batch size
LEARNING_RATE = 0.2  # the initial learning rate
EMBEDDING_SIZE = 100


LOG_FILE = "../../log/word-embedding.log"

# tokenize函数，把一篇文本转化成一个个单词
def word_tokenize(text):
    return text.split()



word_to_idx, idx_to_word, word_counts = utils.get_id_words("", MAX_VOCAB_SIZE)


with open("../data//text8.train.txt", "r", encoding="utf-8") as fin:
    text = fin.read()
text = [w for w in word_tokenize(text.lower())]
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word: i for i, word in enumerate(idx_to_word)}
word_counts = np.array([count for count in vocab.values()], dtype=np.float32)


word_freqs = word_counts/np.sum(word_counts)
word_freqs = word_freqs **(3./4.)
word_freqs = word_freqs/np.sum(word_freqs)
VOCAB_SIZE = len(idx_to_word)


class WordEmbeddingDataset(tud.Dataset):
    """
实现Dataloader
一个dataloader需要以下内容：
     把所有text编码成数字，然后用subsampling预处理这些文字。
     保存vocabulary，单词count，normalized word frequency
    每个iteration sample一个中心词
    根据当前的中心词返回context单词
    根据中心词sample一些negative单词
    返回单词的counts
这里有一个好的tutorial介绍如何使用PyTorch dataloader. 为了使用dataloader，我们需要定义以下两个function:

     __len__ function需要返回整个数据集中有多少个item
     __get__ 根据给定的index返回一个item
     有了dataloader之后，我们可以轻松随机打乱整个数据集，拿到一个batch的数据等等。
    """
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        ''' text: a list of words, all text from the training dataset
            word_to_idx: the dictionary from word to idx
            idx_to_word: idx to word mapping
            word_freq: the frequency of each word
            word_counts: the word counts
        '''
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word_to_idx.get(t, VOCAB_SIZE - 1) for t in text]
        self.text_encoded = torch.Tensor(self.text_encoded).long()
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

    def __len__(self):
        ''' 返回整个数据集（所有单词）的长度
        '''
        return len(self.text_encoded)

    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的(positive)单词
            - 随机采样的K个单词作为negative sample
        '''
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))  # 周围词
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]  # 取余，防止超出 vocab的范围
        pos_words = self.text_encoded[pos_indices]  # 周围单词
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)  # 负采样,对 input做 几次采样

        return center_word, pos_words, neg_words



dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  #windows环境下 只用 0


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        ''' 初始化输出和输出embedding
        '''
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        initrange = 0.5 / self.embed_size
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.out_embed.weight.data.uniform_(-initrange, initrange)

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.in_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_labels, pos_labels, neg_labels):
        '''
        input_labels: 中心词, [batch_size]
        pos_labels: 中心词周围 context window 出现过的单词 [batch_size * (window_size * 2)]
        neg_labelss: 中心词周围没有出现过的单词，从 negative sampling 得到 [batch_size, (window_size * 2 * K)]

        return: loss, [batch_size]
        '''

        batch_size = input_labels.size(0)

        input_embedding = self.in_embed(input_labels)  # B * embed_size
        pos_embedding = self.out_embed(pos_labels)  # B * (2*C) * embed_size
        neg_embedding = self.out_embed(neg_labels)  # B * (2*C * K) * embed_size

        log_pos = torch.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze()  # B * (2*C)
        log_neg = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze()  # B * (2*C*K)

        log_pos = F.logsigmoid(log_pos).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)  # batch_size

        loss = log_pos + log_neg

        return -loss

    def input_embeddings(self): #输出 embedding
        return self.in_embed.weight.data.cpu().numpy()


model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()


def evaluate(filename, embedding_weights):
    if filename.endswith(".csv"):
        data = pd.read_csv(filename, sep=",")
    else:
        data = pd.read_csv(filename, sep="\t")
    human_similarity = []
    model_similarity = []
    for i in data.iloc[:, 0:2].index:
        word1, word2 = data.iloc[i, 0], data.iloc[i, 1]
        if word1 not in word_to_idx or word2 not in word_to_idx:
            continue
        else:
            word1_idx, word2_idx = word_to_idx[word1], word_to_idx[word2]
            word1_embed, word2_embed = embedding_weights[[word1_idx]], embedding_weights[[word2_idx]]
            model_similarity.append(float(sklearn.metrics.pairwise.cosine_similarity(word1_embed, word2_embed)))
            human_similarity.append(float(data.iloc[i, 2])) #csv中的相似度

    return scipy.stats.spearmanr(human_similarity, model_similarity)# , model_similarity,统计两列数据的相关性

def find_nearest(word):
    index = word_to_idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx_to_word[i] for i in cos_dis.argsort()[:10]]



"""
训练模型：

模型一般需要训练若干个epoch
每个epoch我们都把所有的数据分成若干个batch
把每个batch的输入和输出都包装成cuda tensor
forward pass，通过输入的句子预测每个单词的下一个单词
用模型的预测和正确的下一个单词计算cross entropy loss
清空模型当前gradient
backward pass
更新模型参数
每隔一定的iteration输出模型在当前iteration的loss，以及在验证数据集上做模型的评估

"""
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
for a in range(0, NUM_EPOCHS):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        if USE_CUDA:
            input_labels = input_labels.cuda()
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("epoch: {}, iter: {}, loss: {}".format(a, i, loss.item()))

        if i % 1000 == 0:
            embedding_weights = model.input_embeddings()
            sim_simlex = evaluate("embedding/simlex-999.txt", embedding_weights)
            sim_men = evaluate("embedding/men.txt", embedding_weights=embedding_weights)
            sim_353 = evaluate("embedding/wordsim353.csv", embedding_weights)
            print("epoch: {}, iteration: {}, simlex-999: {}, men: {}, sim353: {}, nearest to monster: {}\n".format(
                a, i, sim_simlex, sim_men, sim_353, find_nearest("monster")))

        if i == 1000:
            break
        embedding_weights = model.input_embeddings()
        np.save("embedding-{}".format(EMBEDDING_SIZE), embedding_weights)
        torch.save(model.state_dict, "embedding-{}.th".format(EMBEDDING_SIZE))


model.load_state_dict(torch.load("embedding-{}.th".format(EMBEDDING_SIZE)))

#在 MEN 和 Simplex-999 数据集上做评估
embedding_weights = model.input_embeddings()
print("simlex-999", evaluate("simlex-999.txt", embedding_weights))
print("men", evaluate("men.txt", embedding_weights))
print("wordsim353", evaluate("wordsim353.csv", embedding_weights))

#寻赵nearest neighbors

for word in ['good','fresh',"monster","green","like","chicago"]:
    print(word,find_nearest(word))

#单词之间的关系

man_idx = word_to_idx["man"]
king_idx = word_to_idx["king"]
woman_idx = word_to_idx["woman"]
embedding = embedding_weights[woman_idx] - embedding_weights[man_idx] + embedding_weights[king_idx]
cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
for i in cos_dis.argsort()[:20]:
    print(idx_to_word[i])