import numpy as np
import torch

torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn


class Word2Vec(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocabulary_size = 0

    def tokenize(self, data):
        tokens = [N.split() for N in data]
        return tokens


    def create_vocabulary(self, tokenized_data):
        unique_words = sorted(set(sum(tokenized_data, [])))
        w2id = {w: i for i, w in enumerate(unique_words)}
        id2w = {i: w for i, w in enumerate(unique_words)}

        self.vocabulary_size = len(unique_words)
        self.word2idx = w2id
        self.idx2word = id2w


    def skipgram_embeddings(self, tokenized_data, window_size=2):
        source_tokens = []
        target_tokens = []
        for sent in tokenized_data:
            for i in range(len(sent)):
                for j in range(-window_size, window_size + 1):
                    if (i + j >= 0) and (i + j < len(sent)) and (j != 0):
                        source_tokens.append([self.word2idx[sent[i]]])
                        target_tokens.append([self.word2idx[sent[i + j]]])

        return source_tokens, target_tokens
    

    def cbow_embeddings(self, tokenized_data, window_size=2):
        source_tokens = []
        target_tokens = []
        for sent in tokenized_data:
            for i in range(len(sent)):
                target_tokens.append([self.word2idx[sent[i]]])
                tmp_target_list = []
                for j in range(-window_size, window_size + 1):
                    if (i + j >= 0) and (i + j < len(sent)) and (j != 0):
                        tmp_target_list.append(self.word2idx[sent[i + j]])

                source_tokens.append(tmp_target_list)

        return source_tokens, target_tokens


class SkipGram_Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(SkipGram_Model, self).__init__()
        self.EMBED_DIMENSION = 300 # please use this to set embedding_dim in embedding layer
        self.EMBED_MAX_NORM = 1    # please use this to set max_norm in embedding layer
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.EMBED_DIMENSION,
                                      max_norm=self.EMBED_MAX_NORM
                                      )
        self.linear = nn.Linear(self.EMBED_DIMENSION, self.vocab_size)

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.linear(x)

        return x


class CBOW_Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(CBOW_Model, self).__init__()
        self.EMBED_DIMENSION = 300  # please use this to set embedding_dim in embedding layer
        self.EMBED_MAX_NORM = 1     # please use this to set max_norm in embedding layer
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.EMBED_DIMENSION,
                                      max_norm=self.EMBED_MAX_NORM
                                      )
        self.linear = nn.Linear(self.EMBED_DIMENSION, self.vocab_size)

    def forward(self, inputs):
       # print(inputs)
        x = self.embedding(inputs)
       # print(x.shape)
        x = torch.mean(x, 0)
        x = x.unsqueeze(0)
       # print(x.shape)
        x = self.linear(x)
      #  print(x.shape)
        return x
