import torch
torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNN(nn.Module):
    def __init__(self, vocab, num_classes):
        super(RNN, self).__init__()
        self.embed_len = 50  # embedding_dim default value for embedding layer
        self.hidden_dim = 75 # hidden_dim default value for rnn layer
        self.n_layers = 2    # num_layers default value for rnn
        self.vocab = vocab
        self.num_classes = num_classes

        self.emb = nn.Embedding(len(self.vocab), self.embed_len)
        self.rnn = nn.RNN(self.embed_len, self.hidden_dim, self.n_layers, bidirectional=True, batch_first=True)
        self.lin = nn.Linear(2*self.hidden_dim, self.num_classes)
        self.dropout = nn.Dropout(0.2)


    def forward(self, inputs, inputs_len):
        x = self.emb(inputs)

        x = torch.nn.utils.rnn.pack_padded_sequence(x,inputs_len, batch_first=True, enforce_sorted=False)

        output, hidden = self.rnn(x)  # RNN
        x = torch.cat([hidden[self.n_layers-1,:,:], hidden[self.n_layers,:,:]], dim=1)

        x = self.lin(x)
        x = self.dropout(x)

        return x

