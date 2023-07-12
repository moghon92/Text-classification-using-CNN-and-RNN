import torch
torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, w2vmodel, num_classes, window_sizes=(1,2,3,5)):
        super(CNN, self).__init__()
        self.weights = torch.from_numpy(w2vmodel.wv.vectors).type(torch.float32) # use this to initialize the embedding layer
        self.EMBEDDING_SIZE = 500  # Use this to set the embedding_dim in embedding layer
        self.NUM_FILTERS = 10      # Number of filters in CNN
        self.window_sizes = window_sizes
        self.num_classes = num_classes

        self.emb = nn.Embedding.from_pretrained(self.weights, padding_idx=3060)
        self.convs = nn.ModuleList()

        for i in range(len(window_sizes)):
            self.convs.append(nn.Conv2d(in_channels=1,
                                        out_channels=self.NUM_FILTERS,
                                        kernel_size=(self.window_sizes[i], self.EMBEDDING_SIZE),
                                        padding=(self.window_sizes[i] - 1, 0)))

        self.linear = nn.Linear(self.NUM_FILTERS*len(window_sizes), self.num_classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.emb(x)
        x = x.unsqueeze(1)
        #print(x.shape)

        out = []
        for c_layer in self.convs:

            y = c_layer(x)
            y = F.tanh(y).squeeze(3)
            #print(y.shape)
            y = F.max_pool1d(y, y.size(2)).squeeze(2)
            #print(y.shape)
            out.append(y)


        #print(x.shape)
        x = torch.cat(out, 1)
        x = self.linear(x)
        x = F.softmax(x, dim=1)

        return x
