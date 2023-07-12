## Classification with CNN

Convolutional layers are used to find patterns by sliding small kernel window over input. Instead of multiplying the filters on the small regions of the images, it slides through embedding vectors of few words as mentioned by window size. For looking at sequences of word embeddings, the window has to look at multiple word embeddings in a sequence. They will be rectangular with size window_size * embedding_size. For example, if window size is 3 then kernel will be 3*500. This essentially represents n-grams in the model. The kernel weights (filter) are multiplied to word embeddings in pairs and summed up to get output values. As the network is being learned, these kernel weights are also being learned.

We will be using convolutional network with pre-trained word2vec models for classification. We implement a convolutional neural network for text classification similar to the CNN-rand baseline described by [Kim (2014)](https://aclanthology.org/D14-1181.pdf). We use pre-trained word2vec models for feasibility of finding appropriate embeddings. The architecture of our model looks like :

<p align="center"><img src="https://cezannec.github.io/assets/cnn_text/complete_text_classification_CNN.png" width="75%" align="center"></p>

We will be using an Embedding layer loaded with a word2vec model, followed by a convolution layer, and a linear layer.



## Classification with RNN


We will be using recurrent neural networks with pre-trained word2vec models for classification. We use pre-trained word2vec models for feasibility of finding appropriate embeddings. The architecture of our model looks like :

<p align="center"><img src="https://www.tensorflow.org/static/text/tutorials/images/bidirectional.png" width="75%" align="center"></p>

We will be using an Embedding layer loaded, followed by a RNN layer, and a linear layer.

We will would be using the Clickbait and Web of science dataset for this task.