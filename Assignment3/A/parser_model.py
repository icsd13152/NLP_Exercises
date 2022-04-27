#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2020-2021: Homework 3
parser_model.py: Feed-Forward Neural Network for Dependency Parsing
Sahil Chopra <schopra8@stanford.edu>
Haoshen Hong <haoshen@stanford.edu>
"""
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Cube(nn.Module):

    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return torch.pow(input,3)

class ParserModel(nn.Module):
    """ Feedforward neural network with an embedding layer and two hidden layers.
    The ParserModel will predict which transition should be applied to a
    given partial parse configuration.

    PyTorch Notes:
        - Note that "ParserModel" is a subclass of the "nn.Module" class. In PyTorch all neural networks
            are a subclass of this "nn.Module".
        - The "__init__" method is where you define all the layers and parameters
            (embedding layers, linear layers, dropout layers, etc.).
        - "__init__" gets automatically called when you create a new instance of your class, e.g.
            when you write "m = ParserModel()".
        - Other methods of ParserModel can access variables that have "self." prefix. Thus,
            you should add the "self." prefix layers, values, etc. that you want to utilize
            in other ParserModel methods.
        - For further documentation on "nn.Module" please see https://pytorch.org/docs/stable/nn.html.
    """
    def __init__(self, embeddings, n_features=36,
        hidden_size=200, n_classes=3, dropout_prob=0.5):
        """ Initialize the parser model.

        @param embeddings (ndarray): word embeddings (num_words, embedding_size)
        @param n_features (int): number of input features
        @param hidden_size (int): number of hidden units
        @param n_classes (int): number of output classes
        @param dropout_prob (float): dropout probability
        """
        super(ParserModel, self).__init__()
        print("emb size ",embeddings.shape[1])
        print("num of emb ",embeddings.shape[0])
        print("features ",n_features)
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.pretrained_embeddings = nn.Embedding(embeddings.shape[0],self.embed_size)
        self.pretrained_embeddings.weight = nn.Parameter(torch.tensor(embeddings))
        # self.freezed_embeddings = nn.Embedding.from_pretrained(self.pretrained_embeddings.weight,freeze=True)
        # self.emb = nn.Embedding(embeddings.shape[0],self.embed_size) #use this line to create Embeddins during training

        # self.embed_to_hidden2 = nn.Linear(self.n_features*self.embed_size, 100, bias=True)
        # nn.init.xavier_uniform_(self.embed_to_hidden2.weight) #in-place function
        # self.embed_to_hidden = nn.Linear(100, self.hidden_size, bias=True)
        #uncomment the below to get original architecture
        self.embed_to_hidden = nn.Linear(self.n_features*self.embed_size, self.hidden_size, bias=True)
        nn.init.xavier_uniform_(self.embed_to_hidden.weight) #in-place function
        # self.activation_cube = Cube()
        self.hidden_to_logits = nn.Linear(self.hidden_size, self.n_classes, bias=True)
        nn.init.xavier_uniform_(self.hidden_to_logits.weight)
        self.dropout = nn.Dropout(p=dropout_prob)
        # self.activation_func = nn.Softmax()

    def embedding_lookup(self, w):
        """ Utilize `w` to select embeddings from embedding matrix `self.embeddings`
            @param w (Tensor): input tensor of word indices (batch_size, n_features)

            @return x (Tensor): tensor of embeddings for words represented in w
                                (batch_size, n_features * embed_size)
        """
        
        x = self.pretrained_embeddings(w)
        # x = self.emb(w)
        x = x.view(x.size(0),-1)  # resize x into 2 dimensions.
        
        return x


    def forward(self, w):
        """ Run the model forward.

            Note that we will not apply the softmax function here because it is included in the loss function nn.CrossEntropyLoss

            PyTorch Notes:
                - Every nn.Module object (PyTorch model) has a `forward` function.
                - When you apply your nn.Module to an input tensor `w` this function is applied to the tensor.
                    For example, if you created an instance of your ParserModel and applied it to some `w` as follows,
                    the `forward` function would called on `w` and the result would be stored in the `output` variable:
                        model = ParserModel()
                        output = model(w) # this calls the forward function
                - For more details checkout: https://pytorch.org/docs/stable/nn.html#torch.nn.Module.forward

        @param w (Tensor): input tensor of tokens (batch_size, n_features)

        @return logits (Tensor): tensor of predictions (output after applying the layers of the network)
                                 without applying softmax (batch_size, n_classes)
        """

        embeddings = self.embedding_lookup(w)
        # print(embeddings.shape)
        # h1 = F.relu(self.embed_to_hidden2(embeddings))

        h2 = F.relu(self.embed_to_hidden(embeddings))
        # h2 = self.activation_cube(self.embed_to_hidden(embeddings))
        logits = self.hidden_to_logits(self.dropout(h2))
        # probs = self.activation_func(logits)
        return logits
