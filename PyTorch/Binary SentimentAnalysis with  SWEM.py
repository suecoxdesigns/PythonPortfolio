# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 06:51:17 2022

@author: sueco
"""



#%% Text Classification

# Here we will build a sentiment analysis model for movie reviews: positive or negative


#%%  Loading AG News with Torchtext

    
import torchtext

from torchtext.legacy.datasets import text_classification
import torchtext

ngrams = 1
train_csv_path = 'datasets/ag_news_csv/test.csv'
test_csv_path = 'datasets/ag_news_csv/train.csv'
vocab = torchtext.vocab.build_vocab_from_iterator(
    text_classification._csv_iterator(train_csv_path, ngrams),specials=["<unk>"])
vocab.set_default_index(vocab['<unk>'])
train_data, train_labels = text_classification._create_data_from_iterator(
        vocab, text_classification._csv_iterator(train_csv_path, ngrams, yield_cls=True), False)
test_data, test_labels = text_classification._create_data_from_iterator(
        vocab, text_classification._csv_iterator(test_csv_path, ngrams, yield_cls=True), False)
if len(train_labels ^ test_labels) > 0:
    raise ValueError("Training and test labels don't match")
agnews_train = text_classification.TextClassificationDataset(vocab, train_data, train_labels)
agnews_test = text_classification.TextClassificationDataset(vocab, test_data, test_labels)


#%% Checking data format

    
print(agnews_train[0])
    
print("Length of the first text example: {}".format(len(agnews_train[0][1])))
print("Length of the second text example: {}".format(len(agnews_train[1][1])))

# lets pad the first two sequences to the same length.

from torch.nn.utils.rnn import pad_sequence

padded_exs = pad_sequence([agnews_train[0][1], agnews_train[1][1]])


#%% define a batch loader with padding


import numpy as np
import torch

def collator(batch):
    labels = torch.tensor([example[0] for example in batch])   # example[0] are the labels
    sentences = [example[1] for example in batch]               # example[1] are the sentences
    data = pad_sequence(sentences)
    
    return[data,labels]


#Create our DataLoader 

BATCH_SIZE = 128

train_loader = torch.utils.data.DataLoader(agnews_train, batch_size = BATCH_SIZE, shuffle=True, collate_fn = collator)
# # take in a batch of size BATCH_SIZE
# shuffle it
# apply the collator - which pulls out labels and sentences
# and then padds the sentences to make them all the same length

test_loader = torch.utils.data.DataLoader(agnews_test, batch_size = BATCH_SIZE, shuffle=False, collate_fn = collator)


#%%%%%%%%%%%%%%%%%%%%%%%  SWEM   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%  Simple Word Embedding Model
# learning word embeddings

VOCAB_SIZE = len(agnews_train.get_vocab())
EMBED_DIM = 100
HIDDEN_DIM = 64
NUM_OUTPUTS = len(agnews_train.get_labels())
NUM_EPOCHS = 10

#%%  Now define the model

import torch.nn as nn
import torch.nn.functional as F

class SWEM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_dim, num_outputs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_size)
        #amkes a lookup table that stores embeddings of a fixed dictionary size
        # often used to store word embeddings and retreive them using indicies
        # input: list of indices.  Output: corresponding word embeddings
        
        self.fc1 = nn.Linear(embedding_size, hidden_dim)
        # applies a linear transformation to the incoming data 
        # output = x W1 + b1
        #in_features (size of each input sample): here: embedding_size
        #out_features - size of each output sample (here hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_outputs)
        #same as above but now
        #input size = size of hidden dim
        #output size = size of outputs: here the number of labels (1 for each word)
        
    def forward(self, x):
        embed = self.embedding(x)
        #pull out the embeddings using the self.embedding fucntion defined above
        #output: 1 embedding for each work of length EMBED_DIM
        embed_mean = torch.mean(embed, dim=0)
        #average these embeddings row wise 
        #input BATCH_SIZE x EMBED_DIM
        #output 1 x BATCH_SIZE
        
        h = self.fc1(embed_mean)
        #Run the mean value through the 1st linear function
        #input: 1 x BATCH_SIZE
        # output 1 x hidden_dim
        h = F.relu(h)
        #put through the non-linear function
        # input 1 x hidden_dim
        # output 1 x hidden_dim
        h = self.fc2(h)
        # put through second linear function
        # input 1 x hidden_dim
        # output 1 x num_outputs (i.e. one for each word label)
        return h
    

#%% Running the model


# Instantiate model
model = SWEM(
    vocab_size = VOCAB_SIZE,
    embedding_size = EMBED_DIM,
    hidden_dim = HIDDEN_DIM,
    num_outputs = NUM_OUTPUTS
)



# Binary cross-entropy (BCE) Loss and Adam Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Iterate through train set minibatchs 
for epoch in range(NUM_EPOCHS):
    correct = 0
    num_examples = 0
    for inputs, labels in train_loader:
        # Zero out the gradients
        optimizer.zero_grad()
        
        # Forward pass
        y = model(inputs)
        #estimate word embedding labels
        loss = criterion(y, labels)
        #figure out how far off from the true values these estimate are
        
        # Backward pass
        loss.backward()
        #determine how to move in the search space to make better estiamtes
        optimizer.step()
        # make better estimates
        
        predictions = torch.argmax(y, dim=1)  #This needed to be argmax 
        # returns the indicies of the maximum values of all elements in the input tensor
        # Use the estimated values to make predicitons of what categories each of the news article is in
        correct += torch.sum((predictions == labels).float())
        # see how many we got right
        num_examples += len(inputs)
    
    # Print training progress
    if epoch % 1 == 0:
        acc = correct/num_examples
        print("Epoch: {0} \t Train Loss: {1} \t Train Acc: {2}".format(epoch, loss, acc))



## Testing
correct = 0
num_test = 0

with torch.no_grad():
    # Iterate through test set minibatchs 
    for inputs, labels in test_loader:
        # Forward pass
        y = model(inputs)
        
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())
        num_test += len(inputs)
    
print('Test accuracy: {}'.format(correct/num_test))



