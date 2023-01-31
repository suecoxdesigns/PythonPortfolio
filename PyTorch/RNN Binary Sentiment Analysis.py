# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 06:51:17 2022

@author: sueco
"""



#%% Text Classification

#Sentiment analysis model for movie reviews. 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  RNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



# load the data

from torchtext.legacy.datasets import text_classification
import torchtext
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm, trange
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

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


# rearrange the data to the right format

def collator_fn(batch):
    labels = torch.tensor([example[0] for example in batch])
    sentences = [example[1] for example in batch]
    data = pad_sequence(sentences)
    real_sentc_length = [len(example[1]) for example in batch]  
    
    return [data, labels], real_sentc_length

BATCH_SIZE = 128

train_loader = torch.utils.data.DataLoader(agnews_train, batch_size = BATCH_SIZE, shuffle=True, collate_fn = collator_fn)
test_loader = torch.utils.data.DataLoader(agnews_test, batch_size = BATCH_SIZE, shuffle=False, collate_fn = collator_fn)


# define model parameters

VOCAB_SIZE = len(agnews_train.get_vocab())
EMBED_DIM = 300
HIDDEN_DIM = 64
NUM_OUTPUTS = len(agnews_train.get_labels())
NUM_EPOCHS = 10


class RNNModule(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_dim, num_outputs):
        super().__init__()
        self.vocab_size=vocab_size
        self.embedding_size=embedding_size
        self.hidden_dim=hidden_dim
        self.num_outputs=num_outputs
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_dim, num_layers=1,batch_first=False)
        self.lin = nn.Linear(hidden_dim, num_outputs)

    def forward(self, x, real_seq_length):
        embed = self.embedding(x)

        # transmitting the sequence lengths to the padded "package"
        embed_packed = pack_padded_sequence(embed, real_seq_length, enforce_sorted=False)          
        h0 = torch.zeros(1, embed.size(1), self.hidden_dim)
        out, hidden = self.rnn(embed_packed,h0)

        # If you read on the documentation, the hidden state in a RNN the hidden state of the
        # last timestep. Which, according to https://towardsdatascience.com/pytorch-basics-how-to-train-your-neural-net-intro-to-rnn-cb6ebc594677
        # is equal to the
        # last position in the output. You can do either hidden.squeeze(0) or
        # try to access output with something like this:
        ### unpack the sequence with pad_packed_sequence:
        ### output_padded, output_lengths = pad_packed_sequence(out)
        ### and then provide the array output_padded[:, real_seq_length, :].
        ### You might need to mess around with it though, since lists do not accept
        ### other lists as indexes.
       
        h = self.lin(hidden.squeeze(0))             
        
        return h             

# Instantiate model
model = RNNModule(VOCAB_SIZE,EMBED_DIM,HIDDEN_DIM,NUM_OUTPUTS)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Iterate through train set minibatchs 
for epoch in range(NUM_EPOCHS):
    correct = 0
    num_examples = 0
    for (inputs, labels), real_sentences_lengths in tqdm(train_loader):
        # Zero out the gradients
        optimizer.zero_grad()
        
        # Forward pass
        y = model(inputs, real_sentences_lengths)
        loss = criterion(y, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())
        num_examples += len(labels)
    # Print training progress
    acc = correct/num_examples
    print("Epoch: {0} \t Train Loss: {1} \t Train Acc: {2}".format(epoch, loss, acc))

## Testing
correct = 0
num_test = 0

with torch.no_grad():
    # Iterate through test set minibatchs 
    for  (inputs, labels), real_sentc_length in tqdm(test_loader):
        # Forward pass
        y = model(inputs, real_sentc_length)
        
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())
        num_test += len(labels)
    
print('Test accuracy: {}'.format(correct/num_test))

