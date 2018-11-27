import numpy as np
import torch
from torch.autograd import Variable
import os
from torch.nn import Parameter


def get_sequences_lengths(sequences, masking=0, dim=1):
    if len(sequences.size()) > 2:
        sequences = sequences.sum(dim=2)

    masks = torch.ne(sequences, masking)

    lengths = masks.sum(dim=dim)

    return lengths


def cuda(obj):
    if torch.cuda.is_available():
        obj = obj.cuda()
        #obj = obj
    return obj


def variable(obj, volatile=False):
    if isinstance(obj, (list, tuple)):
        return [variable(o, volatile=volatile) for o in obj]

    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)

    obj = cuda(obj)
    obj = Variable(obj, volatile=volatile)
    return obj


def argmax(inputs, dim=-1):
    values, indices = inputs.max(dim=dim)
    return indices


def get_sentence_from_indices(indices, vocab, eos_token, join=True):
    tokens = []
    for idx in indices:
        token = vocab.id2token[idx]

        if token == eos_token:
            break

        tokens.append(token)

    if join:
        tokens = ' '.join(tokens)

    tokens = tokens

    return tokens

def get_pretrained_embeddings(embeddings_dir, dataset):
    embeddings = np.load(embeddings_dir)
    emb_tensor = torch.FloatTensor(embeddings)
    return emb_tensor

def get_description(description_filename):
    if os.path.exists(description_filename):
        description = list()
        with open(description_filename, 'r', encoding='utf8') as f:
            for line in f:
                description.append(line)
        return description
    else:
        return ['none']

def save_checkpoint(model, loss, optimizer, filename, description_filename,
                    epoch, train_loss, val_loss):
    state = {
            'state_dict': model.state_dict(),
            'loss': loss,
            'optimizer' : optimizer.state_dict(),
            'description' : get_description(description_filename),
            'epoch' : epoch,
            'train_loss' : train_loss,
            'val_loss' : val_loss
        }
    torch.save(state, filename)



def load_checkpoint(filename, model, optimizer):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss = checkpoint['loss']
        description = checkpoint['description']
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
        else:
            epoch = 0
        if 'train_loss' in checkpoint:
            train_loss = checkpoint['train_loss']
            val_loss = checkpoint['val_loss']
        else:
            train_loss = []
            val_loss = []
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        loss = 500
        description = ['none']
        epoch = 0
        train_loss = []
        val_loss = []

    return model, optimizer, loss, description, epoch, train_loss, val_loss


def freeze_layer(layer, bool):
    for param in layer.parameters():

        param.requires_grad = not bool
    layer.training = not bool
    return layer

def accuracy(targets, predicted):
    batch_size = targets.size()
    sentence_acc = [1] * batch_size[0]
    token_acc = []
    for batch in range(0, batch_size[0]):
        for token in range(0, batch_size[1]):
            if targets[batch, token] != 0:
                if targets[batch, token] != predicted[batch, token]:
                    sentence_acc[batch] = 0
                    token_acc.append(0)
                else:
                    token_acc.append(1)

    sentence_accuracy = sum(sentence_acc)/len(sentence_acc)
    token_accuracy = sum(token_acc) / len(
        token_acc)
    return sentence_accuracy, token_accuracy