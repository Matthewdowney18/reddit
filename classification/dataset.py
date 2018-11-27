import csv
from collections import Counter

import numpy as np
import torch.utils.data
from nltk import word_tokenize
import string


class Vocab(object):
    def __init__(self, special_tokens=None):
        super(Vocab, self).__init__()

        self.nb_tokens = 0

        self.token2id = {}
        self.id2token = {}

        self.token_counts = Counter()

        self.special_tokens = []
        if special_tokens is not None:
            self.special_tokens = special_tokens
            self.add_document(self.special_tokens)

    def add_document(self, document):
        for token in document:
            self.token_counts[token] += 1

            if token not in self.token2id:
                self.token2id[token] = self.nb_tokens
                self.id2token[self.nb_tokens] = token
                self.nb_tokens += 1

    def add_documents(self, documents):
        for doc in documents:
            self.add_document(doc)

    def prune_vocab(self, min_count=2):
        nb_tokens_before = len(self.token2id)

        tokens_to_delete = set([t for t, c in self.token_counts.items() if c < min_count])
        tokens_to_delete -= set(self.special_tokens)

        for token in tokens_to_delete:
            self.token_counts.pop(token)

        self.token2id = {t: i for i, t in enumerate(self.token_counts.keys())}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.nb_tokens = len(self.token2id)

        print('Vocab pruned: {} -> {}'.format(nb_tokens_before, self.nb_tokens))

    def __getitem__(self, item):
        return self.token2id[item]

    def __contains__(self, item):
        return item in self.token2id

    def __len__(self):
        return self.nb_tokens

    def __str__(self):
        return 'Vocab: {} tokens'.format(self.nb_tokens)

class PairsDataset(torch.utils.data.Dataset):
    PAD_TOKEN = '<pad>'
    EOS_TOKEN = '</s>'
    INIT_TOKEN = '<s>'
    UNK_TOKEN = '<unk>'

    def __init__(self, filename, max_len=64, min_count=300):
        exclude = set(string.punctuation)
        sentence_1, sentence_2, labels = self._read_file(filename, max_len)
        self.sentence_1 = [[word for word in word_tokenize(s.lower())[:max_len] if
                       word not in exclude] for s in sentence_1]
        self.sentence_2 = [[word for word in word_tokenize(s.lower())[:max_len] if
                       word not in exclude] for s in sentence_2]
        self.sentence = self.sentence_2 + self.sentence_1

        self.labels = []
        for label in labels:
            if label == 'positive':
                self.labels.append(1)
            else:
                self.labels.append(0)

        self.max_len = max_len

        self.vocab = Vocab(special_tokens=[PairsDataset.PAD_TOKEN,
                                           PairsDataset.EOS_TOKEN,
                                           PairsDataset.UNK_TOKEN,
                                           PairsDataset.INIT_TOKEN])

        self.vocab.add_documents(self.sentence)
        self.vocab.prune_vocab(min_count=min_count)


    def _read_file3(self, filename):
        with open(filename, 'r', encoding='utf8') as f:
            sentence = []
            for line in f:
                if line.count(' '):
                    sentence.append(line[:-2])
        return sentence

    def _read_file(self, filename, max_len):
        with open(filename, 'r', encoding='utf8') as f:
            csv_reader = csv.reader(f, delimiter=',')
            sentence_1 = []
            sentence_2 = []
            labels = []
            i = True
            for row in csv_reader:
                if i is True:
                    i = False
                    continue
                if row[1].count(' ') < max_len or row[2].count(' ') < max_len:
                    sentence_1.append(row[1])
                    sentence_2.append(row[2])
                    labels.append(row[4])
        return sentence_1, sentence_2, labels

    def _process_sentence(self, sentence):
        sentence = sentence[:self.max_len - 1]
        sentence.append(SentenceDataset.EOS_TOKEN)

        needed_pads = self.max_len - len(sentence)
        if needed_pads > 0:
            sentence = sentence + [SentenceDataset.PAD_TOKEN] * needed_pads

        sentence = [
            self.vocab[token] if token in self.vocab else self.vocab[SentenceDataset.UNK_TOKEN]
            for token in sentence
        ]

        sentence = np.array(sentence, dtype=np.long)

        return sentence

    def __getitem__(self, index):
        sentence_1 = self._process_sentence(self.sentence_1[index])
        sentence_2 = self._process_sentence(self.sentence_2[index])
        label = self.labels[index]
        return sentence_1, sentence_2, label

    def __len__(self):
        return len(self.sentence_1)


class SentenceDataset(torch.utils.data.Dataset):
    PAD_TOKEN = '<pad>'
    EOS_TOKEN = '</s>'
    INIT_TOKEN = '<s>'
    UNK_TOKEN = '<unk>'

    def __init__(self, filename, max_len=64, min_count=300):
        exclude = set(string.punctuation)
        sentence_1, sentence_2 = self._read_file(filename, max_len)
        sentence_1 = [[word for word in word_tokenize(s.lower())[:max_len] if
                       word not in exclude] for s in sentence_1]
        sentence_2 = [[word for word in word_tokenize(s.lower())[:max_len] if
                       word not in exclude] for s in sentence_2]
        self.sentence = sentence_2 + sentence_1
        for sentence in self.sentence:
            print(sentence)

        self.max_len = max_len

        self.vocab = Vocab(special_tokens=[SentenceDataset.PAD_TOKEN,
                                           SentenceDataset.EOS_TOKEN,
                                           SentenceDataset.UNK_TOKEN,
                                           SentenceDataset.INIT_TOKEN])

        self.vocab.add_documents(self.sentence)
        self.vocab.prune_vocab(min_count=min_count)



    def _read_file(self, filename, max_len):
        with open(filename, 'r', encoding='utf8') as f:
            csv_reader = csv.reader(f, delimiter=',')
            sentence_1 = []
            sentence_2 = []
            i = True
            for row in csv_reader:
                if i is True:
                    i = False
                    continue
                if row[1].count(' ') < max_len or row[2].count(' ') < max_len:
                    sentence_1.append(row[1])
                    sentence_2.append(row[2])
        return sentence_1, sentence_2

    def _process_sentence(self, sentence):
        sentence = sentence[:self.max_len - 1]
        sentence.append(SentenceDataset.EOS_TOKEN)

        needed_pads = self.max_len - len(sentence)
        if needed_pads > 0:
            sentence = sentence + [SentenceDataset.PAD_TOKEN] * needed_pads

        sentence = [
            self.vocab[token] if token in self.vocab else self.vocab[SentenceDataset.UNK_TOKEN]
            for token in sentence
        ]

        sentence = np.array(sentence, dtype=np.long)

        return sentence

    def __getitem__(self, index):
        sentence = self._process_sentence(self.sentence[index])
        return sentence

    def __len__(self):
        return len(self.sentence)