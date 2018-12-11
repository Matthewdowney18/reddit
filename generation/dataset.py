import csv
from collections import Counter

import numpy as np
import torch.utils.data
import random


def _read_file(filename, max_len):
    with open(filename, 'r', encoding='utf8') as f:
        csv_reader = csv.reader(f, delimiter=',')
        sentence_1 = []
        sentence_2 = []
        for row in csv_reader:
            if row[0].count(' ') < max_len or row[1].count(' ') < max_len:
                sentence_1.append(row[0].split(' '))
                sentence_2.append(row[1].split(' '))
    return sentence_1, sentence_2



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


class SentenceDataset(torch.utils.data.Dataset):
    PAD_TOKEN = '<pad>'
    EOS_TOKEN = '</s>'
    INIT_TOKEN = '<s>'
    UNK_TOKEN = '<unk>'

    def __init__(self, filename, max_len=64, min_count=300, vocab=None):
        self.sentence_1, self.sentence_2 = _read_file(filename, max_len-1)

        self.max_len = max_len

        if vocab is None:
            self.vocab = Vocab(special_tokens=[SentenceDataset.PAD_TOKEN,
                                               SentenceDataset.EOS_TOKEN,
                                               SentenceDataset.UNK_TOKEN,
                                               SentenceDataset.INIT_TOKEN])
        else:
            self.vocab = vocab

        self.vocab.add_documents(self.sentence_1)
        self.vocab.add_documents(self.sentence_2)
        self.vocab.prune_vocab(min_count=min_count)

    def prune_examples(self, num_examples):
        examples = list(zip(self.sentence_1, self.sentence_2))
        random.shuffle(examples)
        examples = examples[:num_examples]
        self.sentence_1, self.sentence_2 = zip(*examples)
        self.sentence = self.sentence_1 + self.sentence_2

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
        return sentence_1, sentence_2

    def __len__(self):
        return len(self.sentence_1)
