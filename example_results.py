import numpy as np
import torch
import torch.utils.data
import pandas as pd

from dataset import PairsDataset
from model import Seq2SeqModel

from utils import cuda, get_pretrained_embeddings, load_checkpoint

def main():
    hidden_size = 256
    embedding_dim = 300
    pretrained_embeddings = None
    max_len = 30
    min_count = 2
    weight_decay = 0.00001
    learning_rate = 0.005
    use_autoencoder_model = False
    dataset_path = "/home/mattd/PycharmProjects/reddit/data/"
    project_file = "/home/mattd/PycharmProjects/reddit"
    model_group = "/classifier_example"

    num_trials = 3
    val_lengths = [-1, 300000, 250000, 200000]

    results = {}

    trials = range(0, num_trials)

    dataset_train_filename = "{}train.csv".format(dataset_path)
    dataset_val_filename = "{}validation.csv".format(dataset_path)

    dataset_train = PairsDataset(
        dataset_train_filename, max_len, min_count)
    dataset_val = PairsDataset(dataset_val_filename, max_len, min_count,
                               dataset_train.vocab)

    vocab_size = len(dataset_train.vocab)
    padding_idx = dataset_train.vocab[PairsDataset.PAD_TOKEN]
    init_idx = dataset_train.vocab[PairsDataset.INIT_TOKEN]


    model = Seq2SeqModel(hidden_size, padding_idx, init_idx,
                         max_len, vocab_size, embedding_dim,
                         pretrained_embeddings)

    model = cuda(model)

    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(
        parameters, amsgrad=True, weight_decay=weight_decay, lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for length in val_lengths:
        results[length] = []
        for trial in trials:
            model_filename = '{}{}s/classifier_{}_{}_{}'.format(
                project_file, model_group, trial, length, 0)
            model, optimizer, lowest_loss, description, last_epoch, train_loss,\
                val_loss, found_model, metrics = load_checkpoint(
                model_filename, model, optimizer, use_autoencoder_model)
            results[length].append(lowest_loss)
        results[length].append(np.mean(results[length]))

    df = pd.DataFrame()
if __name__ == '__main__':
    main()


