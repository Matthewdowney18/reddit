import torch
from matplotlib import pyplot

from utils import load_checkpoint
from classification.model import Seq2SeqModel
from classification.dataset import SentenceDataset

def plot_data(sample_1, sample_2):
    pyplot.figure(1)
    pyplot.plot(sample_1, 'r-', sample_2, 'b-')
    pyplot.title('i dont know man')
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.show()


def main():
    num_training_examples = -1
    hidden_size = 256
    embedding_dim = 300
    pretrained_embeddings = None
    max_len = 30
    min_count = 2
    max_grad_norm = 5
    val_len = 500
    weight_decay = 0.00001
    model_group = "/classifier_examples"
    model_name = "/classifier_{}".format(num_training_examples)
    project_file = "/home/mattd/PycharmProjects/reddit_classification"
    dataset_path = "/home/mattd/PycharmProjects/reddit_classification/data/"

    model_filename = '{}{}s{}_{}'.format(
        project_file, model_group, model_name, model_version - 1)


    eng_fr_filename = '/home/mattd/pycharm/encoder/RRall.csv'
    dataset = SentenceDataset(eng_fr_filename, max_len, min_count)

    vocab_size = len(dataset.vocab)
    padding_idx = dataset.vocab[SentenceDataset.PAD_TOKEN]
    init_idx = dataset.vocab[SentenceDataset.INIT_TOKEN]


    model = Seq2SeqModel(hidden_size, padding_idx, init_idx,
            max_len, vocab_size, embedding_dim, pretrained_embeddings)

    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, amsgrad=True,
                                 weight_decay=weight_decay)

    model, optimizer, lowest_loss, description, last_epoch, \
    train_loss_1, val_loss_1 = load_checkpoint(model_filename_1, model,
                                               optimizer)


    #model = Seq2SeqModelAttention(
    #    pretrained_embeddings, hidden_size, padding_idx, init_idx,
    #    max_len, vocab_size, embedding_dim)

    #parameters = list(model.parameters())
    #optimizer = torch.optim.Adam(parameters, amsgrad=True,
    #                             weight_decay=weight_decay)

    #model, optimizer, lowest_loss, description, last_epoch, \
    #train_loss_2, val_loss_2 = load_checkpoint(model_filename_2, model,
    #                                          optimizer)

    plot_data(train_loss_1, val_loss_1)
    #plot_data(train_loss_2, val_loss_2)

if __name__ == '__main__':
    main()