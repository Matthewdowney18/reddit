import numpy as np
import torch
import torch.utils.data
import time

from dataset import SentenceDataset, PairsDataset
from model import Seq2SeqModelAttention
from utils import variable, cuda, argmax, get_sentence_from_indices, \
    get_pretrained_embeddings, save_checkpoint, load_checkpoint, freeze_layer, \
    accuracy


def main():
    nb_epochs = 50
    batch_size = 300
    hidden_size = 256
    embedding_dim = 300
    pretrained_embeddings = None
    max_grad_norm = 5
    max_len = 20
    min_count = 2
    weight_decay = 0.00001
    model_group = "/classifier"
    model_name = "/classifier_1"
    project_file = "/home/mattd/PycharmProjects/reddit"
    dataset_path = '/home/mattd/datasets/AskReddit/'
    # embedding_filename = 'embeddings_20_1.npy's

    string = 'nb_epochs: {}\nbatch_size: {}\nhidden_size: {}\nembedding_dim: ' \
             '{}\npretrained_embeddings: {}\nmax_len: {}\nmin_countmin_count: '\
             '{}\nweight_decay: {}\nmodel_group: {}\nmodel_name: {}\n'.format(
                nb_epochs, batch_size, hidden_size, embedding_dim,
                pretrained_embeddings, max_len, min_count, weight_decay,
                model_group, model_name)
    print(string)
    output = string + '\n'

    # embedding_filename = 'embeddings_20_1.npy's'

    model_filename = '{}{}s{}'.format(
        project_file, model_group, model_name)

    description_filename = \
        '{}/description/description_1.txt'.format(project_file)

    output_file = '{}{}_outputs{}'.format(
        project_file, model_group, model_name)

    string = ''

    # eng_fr_filename = '/mnt/data1/datasets/yelp/merged/train'
    dataset_train_filename = "{}train.csv".format(dataset_path)
    dataset_val_filename = "{}validation.csv".format(dataset_path)

    dataset_train = SentenceDataset(dataset_train_filename, max_len, min_count)
    dataset_val = SentenceDataset(dataset_val_filename, max_len, min_count,
                               dataset_train.vocab)

    string = 'Train {}, val: {}'.format(len(dataset_train), len(dataset_val))
    print(string)
    output += string + '\n'


    #embeddings_dir = '{}/{}'.format(project_file, embedding_filename)
    #pretrained_embeddings = cuda(
    #    get_pretrained_embeddings(embeddings_dir, dataset))
    #embedding_dim = pretrained_embeddings.shape[1]

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size, shuffle=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size, shuffle=False)

    vocab_size = len(dataset_val.vocab)
    padding_idx = dataset_val.vocab[SentenceDataset.PAD_TOKEN]
    init_idx = dataset_val.vocab[SentenceDataset.INIT_TOKEN]

    model = Seq2SeqModelAttention(hidden_size, padding_idx, init_idx,
        max_len, vocab_size, embedding_dim, pretrained_embeddings)

    model = cuda(model)

    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, amsgrad=True, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset_val.vocab[
        SentenceDataset.PAD_TOKEN])

    model, optimizer, lowest_loss, description, last_epoch, \
    train_loss, val_loss = load_checkpoint(model_filename, model, optimizer)

    print(description)
    output = output + str(description) + '\n'
    outfile = open(output_file, 'w')
    outfile.write(output)
    outfile.close()

    phases = ['train', 'val', ]
    data_loaders = [data_loader_train, data_loader_val, ]


    for epoch in range(last_epoch, last_epoch+nb_epochs):
        start = time.clock()

        #if epoch == 6:
        #    model.unfreeze_embeddings()
        #    parameters = list(model.parameters())
        #    optimizer = torch.optim.Adam(
        #        parameters, amsgrad=True, weight_decay=weight_decay)

        for phase, data_loader in zip(phases, data_loaders):
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = []
            epoch_sentenence_accuracy = []
            epoch_token_accuracy = []

            for i, inputs in enumerate(data_loader):
                optimizer.zero_grad()

                inputs = variable(inputs)
                targets = variable(inputs)

                outputs = model.auto_encoder(inputs, targets)

                targets = targets.view(-1)
                outputs = outputs.view(targets.size(0), -1)

                loss = criterion(outputs, targets)

                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
                    optimizer.step()

                else:
                    predicted = torch.argmax(outputs.view(-1, max_len,
                        vocab_size), -1)
                    batch_sentenence_accuracy, batch_token_accuracy = accuracy(
                        targets.view(-1, max_len), predicted)
                    epoch_sentenence_accuracy.append(batch_sentenence_accuracy)
                    epoch_token_accuracy.append(batch_token_accuracy)
                epoch_loss.append(float(loss))

            epoch_loss = np.mean(epoch_loss)

            if phase == 'train':
                train_loss.append(epoch_loss)
                string = ('Epoch {:03d} | {} loss: {:.3f}'.format(
                    epoch, phase, epoch_loss))

                print(string, end='\n')
                output = output + string + '\n'
            else:
                averege_epoch_sentenence_accuracy = sum(epoch_sentenence_accuracy) / \
                    len(epoch_sentenence_accuracy)
                averege_epoch_token_accuracy = sum(epoch_token_accuracy) / \
                    len(epoch_token_accuracy)
                time_taken = time.clock() - start

                val_loss.append(epoch_loss)
                string = ' {} loss: {:.3f} | time: {:.3f}'.format(
                    phase, epoch_loss, time_taken)
                print(string, end='')
                output = output + string + '\n'

                string = '| sentence accuracy:{:.3f}| token accuracy:{:.3f}'.format(
                    averege_epoch_sentenence_accuracy, averege_epoch_token_accuracy)
                print(string, end='\n')
                output = output + string + '\n'
                if epoch_loss < lowest_loss:
                    save_checkpoint(
                        model, epoch_loss, optimizer, model_filename,
                        description_filename, epoch, train_loss, val_loss)
                    lowest_loss = epoch_loss

            # print random sentence
            if phase == 'val':
                random_idx = np.random.randint(len(dataset_val))
                inputs = dataset_val[random_idx]
                targets = inputs
                inputs_var = variable(inputs)

                outputs_var = model.auto_encoder(inputs_var.unsqueeze(0)) #
                # unsqueeze to
                # get the batch dimension
                outputs = argmax(outputs_var).squeeze(0).data.cpu().numpy()

                string = '> {}\n'.format(get_sentence_from_indices(
                    inputs, dataset_val.vocab, SentenceDataset.EOS_TOKEN))

                string = string + u'= {}\n'.format(get_sentence_from_indices(
                    targets, dataset_val.vocab, SentenceDataset.EOS_TOKEN))

                string = string + u'< {}'.format(get_sentence_from_indices(
                    outputs, dataset_val.vocab, SentenceDataset.EOS_TOKEN))
                print(string, end='\n')
                output = output + string + '\n' + '\n'
        outfile = open(output_file, 'w')
        outfile.write(output)
        outfile.close()


if __name__ == '__main__':
    main()
