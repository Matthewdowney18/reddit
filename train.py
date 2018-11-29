import numpy as np
import torch
import torch.utils.data
import time
from tqdm import tqdm

from dataset import PairsDataset
from model import Seq2SeqModelAttention
from utils import variable, cuda, argmax, get_sentence_from_indices, \
    get_pretrained_embeddings, save_checkpoint, load_checkpoint, freeze_layer, \
    accuracy


def main():
    nb_epochs = 50
    batch_size = 400
    hidden_size = 256
    embedding_dim = 300
    pretrained_embeddings = None
    max_grad_norm = 5
    max_len = 20
    min_count = 2
    weight_decay = 0.00001
    learning_rate = 0.001
    use_autoencoder_model = True
    model_group = "/classifier"
    model_name = "/classifier_1"
    autoencoder_location = "/auto_encoders/auto_encoder_1"
    project_file = "/home/mattd/PycharmProjects/reddit"
    dataset_path = '/home/mattd/datasets/AskReddit/'
    # embedding_filename = 'embeddings_20_1.npy'

    string = 'nb_epochs: {}\nbatch_size: {}\nhidden_size: {}\nembedding_dim: ' \
             '{}\npretrained_embeddings: {}\nmax_len: {}\nmin_countmin_count: '\
             '{}\nweight_decay: {}\nlearning_rate: {}\nmodel_group: ' \
             '{}\nmodel_name: {}\nautoencoder_location: {}\n'.format(
        nb_epochs, batch_size, hidden_size, embedding_dim,
        pretrained_embeddings, max_len, min_count, weight_decay,
        learning_rate, model_group, model_name, autoencoder_location)
    print(string)
    output = string + '\n'

    model_filename = '{}{}s{}'.format(
        project_file, model_group, model_name)

    autoencoder_filename = '{}{}'.format(project_file, autoencoder_location)

    if use_autoencoder_model:
        load_filename = autoencoder_filename
    else:
        load_filename = model_filename

    description_filename = \
        '{}/description/description_1.txt'.format(project_file)

    output_file = '{}{}_outputs{}'.format(
        project_file, model_group, model_name)

    # eng_fr_filename = '/mnt/data1/datasets/yelp/merged/train'
    dataset_train_filename = "{}train.csv".format(dataset_path)
    dataset_val_filename = "{}validation.csv".format(dataset_path)

    dataset_train = PairsDataset(dataset_train_filename, max_len, min_count)
    dataset_val = PairsDataset(dataset_val_filename, max_len, min_count,
                               dataset_train.vocab)

    string = 'Train {}, val: {}'.format(len(dataset_train), len(dataset_val))
    print(string)
    output += string + '\n'

    #embeddings_dir = '/home/mattd/pycharm/encoder' \
    #                 '/embeddings_3min.npy'
    #pretrained_embeddings = cuda(
    #    get_pretrained_embeddings(embeddings_dir, dataset))
    #embedding_dim = pretrained_embeddings.shape[1]

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size, shuffle=True)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size, shuffle=False)

    vocab_size = len(dataset_train.vocab)
    padding_idx = dataset_train.vocab[PairsDataset.PAD_TOKEN]
    init_idx = dataset_train.vocab[PairsDataset.INIT_TOKEN]

    model = Seq2SeqModelAttention(hidden_size, padding_idx, init_idx,
        max_len, vocab_size, embedding_dim, pretrained_embeddings)

    model = cuda(model)

    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(
        parameters, amsgrad=True, weight_decay=weight_decay, lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset_val.vocab[
        PairsDataset.PAD_TOKEN])

    model, optimizer, lowest_loss, description, last_epoch, train_loss,\
        val_loss, found_model = load_checkpoint(load_filename, model,
        optimizer, use_autoencoder_model)

    #print previous model info
    if found_model:
        string = 'Loaded Model:\nlowest_validation_loss: {}\ndescription: {}' \
                 '\nlast_epoch'.format(lowest_loss, description, last_epoch)
    else: 
        string = 'No model found at {}\n'.format(load_filename)

    print(string)
    output = output + string + '\n'

    outfile = open(output_file, 'w')
    outfile.write(output)
    outfile.close()

    phases = ['train', 'val', ]
    data_loaders = [data_loader_train, data_loader_val, ]

    intervals = 6

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
            j = 1

            for i, (sentence_1, sentence_2, labels) in tqdm(enumerate(
                    data_loader)):
                optimizer.zero_grad()

                sentence_1 = variable(sentence_1)
                sentence_2 = variable(sentence_2)
                targets = variable(labels)

                outputs = model(sentence_1, sentence_2, targets)

                targets = targets.view(-1)
                outputs = outputs.view(targets.size(0), -1)
                loss = criterion(outputs, targets)

                epoch_loss.append(float(loss))
                average_epoch_loss = np.mean(epoch_loss)

                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
                    optimizer.step()
                    if (len(data_loader) / intervals)*j <= i:
                        train_loss.append(average_epoch_loss)
                        string = (
                            'Epoch {:03d} Example {:03d} | {} loss: {:.3f}'.format(
                             epoch, i, phase, average_epoch_loss))
                        print(string, end='\n')
                        output = output + string + '\n'
                        j += 1


            if phase == 'train':
                train_loss.append(epoch_loss)
                string = ('Epoch {:03d} | {} loss: {:.3f}'.format(
                    epoch, phase, epoch_loss))

                print(string, end='\n')
                output = output + string + '\n'
            else:
                time_taken = time.clock() - start

                val_loss.append(epoch_loss)
                string = ' {} loss: {:.3f} | time: {:.3f}'.format(
                    phase, epoch_loss, time_taken)
                print(string, end='\n')
                output = output + string + '\n'


                if epoch_loss < lowest_loss:
                    save_checkpoint(
                        model, epoch_loss, optimizer, model_filename,
                        description_filename, epoch, train_loss, val_loss)
                    lowest_loss = epoch_loss

            # print random sentence
            if phase == 'val':
                time_taken = time.clock() - start

                val_loss.append(average_epoch_loss)
                string = ' {} loss: {:.3f} | time: {:.3f}'.format(
                    phase, average_epoch_loss, time_taken)
                print(string, end='')
                output = output + '\n' + string + '\n'

                if average_epoch_loss < lowest_loss:
                    save_checkpoint(
                        model, average_epoch_loss, optimizer, model_filename,
                        description_filename, epoch, train_loss, val_loss)
                    lowest_loss = average_epoch_loss

                random_idx = np.random.randint(len(dataset_val))
                sentence_1, sentence_2, labels = dataset_val[random_idx]
                targets = labels
                sentence_1_var = variable(sentence_1)
                sentence_2_var = variable(sentence_2)

                outputs_var = model(sentence_1_var.unsqueeze(0),
                                    sentence_2_var.unsqueeze(0)) # unsqueeze
                #  to get the batch dimension
                outputs = outputs_var.squeeze(0).data.cpu().numpy()

                string = '> {}\n'.format(get_sentence_from_indices(
                    sentence_1, dataset_val.vocab, PairsDataset.EOS_TOKEN))

                string = string + u'> {}\n'.format(get_sentence_from_indices(
                    sentence_2, dataset_val.vocab, PairsDataset.EOS_TOKEN))

                string = string + u'target:{}  output:{}'.format(targets,
                         outputs)
                print(string, end='\n\n')
                output = output + string + '\n' + '\n'
        outfile = open(output_file, 'w')
        outfile.write(output)
        outfile.close()


if __name__ == '__main__':
    main()