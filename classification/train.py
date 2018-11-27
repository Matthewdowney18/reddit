import numpy as np
import torch
import torch.utils.data
import time

from classification.dataset import PairsDataset
from classification.model import Seq2SeqModelAttention
from classification.utils import variable, cuda, argmax, get_sentence_from_indices, \
    get_pretrained_embeddings, save_checkpoint, load_checkpoint, freeze_layer, \
    accuracy


def main():
    nb_epochs = 100
    batch_size = 400
    hidden_size = 256
    embedding_dim = 300
    pretrained_embeddings = None
    max_len = 10
    min_count = 1
    max_grad_norm = 5
    val_len = 200
    weight_decay = 0.00001
    use_old_model = True
    model_group = "/auto_encoder"
    model_name = "/autoencoder_1"
    project_file = "/home/mattd/PycharmProjects/reddit"
    dataset_filename = "RR_negative.csv"
    # embedding_filename = 'embeddings_20_1.npy'

    model_filename = '{}{}s{}'.format(
        project_file, model_group, model_name)

    description_filename = \
        '{}/description/description_1.txt'.format(project_file)

    output_file = '{}{}_outputs{}'.format(
        project_file, model_group, model_name)

    # eng_fr_filename = '/mnt/data1/datasets/yelp/merged/train'
    eng_fr_filename = '/home/mattd/PycharmProjects/agreement_encoder' \
                      '/RR_negative.csv'

    dataset = PairsDataset(eng_fr_filename, max_len, min_count)
    string = 'Dataset: {}'.format(len(dataset))
    print(string)
    output = string + '\n'

    train_len = len(dataset) - val_len
    dataset_train, dataset_val = torch.utils.data.dataset.random_split(dataset, [train_len, val_len])

    string = 'Train {}, val: {}'.format(len(dataset_train), len(dataset_val))
    print(string)
    output = output + string + '\n'

    #embeddings_dir = '/home/mattd/pycharm/encoder' \
    #                 '/embeddings_3min.npy'
    #pretrained_embeddings = cuda(
    #    get_pretrained_embeddings(embeddings_dir, dataset))
    #embedding_dim = pretrained_embeddings.shape[1]

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size, shuffle=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size, shuffle=False)

    vocab_size = len(dataset.vocab)
    padding_idx = dataset.vocab[PairsDataset.PAD_TOKEN]
    init_idx = dataset.vocab[PairsDataset.INIT_TOKEN]

    model = Seq2SeqModelAttention(hidden_size, padding_idx, init_idx,
        max_len, vocab_size, embedding_dim, pretrained_embeddings)

    model = cuda(model)

    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, amsgrad=True, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset.vocab[
        PairsDataset.PAD_TOKEN])

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

            for i, (sentence_1, sentence_2, labels) in enumerate(data_loader):
                optimizer.zero_grad()

                sentence_1 = variable(sentence_1)
                sentence_2 = variable(sentence_2)
                targets = variable(labels)

                outputs = model(sentence_1, sentence_2, targets)

                targets = targets.view(-1)
                outputs = outputs.view(targets.size(0), -1)
                loss = criterion(outputs, targets)

                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
                    optimizer.step()

            epoch_loss = np.mean(epoch_loss)

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
                    sentence_1, dataset.vocab, PairsDataset.EOS_TOKEN))

                string = string + u'> {}\n'.format(get_sentence_from_indices(
                    sentence_2, dataset.vocab, PairsDataset.EOS_TOKEN))

                string = string + u'target:{}  output:{}'.format(targets,
                         outputs)
                print(string, end='\n\n')
                output = output + string + '\n' + '\n'
        outfile = open(output_file, 'w')
        outfile.write(output)
        outfile.close()


if __name__ == '__main__':
    main()