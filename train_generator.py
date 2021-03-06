import numpy as np
import torch
import torch.utils.data
import time
from tqdm import tqdm


from generation.dataset import SentenceDataset
from generation.model import Seq2SeqModel, Seq2SeqModelAttention
from utils import variable, cuda, argmax, get_sentence_from_indices, \
    get_pretrained_embeddings, save_checkpoint, load_checkpoint, freeze_layer, \
    encoder_accuracy, bleu, load_params, check_files


def main():
    file = {
    "model_group" : "/seq_len_exp",
    "model_name" : "/generation_6",
    "model_version" : 0,
    "project_file": "/home/mattd/PycharmProjects/reddit/generation"}

    file["dataset_path"] = "{}/data/".format(file["project_file"])

    file["model_filename"] = '{}{}s{}_{}'.format(file["project_file"],
        file["model_group"], file["model_name"], file["model_version"])

    file["output_file"] = '{}{}_outputs{}_{}'.format(file["project_file"],
        file["model_group"], file["model_name"], file["model_version"])

    #check_files(file)

    use_old_model = file["model_version"] != 0
    params = {}

    if use_old_model:
        file["old_model_filename"] = '{}{}s{}_{}'.format(file["project_file"],
            file["model_group"], file["model_name"], file["model_version"] - 1)
        params, old_files = load_params(file["old_model_filename"])
        use_old_model = old_files != {}

    if not use_old_model:
        params = {
        "attention" : True,
        "batch_size" : 325,
        "hidden_size" : 256,
        "embedding_dim" : 300,
        "pretrained_embeddings" : True,
        "max_grad_norm" : 5,
        "max_len" : 30,
        "min_count": 2,
        "weight_decay": 0.00001,
        "learning_rate": 0.005,
        }

    params["num_training_examples"] = 78260
    params["num_val_examples"] = -1
    params["nb_epochs"] = 40

    if params["pretrained_embeddings"]:
        file["pretrained_embeddings_file"] = \
            "/embeddings/embeddings_min{}_max{}.npy".format(
            params["min_count"], params["max_len"])

    string= ""
    for k, v in file.items():
        string += "{}: {}\n".format(k, v)
    for k, v in params.items():
        string += "{}: {}\n".format(k, v)

    print(string)
    output = string + '\n'

    # eng_fr_filename = '/mnt/data1/datasets/yelp/merged/train'
    dataset_train_filename = "{}train.csv".format(file["dataset_path"])
    dataset_val_filename = "{}validation.csv".format(file["dataset_path"])

    dataset_train = SentenceDataset(
        dataset_train_filename, params["max_len"], params["min_count"])
    dataset_val = SentenceDataset(dataset_val_filename,
        params["max_len"], params["min_count"], dataset_train.vocab)

    string = 'Vocab size {}\n'.format(len(dataset_train.vocab))
    string += 'Train {} '.format(len(dataset_train))

    if params["num_training_examples"] != -1:
        dataset_train.prune_examples(params["num_training_examples"])
        string += '-> {}'.format(len(dataset_train))

    string += '\nVal: {}'.format(len(dataset_val))

    if params["num_val_examples"] != -1:
        dataset_val.prune_examples(params["num_val_examples"])
        string += '-> {}'.format(len(dataset_val))

    print(string)
    output += string + '\n'

    if params["pretrained_embeddings"]:
        embeddings_dir = '{}{}'.format(
            file["project_file"], file["pretrained_embeddings_file"])
        pretrained_embeddings = cuda(get_pretrained_embeddings(embeddings_dir))
        params["embedding_dim"] = pretrained_embeddings.shape[1]
    else: pretrained_embeddings = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, params["batch_size"], shuffle=True)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, params["batch_size"], shuffle=False)

    vocab_size = len(dataset_train.vocab)
    padding_idx = dataset_train.vocab[SentenceDataset.PAD_TOKEN]
    init_idx = dataset_train.vocab[SentenceDataset.INIT_TOKEN]

    if params["attention"] is True:
        model = Seq2SeqModelAttention(params["hidden_size"], padding_idx,
            init_idx, params["max_len"], vocab_size, params["embedding_dim"],
            pretrained_embeddings)
    else:
        model = Seq2SeqModel(params["hidden_size"], padding_idx, init_idx,
            params["max_len"], vocab_size, params["embedding_dim"],
            pretrained_embeddings)

    model = cuda(model)

    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, amsgrad=True,
        weight_decay=params["weight_decay"], lr=params["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    if use_old_model:
        model, optimizer= load_checkpoint(
            file["old_model_filename"], model, optimizer)

    lowest_loss = 100
    train_loss = []
    val_loss = []
    best_model = model
    best_optimizer = optimizer
    average_epoch_loss = 0

    metrics = {"token_accuracy": [],
               "sentence_accuracy": [],
               "perplexity": [],
               "bleu": {'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': []}}

    outfile = open(file["output_file"], 'w')
    outfile.write(output)
    outfile.close()

    phases = ['train', 'val', ]
    data_loaders = [data_loader_train, data_loader_val, ]

    intervals = 2
    highest_acc = 0

    for epoch in range(0, params["nb_epochs"]):
        start = time.clock()
        string = 'Epoch: {}\n'.format(epoch)
        print(string, end='')
        output = output + '\n' + string

        #if epoch == 6:
        #    model.unfreeze_embeddings()
        #    parameters = list(model.parameters())
        #    optimizer = torch.optim.Adam(
        #        parameters, amsgrad=True, weight_decay=weight_decay)

        for phase, data_loader in zip(phases, data_loaders):
            if phase == 'train':
                model.train()
                string = 'Train: \n'
            else:
                model.eval()
                string = 'Validation \n'

            print(string, end='')
            output = output + '\n' + string

            epoch_bleu = {'bleu_1':[], 'bleu_2': [], 'bleu_3': [], 'bleu_4':[]}
            epoch_loss = []
            epoch_sentenence_accuracy = []
            epoch_token_accuracy = []
            j = 1

            for i, (sentence_1, sentence_2) in tqdm(enumerate(data_loader)):
                optimizer.zero_grad()

                sentence_1 = variable(sentence_1)
                sentence_2 = variable(sentence_2)

                outputs = model(sentence_1, sentence_2)

                targets = sentence_2.view(-1)
                outputs = outputs.view(targets.size(0), -1)
                loss = criterion(outputs, targets)

                epoch_loss.append(float(loss))
                average_epoch_loss = np.mean(epoch_loss)

                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters,
                        params["max_grad_norm"])
                    optimizer.step()
                    if (len(data_loader) / intervals)*j <= i+1:
                        string = (
                            'Example {:03d} | {} loss: {:.3f}'.format(
                              i, phase, average_epoch_loss))
                        print(string, end='\n')
                        output = output + string + '\n'
                        j += 1
                else:
                    # get result metrics
                    predicted = torch.argmax(
                        outputs.view(-1, params["max_len"], vocab_size), -1)
                    batch_sentence_accuracy, batch_token_accuracy = \
                        encoder_accuracy(
                        targets.view(-1, params["max_len"]), predicted)
                    epoch_sentenence_accuracy.append(batch_sentence_accuracy)
                    epoch_token_accuracy.append(batch_token_accuracy)

                    bleu_1 = bleu(targets.view(-1, params["max_len"]), predicted, 1)
                    bleu_2 = bleu(targets.view(-1, params["max_len"]), predicted, 2)
                    bleu_3 = bleu(targets.view(-1, params["max_len"]), predicted, 3)
                    bleu_4 = bleu(targets.view(-1, params["max_len"]), predicted, 4)
                    epoch_bleu["bleu_1"].append(bleu_1)
                    epoch_bleu["bleu_2"].append(bleu_2)
                    epoch_bleu["bleu_3"].append(bleu_3)
                    epoch_bleu["bleu_4"].append(bleu_4)

            if phase == 'val':
                time_taken = time.clock() - start

                val_loss.append(average_epoch_loss)

                string = ' {} loss: {:.3f} | time: {:.3f}'.format(
                    phase, average_epoch_loss, time_taken)
                string += ' | lowest loss: {:.3f} highest accuracy: ' \
                    '{:.3f}'.format(lowest_loss, highest_acc)
                print(string, end='\n')
                output = output + '\n' + string + '\n'

                average_epoch_sentenence_accuracy = np.mean(
                    epoch_sentenence_accuracy)
                average_epoch_token_accuracy = np.mean(
                    epoch_token_accuracy)
                perplexity = np.exp(average_epoch_loss)

                average_epoch_bleu = {'bleu_1': np.mean(epoch_bleu['bleu_1']),
                                      'bleu_2': np.mean(epoch_bleu['bleu_2']),
                                      'bleu_3': np.mean(epoch_bleu['bleu_3']),
                                      'bleu_4': np.mean(epoch_bleu['bleu_4'])}

                metrics["token_accuracy"].append(average_epoch_token_accuracy)
                metrics["sentence_accuracy"].append(
                    average_epoch_sentenence_accuracy)
                metrics["perplexity"].append(perplexity)
                metrics["bleu"]["bleu_1"].append(average_epoch_bleu['bleu_1'])
                metrics["bleu"]["bleu_2"].append(average_epoch_bleu['bleu_2'])
                metrics["bleu"]["bleu_3"].append(average_epoch_bleu['bleu_3'])
                metrics["bleu"]["bleu_4"].append(average_epoch_bleu['bleu_4'])


                if average_epoch_loss < lowest_loss:
                    best_model = model
                    best_optimizer = optimizer
                    best_epoch = epoch
                    lowest_loss = average_epoch_loss

                save_checkpoint(
                    best_epoch, best_model, best_optimizer,
                    epoch, model, optimizer, train_loss, val_loss, metrics,
                    params, file)


                if average_epoch_token_accuracy > highest_acc:
                    highest_acc = average_epoch_token_accuracy

                string = "Token_accuracy: {:.3f}\nSentence_accuracy: {:.3f}\n".format(
                    average_epoch_token_accuracy, average_epoch_sentenence_accuracy)
                string += "Perplexity: {:.3f}\n".format(perplexity)
                string += "Bleu: {}\n".format(average_epoch_bleu)
                print(string, end='\n')
                output = output + string + '\n'

                random_idx = np.random.randint(len(dataset_val))
                sentence_1, sentence_2 = dataset_val[random_idx]
                sentence_1_var = variable(sentence_1)
                sentence_2_var = variable(sentence_2)

                outputs_var = model(sentence_1_var.unsqueeze(0)) # unsqueeze
                #  to get the batch dimension
                outputs = argmax(outputs_var).squeeze(0).data.cpu().numpy()

                string = '> {}\n'.format(get_sentence_from_indices(
                    sentence_1, dataset_val.vocab, SentenceDataset.EOS_TOKEN))

                string = string + u'< {}\n'.format(get_sentence_from_indices(
                    outputs, dataset_val.vocab, SentenceDataset.EOS_TOKEN))

                string = string + u'= {}\n'.format(get_sentence_from_indices(
                    sentence_2, dataset_val.vocab, SentenceDataset.EOS_TOKEN))

                print(string, end='\n\n')
                output = output + string + '\n' + '\n'
            else:
                train_loss.append(average_epoch_loss)
        outfile = open(file["output_file"], 'w')
        outfile.write(output)
        outfile.close()


if __name__ == '__main__':
    main()
