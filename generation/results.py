import torch
import pandas as pd
import matplotlib.pyplot as plt
import os

def fix_loss(loss):
    loss = [b for a, b in enumerate(loss) if a%2 == 0]
    return loss

def get_results(filename):
    results = {}
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        results["params"] = checkpoint["params"]
        results["epoch"] = checkpoint["epoch"]
        results["best_epoch"] = checkpoint["best_epoch"]
        results["metrics"] = checkpoint["metrics"]
        results["metrics"]["train_loss"] = fix_loss(checkpoint["train_loss"])
        #results["metrics"]["train_loss"] = checkpoint["train_loss"]
        results["metrics"]["val_loss"] = checkpoint["val_loss"]
        results["file"] = checkpoint["file"]
    else:
        print("error with file")
        exit()
    return results

def plot(data, lengths):
    bleu_df = pd.DataFrame.from_dict(data)
    bleu_df.set_axis(lengths, axis='index', inplace=True)
    bleu_df.sort_index(axis=0, inplace=True)
    bleu_df.index.name = 'max_len'

    bleu_df.plot(subplots=True)
    plt.show()

def plot_loss(results):
    loss = {"train" : results["metrics"]["train_loss"],
            "val" : results["metrics"]["val_loss"]}
    df = pd.DataFrame.from_dict(loss)
    df.index.name = 'epoch'
    df.plot()
    plt.title(results["file"]["model_name"])
    plt.yticks(range(3, 8))
    plt.show()

def main():
    file = {
        "model_group": "/seq_len_exp",
        "project_file": "/home/mattd/PycharmProjects/reddit/generation"}

    file["dataset_path"] = "{}/data/".format(file["project_file"])

    file["models_directory"] = '{}{}s/results/'.format(file["project_file"],
                                                 file["model_group"])

    bleu_scores = {"bleu_1" : [], "bleu_2" : [], "bleu_3" : [], "bleu_4" : []}
    perplexities = {"perplexity" : []}
    lengths = []

    models = os.listdir(file["models_directory"])
    for model in models:
        results = get_results(file["models_directory"]+model)
        best_epoch = results["best_epoch"]
        lengths.append(int(results["params"]["max_len"]))
        bleu_scores["bleu_1"].append(
            results["metrics"]["bleu"]["bleu_1"][best_epoch])
        bleu_scores["bleu_2"].append(
            results["metrics"]["bleu"]["bleu_2"][best_epoch])
        bleu_scores["bleu_3"].append(
            results["metrics"]["bleu"]["bleu_3"][best_epoch])
        bleu_scores["bleu_4"].append(
            results["metrics"]["bleu"]["bleu_4"][best_epoch])

        perplexities["perplexity"].append(
            results["metrics"]["perplexity"][best_epoch])

        plot_loss(results)


    plot(bleu_scores, lengths)
    plot(perplexities, lengths)


if __name__ == '__main__':
    main()
