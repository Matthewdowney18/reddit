from nltk import word_tokenize
import string
import csv
import re

'''
this file takes the csv of the reddit_classification dataset, and it prepares it for training.
first it strips the text of urls numbers, and then it takes out puncuation, 
lower cases the words, and then it tokenizes, and joins the tokens with 
spaces to make it easy to do things ye dawg 
'''


def print_file(data, filename):
    with open(filename, 'w', encoding='utf8') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerows(data)

    f.close()


def process_sentence(sentence):
    regexp_patterns = [
        ("\(*https*://.*\)*", ' _LINK_ '),
        ('[+-]?([0-9]*[.])?[0-9]+', ' _NUM_ ')]
    for pattern in regexp_patterns:
        for i, s in enumerate(sentence):
            sentence[i] = re.sub(pattern[0], pattern[1], s)

    exclude = set(string.punctuation)
    sentence = [" ".join([word for word in word_tokenize(s.lower()) if
                        word not in exclude]) for s in sentence]

    return sentence


def read_file(filename):
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

            sentence_1.append(row[0])
            sentence_2.append(row[1])
            labels.append(int(row[2]))
            i = True
    f.close()
    return sentence_1, sentence_2, labels


def make_data(dataset_path, data_train_path):
    sentence_1, sentence_2, labels = read_file(dataset_path)
    sentence_2 = process_sentence(sentence_2)
    sentence_1 = process_sentence(sentence_1)
    data = [list(x) for x in zip(sentence_1, sentence_2, labels)]
    #data = [list(x) for x in zip(sentence_1, sentence_2)]
    print_file(data, data_train_path)


def main():
    dataset_path = '/home/mattd/datasets/AskReddit/'
    train_data_path = "/home/mattd/PycharmProjects/reddit/data/"
    make_data('{}{}'.format(dataset_path, 'validation.csv'),
              '{}{}'.format(train_data_path, 'validation.csv'))


if __name__ == '__main__':
    main()
