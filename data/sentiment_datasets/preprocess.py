import numpy as np
import re
import sys
import os
import pickle
import random
from gensim.models.keyedvectors import KeyedVectors

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec using gensim
    """
    word_vecs = {}
    model = KeyedVectors.load_word2vec_format(fname, binary=True)
    for word in model.vocab:
        if word in vocab:
            word_vecs[word] = model.get_vector(word)
    return word_vecs

def line_to_words(line, dataset):
    if dataset == 'SST1' or dataset == 'SST2':
        clean_line = clean_str_sst(line.strip())
    else:
        clean_line = clean_str(line.strip())
    words = clean_line.split(' ')
    words = words[1:]

    return words

def get_vocab(file_list, dataset=''):
    max_sent_len = 0
    word_to_idx = {}
    # Starts at 1 for padding
    idx = 1

    for filename in file_list:
        f = open(filename, "rb")
        for line in f:
            line = line.decode('latin-1')
            words = line_to_words(line, dataset)
            max_sent_len = max(max_sent_len, len(words))
            for word in words:
                if not word in word_to_idx:
                    word_to_idx[word] = idx
                    idx += 1

        f.close()

    return max_sent_len, word_to_idx

def load_data(dataset, train_name, test_name='', dev_name='', padding=4):
    """
    Load training data (dev/test optional).
    """
    f_names = [train_name]
    if not test_name == '': f_names.append(test_name)
    if not dev_name == '': f_names.append(dev_name)

    max_sent_len, word_to_idx = get_vocab(f_names, dataset)

    dev = []
    dev_label = []
    train = []
    train_label = []
    test = []
    test_label = []

    files = []
    data = []
    data_label = []

    f_train = open(train_name, 'rb')
    files.append(f_train)
    data.append(train)
    data_label.append(train_label)
    if not test_name == '':
        f_test = open(test_name, 'rb')
        files.append(f_test)
        data.append(test)
        data_label.append(test_label)
    if not dev_name == '':
        f_dev = open(dev_name, 'rb')
        files.append(f_dev)
        data.append(dev)
        data_label.append(dev_label)

    for d, lbl, f in zip(data, data_label, files):
        for line in f:
            line = line.decode('latin-1')
            words = line_to_words(line, dataset)
            y = int(line.strip().split()[0])
            sent = [word_to_idx[word] for word in words]
            # end padding
            if len(sent) < max_sent_len + padding:
                sent.extend([0] * (max_sent_len + padding - len(sent)))
            # start padding
            sent = [0]*padding + sent

            d.append(sent)
            lbl.append(y)

    f_train.close()
    if not test_name == '':
        f_test.close()
    if not dev_name == '':
        f_dev.close()

    return word_to_idx, np.array(train, dtype=np.int32), np.array(train_label, dtype=np.int32), np.array(test, dtype=np.int32), np.array(test_label, dtype=np.int32), np.array(dev, dtype=np.int32), np.array(dev_label, dtype=np.int32)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " ( ", string) 
    string = re.sub(r"\)", " ) ", string) 
    string = re.sub(r"\?", " ? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

DATA_DIR = os.path.join("data", "sentiment_datasets")
DATA_NAMES = {
    "SST1" : ["stsa.fine.phrases.train", "stsa.fine.dev", "stsa.fine.test"],
    "SST2" : ["stsa.binary.train", "stsa.binary.dev", "stsa.binary.test"],
    "TREC"  : ["TREC.train.all", "", "TREC.test.all"],
    "CR"    : ["custrev.all", "", ""],
    "MPQA"  : ["mpqa.all", "", ""],
    "MR"    : ["rt-polarity.all", "", ""],
    "Subj"  : ["subj.all", "", ""],
    "IMDB"  : ["imdb.train.all", "", "imdb.test.all"]
}

def build_dataset(dataset, use_w2v=True, padding=0):
    # if already have built this dataset, load data.
    if use_w2v and os.path.exists(os.path.join(DATA_DIR, "{}.pkl".format(dataset))):
        print("{}.pkl exists! loading from pkl..".format(dataset))
        with open(os.path.join(DATA_DIR, "{}.pkl".format(dataset)), "rb") as f:
            w2v, train, train_label, test, test_label, dev, dev_label, word_to_idx = \
                pickle.load(f)
        return w2v, train, train_label, test, test_label, dev, dev_label, word_to_idx

    # else, build dataset from raw data.    
    train_path, dev_path, test_path = [os.path.join(DATA_DIR, name) if name != "" else "" for name in DATA_NAMES[dataset]]

    print("loading data..")
    word_to_idx, train, train_label, test, test_label, dev, dev_label = \
        load_data(dataset, train_path, test_name=test_path, dev_name=dev_path, padding=padding)

    V = len(word_to_idx) + 1
    print("Vocab size: {}".format(V))
    print("train size: {}".format(train.shape))

    if use_w2v:
        print("loading word2vec..")
        w2v = load_bin_vec(os.path.join("data", "GoogleNews-vectors-negative300.bin"), word_to_idx)

        # Not all words in word_to_idx are in w2v.
        # Word embeddings initialized to random Unif(-0.25, 0.25)
        print("building embedding matrix..")
        embed = np.random.uniform(-0.25, 0.25, (V, len(list(w2v.values())[0])))
        embed[0] = 0  # padding
        for word, vec in w2v.items():
            embed[word_to_idx[word]] = vec

        print("saving..")
        with open(os.path.join(DATA_DIR, "{}.pkl".format(dataset)), "wb") as f:
            pickle.dump([np.array(embed),
                        train, train_label,
                        test, test_label,
                        dev, dev_label,
                        word_to_idx], f)

    if use_w2v:
        return [np.array(embed), train, train_label, test, 
                test_label, dev, dev_label, word_to_idx]
    else:
        return [train, train_label, test, test_label, dev, dev_label, word_to_idx]

def train_test_dev_split(train, test, dev, train_label, test_label, dev_label):
    cnt = 0
    for data in [train, test, dev]:
        if len(data) != 0:
            cnt += 1
    
    if cnt == 0:
        raise ValueError("not a proper train,dev,test input")
        
    elif cnt == 1:  # only train set is provided.
        train_set = list(zip(train, train_label))
        random.shuffle(train_set)
        idx1 = int(len(train_set) * 0.8)
        idx2 = int(len(train_set) * 0.9)
        train_set, test_set, dev_set = train_set[:idx1], train_set[idx1:idx2], train_set[idx2:]
        train, train_label = list(zip(*train_set))
        test, test_label = list(zip(*test_set))
        dev, dev_label = list(zip(*dev_set))

    elif cnt == 2:  # train/test sets are provided.
        train_set = list(zip(train, train_label))
        random.shuffle(train_set)
        idx1 = int(len(train_set) * 0.9)
        train_set, dev_set = train_set[:idx1], train_set[idx1:]
        train, train_label = list(zip(*train_set))
        dev, dev_label = list(zip(*dev_set))

    elif cnt == 3:  # train/test/dev sets are provided.
        pass
    
    else:
        raise ValueError("Is it possible to reach here??")
        
    return np.array(train), np.array(test), np.array(dev), \
           np.array(train_label), np.array(test_label), np.array(dev_label)