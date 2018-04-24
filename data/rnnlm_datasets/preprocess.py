from collections import defaultdict
import os
import numpy as np

def build_dataset(dataset, max_word_length, eos='+'):
    data_dir = os.path.join("data", "rnnlm_datasets", dataset)
    word_to_idx = {}
    char_to_idx = {}
    
    word_to_idx['|'] = 0
    char_to_idx[' '] = 0
    char_to_idx['{'] = 1
    char_to_idx['}'] = 2
    
    word_tokens = defaultdict(list)
    char_tokens = defaultdict(list)
    
    actual_max_word_length = 0

    for fname in ['train', 'valid', 'test']:
        with open(os.path.join(data_dir, "{}.txt".format(fname)), 'r') as f:
            for line in f:
                line = line.strip()
                line = line.replace('{', '').replace('}', '').replace('|', '')
                line = line.replace('<unk>', ' | ')
                if eos:
                    line = line.replace(eos, '')
            
                for word in line.split():
                    if len(word) > max_word_length - 2:
                        word = word[:max_word_length - 2]
                    
                    if word not in word_to_idx:
                        word_to_idx[word] = len(word_to_idx)
                    word_tokens[fname].append(word_to_idx[word])

                    for c in word:
                        if c not in char_to_idx:
                            char_to_idx[c] = len(char_to_idx)

                    char_array = [char_to_idx[c] for c in '{' + word + '}']
                    char_tokens[fname].append(char_array)

                    actual_max_word_length = max(actual_max_word_length, len(char_array))

                if eos:
                    if eos not in word_to_idx or eos not in char_to_idx:
                        word_to_idx[eos] = len(word_to_idx)
                        char_to_idx[eos] = len(char_to_idx)

                    word_tokens[fname].append(word_to_idx[eos])

                    char_array = [char_to_idx[c] for c in '{' + eos + '}']
                    char_tokens[fname].append(char_array)
                    
    assert actual_max_word_length <= max_word_length

    print()
    print('actual longest token length is:', actual_max_word_length)
    print('size of word vocabulary:', len(word_to_idx))
    print('size of char vocabulary:', len(char_to_idx))
    print('number of tokens in train:', len(word_tokens['train']))
    print('number of tokens in valid:', len(word_tokens['valid']))
    print('number of tokens in test:', len(word_tokens['test']))
    
    word_tensors = {}
    char_tensors = {}
    for fname in ('train', 'valid', 'test'):
        assert len(char_tokens[fname]) == len(word_tokens[fname])
        
        word_tensors[fname] = np.array(word_tokens[fname], dtype=np.int32)
        char_tensors[fname] = np.zeros([len(char_tokens[fname]), actual_max_word_length], dtype=np.int32)
        
        for i, char_array in enumerate(char_tokens[fname]):
            char_tensors[fname][i,:len(char_array)] = char_array
    
    return word_to_idx, char_to_idx, word_tensors, char_tensors, actual_max_word_length

def train_test_dev_split(word_tensors, char_tensors):
    train_word = word_tensors['train']
    valid_word = word_tensors['valid']
    test_word = word_tensors['test']
    train_char = char_tensors['train']
    valid_char = char_tensors['valid']
    test_char = char_tensors['test']
    return train_word, valid_word, test_word, train_char, valid_char, test_char