import os
import sys
import re
import random

import numpy as np
import torch

random.seed(0)

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
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
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def read_corpus(path, clean=True, encoding='utf8', shuffle=False, lower=True, get_noise=False):
    data = []
    labels = []
    if get_noise:
        noises = []
    print(path)
    with open(path, encoding=encoding) as fin:
        for line in fin:
            text, label = line.split('\t')[0], line.split('\t')[1]
            
            label = int(label)
            if clean:
                text = clean_str(text.strip("\n")) if clean else text.strip("\n")
            if lower:
                text = text.lower()
            labels.append(label)
            data.append(text.split())
            if get_noise:
                noise = int(line.split('\t')[2])
                noises.append(int(noise))

    if shuffle:
        perm = list(range(len(data)))
        random.shuffle(perm)
        data = [data[i] for i in perm]
        labels = [labels[i] for i in perm]
        if get_noise:
            noises = [noises[i] for i in perm]

    if get_noise:
        return data, labels, noises
    else:
        return data, labels

def cv_split(data, labels, nfold, test_id):
    assert (nfold > 1) and (test_id >= 0) and (test_id < nfold)
    lst_x = [ x for i, x in enumerate(data) if i%nfold != test_id ]
    lst_y = [ y for i, y in enumerate(labels) if i%nfold != test_id ]
    test_x = [ x for i, x in enumerate(data) if i%nfold == test_id ]
    test_y = [ y for i, y in enumerate(labels) if i%nfold == test_id ]
    perm = list(range(len(lst_x)))
    random.shuffle(perm)
    M = int(len(lst_x)*0.9)
    train_x = [ lst_x[i] for i in perm[:M] ]
    train_y = [ lst_y[i] for i in perm[:M] ]
    valid_x = [ lst_x[i] for i in perm[M:] ]
    valid_y = [ lst_y[i] for i in perm[M:] ]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def cv_split2(data, labels, nfold, valid_id):
    assert (nfold > 1) and (valid_id >= 0) and (valid_id < nfold)
    train_x = [ x for i, x in enumerate(data) if i%nfold != valid_id ]
    train_y = [ y for i, y in enumerate(labels) if i%nfold != valid_id ]
    valid_x = [ x for i, x in enumerate(data) if i%nfold == valid_id ]
    valid_y = [ y for i, y in enumerate(labels) if i%nfold == valid_id ]
    return train_x, train_y, valid_x, valid_y

def pad(sequences, pad_token='<pad>', pad_left=True):
    ''' input sequences is a list of text sequence [[str]]
        pad each text sequence to the length of the longest
    '''
    max_len = max(5,max(len(seq) for seq in sequences))
    if pad_left:
        return [ [pad_token]*(max_len-len(seq)) + seq for seq in sequences ]
    return [ seq + [pad_token]*(max_len-len(seq)) for seq in sequences ]


def create_one_batch(x, y, map2id, oov='<oov>'):
    oov_id = map2id[oov]
    x = pad(x)
    length = len(x[0])
    batch_size = len(x)
    x = [ map2id.get(w, oov_id) for seq in x for w in seq ]
    x = torch.LongTensor(x)
    assert x.size(0) == length*batch_size
    return x.view(batch_size, length).t().contiguous().cuda(), torch.LongTensor(y).cuda()

def create_one_batch_xyz(x, y, z, map2id, oov='<oov>'):
    oov_id = map2id[oov]
    x = pad(x)
    length = len(x[0])
    batch_size = len(x)
    x = [ map2id.get(w, oov_id) for seq in x for w in seq ]
    x = torch.LongTensor(x)
    assert x.size(0) == length*batch_size
    return x.view(batch_size, length).t().contiguous().cuda(), torch.LongTensor(y).cuda(), torch.LongTensor(z).cuda()

def create_one_batch_x(x, map2id, oov='<oov>'):
    oov_id = map2id[oov]
    x = pad(x)
    length = len(x[0])
    batch_size = len(x)
    x = [ map2id.get(w, oov_id) for seq in x for w in seq ]
    x = torch.LongTensor(x)
    assert x.size(0) == length*batch_size
    return x.view(batch_size, length).t().contiguous().cuda()


# shuffle training examples and create mini-batches
def create_batches(x, y, batch_size, map2id, perm=None, sort=False):

    lst = perm or range(len(x))

    # sort sequences based on their length; necessary for SST
    if sort:
        lst = sorted(lst, key=lambda i: len(x[i]))

    x = [ x[i] for i in lst ]
    y = [ y[i] for i in lst ]

    sum_len = 0.
    for ii in x:
        sum_len += len(ii)
    batches_x = [ ]
    batches_y = [ ]
    size = batch_size
    nbatch = (len(x)-1) // size + 1
    for i in range(nbatch):
        bx, by = create_one_batch(x[i*size:(i+1)*size], y[i*size:(i+1)*size], map2id)
        batches_x.append(bx)
        batches_y.append(by)

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_x = [ batches_x[i] for i in perm ]
        batches_y = [ batches_y[i] for i in perm ]

    sys.stdout.write("{} batches, avg sent len: {:.1f}\n".format(
        nbatch, sum_len/len(x)
    ))

    return batches_x, batches_y

# shuffle training examples and create mini-batches
def create_batches_xyz(x, y, z, batch_size, map2id, perm=None, sort=False):

    lst = perm or range(len(x))

    # sort sequences based on their length; necessary for SST
    if sort:
        lst = sorted(lst, key=lambda i: len(x[i]))

    x = [ x[i] for i in lst ]
    y = [ y[i] for i in lst ]
    z = [ z[i] for i in lst ]


    sum_len = 0.
    for ii in x:
        sum_len += len(ii)
    batches_x = [ ]
    batches_y = [ ]
    batches_z = [ ]
    size = batch_size
    nbatch = (len(x)-1) // size + 1
    for i in range(nbatch):
        bx, by, bz = create_one_batch_xyz(x[i*size:(i+1)*size], y[i*size:(i+1)*size], z[i*size:(i+1)*size], map2id)
        batches_x.append(bx)
        batches_y.append(by)
        batches_z.append(bz)

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_x = [ batches_x[i] for i in perm ]
        batches_y = [ batches_y[i] for i in perm ]
        batches_z = [ batches_z[i] for i in perm ]

    sys.stdout.write("{} batches, avg sent len: {:.1f}\n".format(
        nbatch, sum_len/len(x)
    ))

    return batches_x, batches_y, batches_z


# shuffle training examples and create mini-batches
def create_batches_x(x, batch_size, map2id, perm=None, sort=False):

    lst = perm or range(len(x))

    # sort sequences based on their length; necessary for SST
    if sort:
        lst = sorted(lst, key=lambda i: len(x[i]))

    x = [ x[i] for i in lst ]

    sum_len = 0.0
    batches_x = [ ]
    size = batch_size
    nbatch = (len(x)-1) // size + 1
    for i in range(nbatch):
        bx = create_one_batch_x(x[i*size:(i+1)*size], map2id)
        sum_len += len(bx)
        batches_x.append(bx)

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_x = [ batches_x[i] for i in perm ]

    # sys.stdout.write("{} batches, avg len: {:.1f}\n".format(
    #     nbatch, sum_len/nbatch
    # ))

    return batches_x


def load_embedding_npz(path):
    data = np.load(path)
    return [ w.decode('utf8') for w in data['words'] ], data['vals']

def load_embedding_txt(path):
    file_open = gzip.open if path.endswith(".gz") else open
    words = [ ]
    vals = [ ]
    with file_open(path, encoding='utf-8') as fin:
        fin.readline()
        for line in fin:
            line = line.rstrip()
            if line:
                parts = line.split(' ')
                words.append(parts[0])
                vals += [ float(x) for x in parts[1:] ]
    return words, np.asarray(vals).reshape(len(words),-1)

def load_embedding(path):
    if path.endswith(".npz"):
        return load_embedding_npz(path)
    else:
        return load_embedding_txt(path)
