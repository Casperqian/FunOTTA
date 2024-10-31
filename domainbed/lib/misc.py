# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Things that don't belong anywhere else
"""

import hashlib
import json
import os
import sys
from shutil import copyfile
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter

# def make_weights_for_balanced_classes(dataset):
#     labels = dataset.get_labels()
#     counts = Counter(labels.tolist())
#     n_classes = len(counts)

#     weight_per_class = {y: 1 / (counts[y] * n_classes) for y in counts}

#     weights = torch.tensor([weight_per_class[int(y)] for y in labels])
#     return weights


def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


def print_separator():
    print("=" * 80)


def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.4f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    print(sep.join([format_val(x) for x in row]), end_)


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys

    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)

    def get_labels(self):
        labels = [
            self.underlying_dataset.get_labels()[key] for key in self.keys
        ]
        return torch.tensor(labels).long()


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert (n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def evaluate(network, loader, device):
    network.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            outputs = network(data)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    accuracy = accuracy_score(all_labels, all_predictions)
    num_classes = all_probabilities.shape[1] if len(
        all_probabilities.shape) > 1 else 1

    if num_classes == 2:
        auc = roc_auc_score(all_labels, all_probabilities[:, 1])
        f1 = f1_score(all_labels, all_predictions)
    else:
        auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
    network.train()
    return accuracy, auc, f1


def accuracy(network, loader, device, hybrid=False):
    network.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            outputs = network.predict(data)
            if not hybrid:
                _, predicted = torch.max(outputs.data, 1)
            else:
                _, predicted = torch.max(outputs[0].data, 1)
            predictions.extend(predicted.tolist())
            labels.extend(target.tolist())
    accuracy = accuracy_score(labels, predictions)
    network.train()
    return accuracy


class Tee:

    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()
