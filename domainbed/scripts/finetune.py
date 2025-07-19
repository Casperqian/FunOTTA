# The code is modified from domainbed.scripts.train

import argparse
from argparse import Namespace
import collections
import json
import os
import random
import sys
import time
import uuid
from itertools import chain
import itertools
import copy
from tqdm import tqdm

import math
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from scipy.optimize import linear_sum_assignment

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader, DataParallelPassthrough
from domainbed import model_selection
from domainbed.lib.query import Q
import itertools


def accuracy_auc_f1(network, loader, device, adapt=False, epoch=1):
    all_labels = []
    all_predictions = []
    all_probabilities = []

    network.eval()
    for e in range(epoch):
        is_last_epoch = (e + 1 == epoch)
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                if adapt is None:
                    p = network(x)
                else:
                    p = network(x, adapt)

                pred = p.argmax(dim=1)
                probas = torch.softmax(p, dim=1)

                if is_last_epoch:
                    all_labels.extend(y.cpu().detach().numpy())
                    all_predictions.extend(pred.cpu().detach().numpy())
                    all_probabilities.extend(probas.cpu().detach().numpy())

    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    num_classes = all_probabilities.shape[1] if len(
        all_probabilities.shape) > 1 else 1

    acc = accuracy_score(all_labels, all_predictions)

    if num_classes == 2:
        auc = roc_auc_score(all_labels, all_probabilities[:, 1])
        f1 = f1_score(all_labels, all_predictions)
    else:
        auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr')
        f1 = f1_score(all_labels, all_predictions, average='weighted')

    network.train()

    return acc, auc, f1

def finetune(network, loader, device, lr, freeze=False, epoch=5, verbose=True):
    all_labels = []
    all_predictions = []
    all_probabilities = []

    if freeze:
        print(f"Freeze the featurizer.")
        for name, param in network.featurizer.named_parameters():
            param.requires_grad = False
            
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=lr)

    for e in range(epoch):
        network.train()
        running_loss = 0.0

        # 添加 tqdm 进度条
        for x, y in tqdm(loader, desc=f"Training Epoch {e+1}", disable=not verbose):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = network(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)

        # if verbose:
        #     print(f"Loss: {avg_loss:.4f}")

        if e + 1 == epoch:
            network.eval()
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    outputs = network(x)
                    pred = outputs.argmax(dim=1)
                    probas = torch.softmax(outputs, dim=1)

                    all_labels.extend(y.cpu().numpy())
                    all_predictions.extend(pred.cpu().numpy())
                    all_probabilities.extend(probas.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    num_classes = all_probabilities.shape[1] if all_probabilities.ndim > 1 else 1
    acc = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='binary' if num_classes == 2 else 'weighted')
    if num_classes == 2:
        auc = roc_auc_score(all_labels, all_probabilities[:, 1])
    else:
        auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr')
    return acc, auc, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--ft_batch_size', type=int, default=32)
    parser.add_argument('--freeze_extractor', action='store_true')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--epoch', type=int, default=5)
    args_in = parser.parse_args()

    epochs_path = os.path.join(args_in.input_dir, 'results.jsonl')
    records = []
    with open(epochs_path, 'r') as f:
        for line in f:
            records.append(json.loads(line[:-1]))
    records = Q(records)
    r = records[0]
    args = Namespace(**r['args'])
    args.input_dir = args_in.input_dir     

    args.output_dir = args.input_dir

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm,
                                                   args.dataset)
    else:
        hparams = hparams_registry.random_hparams(
            args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))
        
    print('Args in:')
    for k, v in sorted(vars(args_in).items()):
        print('\t{}: {}'.format(k, v))
    print("============================")

    assert os.path.exists(os.path.join(args.output_dir, 'IID_best.pkl'))  # IID_best is produced by train.py

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args_in.data_dir is not None:
        args.data_dir = args_in.data_dir
        print(f"Changing data directory to {args_in.data_dir}.")

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir, args.train_envs, denormalize=False)
    else:
        raise NotImplementedError
    
    from sklearn.model_selection import StratifiedKFold
    from collections import defaultdict

    print("5-fold Finetuning results:")
    results_accum = defaultdict(list)

    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=0)
    
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                len(args.train_envs), -1, hparams)

    algorithm.to(device)
    if hasattr(algorithm, 'network'):
        algorithm.network = DataParallelPassthrough(algorithm.network)
    else:
        for m in algorithm.children():
            m = DataParallelPassthrough(m)

    # load trained model
    ckpt = torch.load(os.path.join(args.output_dir, 'IID_best.pkl'))
    algorithm_dict = ckpt['model_dict']

    # 遍历每个 test environment
    for env_i, env in enumerate(dataset):
        if env_i in args.train_envs:
            continue

        X = list(range(len(env)))
        y = [env[i][1] for i in X]  # 获取标签用于 stratify

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"\n=== Env {env_i} | Fold {fold_idx+1}/{k_folds} ===")

            env_finetune = Subset(env, train_idx)
            env_eval = Subset(env, test_idx)

            ft_loader = FastDataLoader(env_finetune, batch_size=args_in.ft_batch_size, num_workers=dataset.N_WORKERS)
            ev_loader = FastDataLoader(env_eval, batch_size=args_in.ft_batch_size, num_workers=dataset.N_WORKERS)

            if algorithm_dict is not None:
                algorithm.load_state_dict(algorithm_dict)

            # finetune
            acc, auc, f1 = finetune(
                algorithm,
                ft_loader,
                device,
                lr=1e-4,
                freeze=args_in.freeze_extractor,
                epoch=args_in.epoch
            )

            # evaluate
            acc_eval, auc_eval, f1_eval = accuracy_auc_f1(
                algorithm,
                ev_loader,
                device,
                adapt=None
            )
            results_accum[f'env{env_i}_acc'].append(acc_eval)
            results_accum[f'env{env_i}_auc'].append(auc_eval)
            results_accum[f'env{env_i}_f1'].append(f1_eval)

    # 最终取平均
    results = {
        key: round(np.mean(values), 4) for key, values in results_accum.items()
    }
    results_keys = sorted(results.keys())
    misc.print_row(results_keys, colwidth=12)
    misc.print_row([results[key] for key in results_keys], colwidth=12)
    
    results_path = os.path.join(args.output_dir, 'results_finetune.jsonl')
    record = {"args": vars(args_in), "results": results}
    with open(results_path, 'a') as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
