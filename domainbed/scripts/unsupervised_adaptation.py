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
from scipy.optimize import linear_sum_assignment

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader, DataParallelPassthrough
from domainbed import model_selection
from domainbed.lib.query import Q
from domainbed import adapt_algorithms
import itertools


class Dataset:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def generate_featurelized_loader(loader, network, classifier, batch_size=32):
    """
    The classifier adaptation does not need to repeat the heavy forward path, 
    We speeded up the experiments by converting the observations into representations. 
    """
    z_list = []
    z_pair_list = []
    y_list = []
    p_list = []
    network.eval()
    classifier.eval()
    print("Starting Caching Features!")
    for x, y in loader:
        x = x.to(device)
        z = network(x)
        p = classifier(z)

        z_list.append(z.detach().cpu())
        y_list.append(y.detach().cpu())
        p_list.append(p.detach().cpu())
    network.train()
    classifier.train()
    z = torch.cat(z_list)
    z_pair = torch.cat(z_pair_list) if len(z_pair_list) > 0 else None
    y = torch.cat(y_list)
    p = torch.cat(p_list)
    ent = softmax_entropy(p)
    py = p.argmax(1).float().cpu().detach()
    dataset1, dataset2 = Dataset(z, y), Dataset(z, py)
    loader1 = torch.utils.data.DataLoader(dataset1,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          drop_last=True)
    loader2 = torch.utils.data.DataLoader(dataset2,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          drop_last=True)
    return loader1, loader2, ent


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--adapt_algorithm', type=str, default="T3A")
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--evaluate',
                        action='store_true',
                        help='Evaluate base model')
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

    if '-' in args_in.adapt_algorithm:
        args.adapt_algorithm, test_batch_size = args_in.adapt_algorithm.split(
            '-')
        args.test_batch_size = int(test_batch_size)
    else:
        args.adapt_algorithm = args_in.adapt_algorithm
        args.test_batch_size = 32  # default

    args.output_dir = args.input_dir

    alg_name = args_in.adapt_algorithm

    if args.adapt_algorithm in [
            'T3A',
            'TentPreBN',
            'TentClf',
            'PLClf',
            'EATAClf',
            'SARClf',
            'TAST',
            'Ours',
    ]:
        use_featurer_cache = True
    else:
        use_featurer_cache = False
    if os.path.exists(os.path.join(args.output_dir,
                                   'done_{}'.format(alg_name))):
        print("{} has already excecuted".format(alg_name))

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    algorithm_dict = None
    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(
        os.path.join(args.output_dir, 'out_{}.txt'.format(alg_name)))
    sys.stderr = misc.Tee(
        os.path.join(args.output_dir, 'err_{}.txt'.format(alg_name)))

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

    assert os.path.exists(os.path.join(args.output_dir, 'done'))
    assert os.path.exists(os.path.join(
        args.output_dir, 'IID_best.pkl'))  # IID_best is produced by train.py

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
        denormalize = True if args.adapt_algorithm == 'MEMO' else False
        dataset = vars(datasets)[args.dataset](args.data_dir,
                                               args.train_envs,
                                               denormalize=denormalize)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    test_splits = []
    for env_i, env in enumerate(dataset):
        if env_i in args.train_envs:
            continue

        if hparams['class_balanced']:
            weights = misc.make_weights_for_balanced_classes(env)
        else:
            weights = None
        test_splits.append((env, weights))

    eval_loaders = [
        FastDataLoader(dataset=env,
                       batch_size=args.test_batch_size,
                       num_workers=dataset.N_WORKERS) for env, _ in test_splits
    ]
    print('Dataloader length', [len(dl) for dl in eval_loaders])
    eval_weights = [None for _, weights in test_splits]
    eval_loader_names = [
        'env{}_out'.format(i) for i in range(len(test_splits))
    ]  # first env is from the source test loader

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                len(args.train_envs), -1, hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
    if hasattr(algorithm, 'network'):
        algorithm.network = DataParallelPassthrough(algorithm.network)
    else:
        for m in algorithm.children():
            m = DataParallelPassthrough(m)

    # train_minibatches_iterator = zip(*train_loaders)
    # uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    # load trained model
    ckpt = torch.load(os.path.join(args.output_dir, 'IID_best.pkl'))
    algorithm_dict = ckpt['model_dict']
    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    # Evaluate base model
    if args_in.evaluate:
        print("Base model's results")
        results = {}
        evals = zip(eval_loader_names, eval_loaders, eval_weights)
        for name, loader, weights in evals:
            acc, auc, f1 = accuracy_auc_f1(algorithm,
                                           loader,
                                           device,
                                           adapt=None)
            results[name + '_acc'] = round(acc, 4)
            results[name + '_auc'] = round(auc, 4)
            results[name + '_f1'] = round(f1, 4)
        results_keys = sorted(results.keys())
        misc.print_row(results_keys, colwidth=12)
        misc.print_row([results[key] for key in results_keys], colwidth=12)

    print("\nAfter {}".format(alg_name))
    # Cache the inference results
    if use_featurer_cache:
        original_evals = zip(eval_loader_names, eval_loaders, eval_weights)
        loaders = []
        for name, loader, weights in original_evals:
            loader1, loader2, ent = generate_featurelized_loader(
                loader,
                network=algorithm.featurizer,
                classifier=algorithm.classifier,
                batch_size=32)
            loaders.append((name, loader1, weights))
    else:
        loaders = zip(eval_loader_names, eval_loaders, eval_weights)

    evals = []
    for name, loader, weights in loaders:
        evals.append((name, loader, weights))

    last_results_keys = None
    adapt_algorithm_class = adapt_algorithms.get_algorithm_class(
        args.adapt_algorithm)

    if args.adapt_algorithm in ['T3A']:
        adapt_hparams_dict = {
            'filter_K': [20, 50, 100, -1],
        }
    elif args.adapt_algorithm in [
            'TentFull', 'TentPreBN', 'TentClf', 'TentNorm'
    ]:
        adapt_hparams_dict = {'alpha': [0.1, 1.0, 10.0], 'gamma': [1, 3]}
    elif args.adapt_algorithm in ['PseudoLabel', 'PLClf']:
        adapt_hparams_dict = {
            'alpha': [0.1, 1.0, 10.0],
            'gamma': [1, 3],
            'beta': [0.9]
        }
    elif args.adapt_algorithm in ['SHOT', 'SHOTIM']:
        adapt_hparams_dict = {
            'alpha': [0.1, 1.0, 10.0],
            'gamma': [1, 3],
            'beta': [0.9],
            'theta': [0.1],
        }
    elif args.adapt_algorithm in ['MEMO', 'MEMO_BN']:
        adapt_hparams_dict = {
            'aug_batchsize': [2, 4, 8],
            'lr': [1e-4],
        }
    elif args.adapt_algorithm in ['TAST']:
        adapt_hparams_dict = {
            'num_ensemble': [5],
            'filter_K': [20, 50, 100, -1],
            'gamma': [1, 3],
            'lr': [1e-3],
            'tau': [10],
            'k': [2, 4, 8],
            'init_mode': ['kaiming_normal']
        }
    elif args.adapt_algorithm in ['UniDG']:
        adapt_hparams_dict = {
            'lr': [1e-3, 1e-4],
            'gamma': [1, 3],
            'lamb': [1.0, 0.1],
            'filter_K': [20, 50, 100, -1],
        }
    elif args.adapt_algorithm in ['DeYO', 'DeYOClf']:
        adapt_hparams_dict = {
            'lr': [2.5e-4, 1e-4],
            'gamma': [1, 3],
            'ent_thrshold': [0.5 * math.log(dataset.num_classes)],
            'ent_margin': [0.4 * math.log(dataset.num_classes)],
            'plpd_threshold': [0.2, 0.3],
            'aug_type': ["occ", "patch", "pixel"]
        }
    elif args.adapt_algorithm in ['SAR', 'SARClf']:
        adapt_hparams_dict = {
            'alpha': [0.1, 1.0, 10.0],
            'gamma': [1, 3],
            'ent_margin': [0.4 * math.log(dataset.num_classes)],
            'reset_constant_em': [0.1, 0.2]
        }
    elif args.adapt_algorithm in ['EATA', 'EATAClf']:
        adapt_hparams_dict = {
            'alpha': [0.1, 1.0, 10.0],
            'gamma': [1, 3],
            'fisher_alpha': [1],
            'ent_margin': [0.4 * math.log(dataset.num_classes)],
            'd_margin': [0.02, 0.05, 0.1, 0.2, 0.4]
        }
    elif args.adapt_algorithm in ['Ours']:
        adapt_hparams_dict = {
            'num_ensemble': [5],
            'filter_K': [50, 100, 200, -1],
            'gamma': [1, 3],
            'k': [2, 4, 8],
            'lambda1': [0, 1], 
            'lambda2': [0, 1], 
            'init_mode': ['kaiming_normal']
        }
    else:
        raise Exception("Not Implemented Error")

    product = [x for x in itertools.product(*adapt_hparams_dict.values())]
    adapt_hparams_list = [
        dict(zip(adapt_hparams_dict.keys(), r)) for r in product
    ]

    for adapt_hparams in adapt_hparams_list:
        adapt_hparams['cached_loader'] = use_featurer_cache
        adapted_algorithm = adapt_algorithm_class(dataset.input_shape,
                                                  dataset.num_classes,
                                                  len(args.train_envs),
                                                  adapt_hparams, algorithm)
        # adapted_algorithm = DataParallelPassthrough(adapted_algorithm)
        adapted_algorithm.to(device)

        results = adapt_hparams

        for key, val in checkpoint_vals.items():
            results[key] = np.mean(val)

        # Usual evaluation
        for name, loader, weights in evals:
            acc, auc, f1 = accuracy_auc_f1(adapted_algorithm,
                                           loader,
                                           device,
                                           adapt=True,
                                           epoch=args_in.epoch)
            results[name + '_acc'] = round(acc, 4)
            results[name + '_auc'] = round(auc, 4)
            results[name + '_f1'] = round(f1, 4)
            adapted_algorithm.reset()

        del adapt_hparams['cached_loader']
        results_keys = sorted(results.keys())

        if results_keys != last_results_keys:
            misc.print_row(results_keys, colwidth=12)
            last_results_keys = results_keys
        misc.print_row([results[key] for key in results_keys], colwidth=12)

        results.update({'hparams': hparams, 'args': vars(args)})
        # save file
        epochs_path = os.path.join(args.output_dir,
                                   'results_{}.jsonl'.format(alg_name))
        with open(epochs_path, 'a') as f:
            f.write(json.dumps(results, sort_keys=True) + "\n")

    # create done file
    with open(os.path.join(args.output_dir, 'done_{}'.format(alg_name)),
              'w') as f:
        f.write('done')
