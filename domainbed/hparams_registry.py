# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
from domainbed.lib import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""

        random_state = np.random.RandomState(misc.seed_hash(random_seed, name))
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.

    _hparam('data_augmentation', True, lambda r: True)
    _hparam('backbone', 'resnet50', lambda r: 'resnet50')
    _hparam('dropout', 0.0, lambda r: r.choice([0.0, 0.1, 0.5]))
    _hparam('class_balanced', False, lambda r: False)
    _hparam('nonlinear_classifier', False, lambda r: False)

    _hparam('lr', 1e-4, lambda r: 10**r.uniform(-4, -2))
    _hparam('weight_decay', 1e-4, lambda r: 10**r.uniform(-4, -2))
    _hparam('batch_size', 64, lambda r: int(2**r.uniform(3, 6)))

    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
