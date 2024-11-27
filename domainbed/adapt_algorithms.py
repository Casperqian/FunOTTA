import copy
import itertools
import math
import multiprocessing as mp
import random
from typing import List

import numpy as np
import torch
import torchvision
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms

from domainbed.algorithms import Algorithm
from domainbed.lib.Aug import aug

ALGORITHMS = [
    'T3A',
    'TentFull',
    'TentNorm',
    'TentPreBN',
    'TentClf',
    'PseudoLabel',
    'PLClf',
    'SHOT',
    'SHOTIM',
    'TAST',
    'UniDG',
    'DeYO',
    'DeYOClf',
    'Ours',
]


# original codes from "https://github.com/matsuolab/T3A"
def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits


import torch
import torch.nn as nn
from typing import List


class EnsembleLearner(nn.Module):
    def __init__(self, indim, outdim, ensemble_size, init_mode):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.in_features = indim
        self.out_features = outdim
        self.init_mode = init_mode

        # Register parameters
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        self.alpha_be = nn.Parameter(torch.Tensor(self.ensemble_size, self.in_features))
        self.gamma_be = nn.Parameter(torch.Tensor(self.ensemble_size, self.out_features))
        self.ensemble_bias = nn.Parameter(torch.Tensor(self.ensemble_size, self.out_features))

        self.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, k, D1 = x.size() if x.dim() == 3 else (x.size(0), 1, x.size(1))
        x = x.unsqueeze(1) if x.dim() == 2 else x

        r_x = x.view(1, B, k, D1).expand(self.ensemble_size, B, k, D1)
        r_x = r_x.view(self.ensemble_size, B * k, D1) * self.alpha_be.view(self.ensemble_size, 1, D1)
        w_r_x = nn.functional.linear(r_x.view(-1, D1), self.weight, self.bias)
        s_w_r_x = w_r_x.view(self.ensemble_size, B * k, -1)

        s_w_r_x = s_w_r_x * self.gamma_be.view(self.ensemble_size, 1, -1)
        if self.ensemble_bias is not None:
            s_w_r_x += self.ensemble_bias.view(self.ensemble_size, 1, -1)

        s_w_r_x = s_w_r_x.view(self.ensemble_size, B, k, -1).view(-1, k, -1)

        return s_w_r_x.squeeze() if x.dim() == 2 else s_w_r_x

    def reset(self):
        self._initialize_tensor(self.weight)
        self._initialize_tensor(self.alpha_be)
        self._initialize_tensor(self.gamma_be)
        if self.ensemble_bias is not None:
            self._initialize_tensor(self.ensemble_bias)
        if self.bias is not None:
            self._initialize_tensor(self.bias)

    def _initialize_tensor(self, tensor: torch.Tensor):
        init_values = [0, 1]  # Default init values
        if self.init_mode in ['zeros', 'ones', 'uniform', 'normal', 'random_sign']:
            initialize_tensor(tensor, self.init_mode, init_values)
        else:
            initialize_tensor(tensor, self.init_mode)


def initialize_tensor(tensor: torch.Tensor, initializer: str, init_values: List[float] = []) -> None:
    if initializer == "zeros":
        nn.init.zeros_(tensor)
    elif initializer == "ones":
        nn.init.ones_(tensor)
    elif initializer == "uniform":
        nn.init.uniform_(tensor, init_values[0], init_values[1])
    elif initializer == "normal":
        nn.init.normal_(tensor, init_values[0], init_values[1])
    elif initializer == 'xavier_normal':
        nn.init.xavier_normal_(tensor)
    elif initializer == 'kaiming_normal':
        nn.init.kaiming_normal_(tensor)
    else:
        raise NotImplementedError(f"Unknown initializer: {initializer}")

class Ours(Algorithm):

    def __init__(self, input_shape, num_classes, num_domains, hparams,
                 algorithm):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # trained feature extractor and last linear classifier
        self.featurizer = algorithm.featurizer
        self.classifier = algorithm.classifier
        self.pre_classifier = copy.deepcopy(algorithm.classifier)
        for param in self.pre_classifier.parameters():
            param.requires_grad = False

        self.featurizer_state_dict = copy.deepcopy(
            algorithm.featurizer.state_dict())
        self.clf_state_dict = copy.deepcopy(algorithm.classifier.state_dict())

        # store supports and corresponding labels
        warmup_supports = self.classifier.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_labels = F.one_hot(warmup_prob.argmax(1),
                                       num_classes=num_classes).float()
        self.warmup_ent = softmax_entropy(warmup_prob)

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        # hparams
        self.filter_K = hparams['filter_K']
        self.num_ensemble = hparams['num_ensemble']
        self.steps = hparams['gamma']
        self.lr = hparams['lr']
        self.init_mode = hparams['init_mode']
        self.num_classes = num_classes
        self.k = hparams['k']
        self._lambda1 = hparams['lambda1']
        self._lambda2 = hparams['lambda2']
        self.tau = 10

        # modules and its optimizer
        self.mlps = EnsembleLearner(self.featurizer.n_outputs,
                                  self.featurizer.n_outputs // 4,
                                  self.num_ensemble, self.init_mode).cuda()
        self.optimizer = torch.optim.Adam([{
            'params':
            self.classifier.parameters(),
            'lr':
            0.1 * self.lr
        }, {
            'params': self.mlps.parameters(),
            'lr': self.lr
        }])

    def forward(self, x, adapt=False):
        if not self.hparams['cached_loader']:
            z = self.featurizer(x)
        else:
            z = x

        if adapt:
            with torch.no_grad():
                p_supports = self.classifier(z)
                yhat = torch.nn.functional.one_hot(
                    p_supports.argmax(1),
                    num_classes=self.num_classes).float()
                ent = softmax_entropy(p_supports)

                # prediction
                self.supports = self.supports.to(z.device)
                self.labels = self.labels.to(z.device)
                self.supports = torch.cat([self.supports, z])
                self.labels = torch.cat([self.labels, yhat])

        supports, labels = self.select_supports()

        for _ in range(self.steps):
            p = self.forward_and_adapt(z, supports, labels)

        return p

    def select_supports(self):
        filter_K = self.filter_K
        num_classes = self.num_classes
        y_hat = self.labels.argmax(dim=1).long()
        device = self.supports.device

        if filter_K >= 0:
            c_labels, centroids = dynamic_filter(X=self.supports,
                                                     num_clusters=num_classes,
                                                     distance='euclidean',
                                                     device=device)
            dist = torch.cdist(self.supports, centroids, p=2)
            indices = []
            indices1 = torch.LongTensor(list(range(len(dist)))).to(device)
            for i in range(num_classes):
                subset = (c_labels == i).nonzero(as_tuple=True)[0]
                if subset.size(0) > 0:
                    dist_subset = dist[subset, i]
                    _, sorted_indices = torch.sort(dist_subset)
                    indices.append(indices1[subset][sorted_indices][:filter_K])
            indices = torch.cat(indices)
        else:
            indices = torch.LongTensor(list(range(len(
                self.supports)))).to(device)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]

        return self.supports, self.labels

    @torch.enable_grad()
    def forward_and_adapt(self, z, supports, labels):

        with torch.no_grad():
            targets, outputs = self.target_generation(z.squeeze(1), supports,
                                                      labels)

        self.optimizer.zero_grad()

        pre_logits = self.pre_classifier(z)
        classifier_logits = self.classifier(z)  
        ent_s = softmax_entropy(classifier_logits)
        coeff = 1 / (torch.exp(ent_s))
        coeff = (coeff / coeff.sum()) * len(ent_s)
        prototype_logits = self.compute_logits(z, supports, labels, self.mlps)  
        yhat = torch.argmax(prototype_logits, dim=-1) 
        
        embedding = self.mlps(z).view(self.num_ensemble, z.size(0), -1)  
        prototype = labels.T @ supports
        prototype_embedding = self.mlps(prototype).view(self.num_ensemble, self.num_classes, -1) 
        embedding = F.normalize(embedding, dim=-1)
        prototype_embedding = F.normalize(prototype_embedding, dim=-1)

        contrastive_loss, conventional_loss = None, None
        for ens in range(self.num_ensemble): 
            distances = F.pairwise_distance(embedding[ens].unsqueeze(1), prototype_embedding[ens].unsqueeze(0), p=2)  # [B, C]
            mask = torch.arange(self.num_classes, device=embedding.device).unsqueeze(0) == yhat[ens].unsqueeze(1)
            pos_distances = distances[mask].view(z.size(0), -1)  # [B, 1]
            neg_distances = distances[~mask].view(z.size(0), -1)
            contrastive_loss = (contrastive_loss or 0) + 
            (F.cross_entropy(torch.cat([pos_distances, neg_distances], dim=1).float(), 
                             torch.zeros(z.size(0), device=z.device).long(), reduction='none') * coeff).mean(0) / self.num_ensemble
            conventional_loss = (conventional_loss or 0) + 
            F.cross_entropy(prototype_logits[ens], targets[ens]) / self.num_ensemble
        
        # Compute the consistency loss, also update the classifier
        classifier_loss = F.kl_div(classifier_logits.log_softmax(-1), prototype_logits.softmax(-1), reduction='batchmean')
        consistency_loss = F.kl_div(classifier_logits.log_softmax(-1), pre_logits.softmax(-1), reduction='batchmean')

        loss = 0.2 * self._lambda1 * contrastive_loss + conventional_loss + self._lambda2 * (classifier_loss + consistency_loss)
        loss.backward()
        self.optimizer.step()
        # self.update_ema(self.classifier, self.pre_classifier)
 
        return outputs

    def target_generation(self, z, supports, labels):

        dist = self.cosine_distance_einsum(z, supports)
        W = torch.exp(-dist)  # [B, N]

        temp_k = int(min(labels.sum(0)))
        k = min(self.k, temp_k)

        values, indices = torch.topk(W, k, sorted=False)  # [B, k]
        topk_indices = torch.zeros_like(W).scatter_(
            1, indices, 1)  # [B, N] 1 for topk, 0 for else
        temp_labels = self.compute_logits(supports, supports, labels, self.mlps)  # [ens, N, C]
        temp_labels_targets = F.one_hot(
            temp_labels.argmax(-1),
            num_classes=self.num_classes).float()  # [ens, N, C]
        temp_labels_outputs = torch.softmax(temp_labels, -1)  # [ens, N, C]
        topk_indices = topk_indices.unsqueeze(0).repeat(self.num_ensemble, 1, 1)  # [ens, B, N]

        # targets for pseudo labels. we use one-hot class distribution
        targets = torch.bmm(topk_indices, temp_labels_targets)  # [ens, B, C]
        targets = targets / (targets.sum(2, keepdim=True) + 1e-12)  # [ens, B, C]

        # outputs for prediction
        outputs = torch.bmm(topk_indices, temp_labels_outputs)  # [ens, B, C]
        outputs = outputs / (outputs.sum(2, keepdim=True) + 1e-12)  # [ens, B, C]
        outputs = outputs.mean(0)  # [B,C]

        return targets, outputs

    def euclidean_distance_einsum(self, X, Y):
        # X, Y [n, dim], [m, dim] use_featurer_cache > [n,m]
        XX = torch.einsum('nd, nd->n', X, X)[:, None]  # [n, 1]
        YY = torch.einsum('md, md->m', Y, Y)  # [m]
        XY = 2 * torch.matmul(X, Y.T)  # [n,m]
        return XX + YY - XY

    def cosine_distance_einsum(self, X, Y):
        # X, Y [n, dim], [m, dim] -> [n,m]
        X = F.normalize(X, dim=1)
        Y = F.normalize(Y, dim=1)
        XX = torch.einsum('nd, nd->n', X, X)[:, None]  # [n, 1]
        YY = torch.einsum('md, md->m', Y, Y)  # [m]
        XY = 2 * torch.matmul(X, Y.T)  # [n,m]
        return XX + YY - XY

    def compute_logits(self, z, supports, labels, mlp):
        '''
        :param z: unlabeled test examples
        :param supports: support examples
        :param labels: labels of support examples
        :param mlp: multiple projection heads
        :return: classification logits of z
        '''
        B, dim = z.size()
        N, dim_ = supports.size()

        mlp_z = mlp(z)  # (B*ensemble_size,dim//4)
        mlp_supports = mlp(supports)  # (N*ensemble_size,dim_//4)
        assert (dim == dim_)

        logits = torch.zeros(self.num_ensemble, B,
                             self.num_classes).to(z.device)
        for ens in range(self.num_ensemble):

            temp_centroids = (labels /
                              (labels.sum(dim=0, keepdim=True) +
                               1e-12)).T @ mlp_supports[ens * N:(ens + 1) * N]

            # normalize
            temp_z = torch.nn.functional.normalize(mlp_z[ens * B:(ens + 1) *
                                                         B],
                                                   dim=1)
            temp_centroids = torch.nn.functional.normalize(temp_centroids,
                                                           dim=1)
            logits[ens] = self.tau * temp_z @ temp_centroids.T  # [B,C]

        return logits

    def update_ema(self, model, ema_model, decay=0.99):
        """Updates the EMA model parameters."""
        with torch.no_grad():
            for param, ema_param in zip(model.parameters(),
                                        ema_model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data
        self.featurizer.load_state_dict(self.featurizer_state_dict)
        self.classifier.load_state_dict(self.clf_state_dict)
        # self.pre_classifier.load_state_dict(self.clf_state_dict)
        self.mlps.reset()
        self.optimizer = torch.optim.Adam([{
            'params':
            self.classifier.parameters(),
            'lr':
            0.1 * self.lr
        }, {
            'params': self.mlps.parameters(),
            'lr': self.lr
        }])
        torch.cuda.empty_cache()


class T3A(Algorithm):
    """
    Test Time Template Adjustments (T3A)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams,
                 algorithm):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = algorithm.featurizer
        self.classifier = algorithm.classifier

        warmup_supports = self.classifier.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(
            warmup_prob.argmax(1), num_classes=num_classes).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = hparams['filter_K']
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x, adapt=False):
        if not self.hparams['cached_loader']:
            z = self.featurizer(x)
        else:
            z = x
        if adapt:
            # online adaptation
            p = self.classifier(z)
            yhat = torch.nn.functional.one_hot(
                p.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(p)

            # prediction
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])

        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s)))).to(
                y_hat.device)

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).to(y_hat.device)
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat == i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]

        return self.supports, self.labels

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data
        

class TAST(Algorithm):

    def __init__(self, input_shape, num_classes, num_domains, hparams,
                 algorithm):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # trained feature extractor and last linear classifier
        self.featurizer = algorithm.featurizer
        self.classifier = algorithm.classifier

        # store supports and corresponding labels
        warmup_supports = self.classifier.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_labels = F.one_hot(warmup_prob.argmax(1),
                                       num_classes=num_classes)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.ent = self.warmup_ent.data
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data

        # hparams
        self.filter_K = hparams['filter_K']
        self.steps = hparams['gamma']
        self.num_ensemble = hparams['num_ensemble']
        self.lr = hparams['lr']
        self.tau = hparams['tau']
        self.init_mode = hparams['init_mode']
        self.num_classes = num_classes
        self.k = hparams['k']

        # multiple projection heads and its optimizer
        self.mlps = EnsembleLearner(self.featurizer.n_outputs,
                                  self.featurizer.n_outputs // 4,
                                  self.num_ensemble, self.init_mode).cuda()
        self.optimizer = torch.optim.Adam(self.mlps.parameters(),
                                          lr=self.lr,
                                          eps=1e-8)

    def forward(self, x, adapt=False):
        if not self.hparams['cached_loader']:
            z = self.featurizer(x)
        else:
            z = x

        if adapt:
            p_supports = self.classifier(z)
            yhat = torch.nn.functional.one_hot(
                p_supports.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(p_supports)

            # prediction
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])

        supports, labels = self.select_supports()

        for _ in range(self.steps):
            p = self.forward_and_adapt(z, supports, labels)
        return p

    def compute_logits(self, z, supports, labels, mlp):
        '''
        :param z: unlabeled test examples
        :param supports: support examples
        :param labels: labels of support examples
        :param mlp: multiple projection heads
        :return: classification logits of z
        '''
        B, dim = z.size()
        N, dim_ = supports.size()

        mlp_z = mlp(z)  # (B*ensemble_size,dim//4)
        mlp_supports = mlp(supports)  # (N*ensemble_size,dim_//4)
        assert (dim == dim_)

        logits = torch.zeros(self.num_ensemble, B,
                             self.num_classes).to(z.device)
        for ens in range(self.num_ensemble):

            temp_centroids = (labels /
                              (labels.sum(dim=0, keepdim=True) +
                               1e-12)).T @ mlp_supports[ens * N:(ens + 1) * N]

            # normalize
            temp_z = torch.nn.functional.normalize(mlp_z[ens * B:(ens + 1) *
                                                         B],
                                                   dim=1)
            temp_centroids = torch.nn.functional.normalize(temp_centroids,
                                                           dim=1)
            logits[ens] = self.tau * temp_z @ temp_centroids.T  # [B,C]

        return logits

    def select_supports(self):
        '''
        we filter support examples with high prediction entropy
        :return: filtered support examples.
        '''
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        device = ent_s.device
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s)))).to(device)
        else:
            indices = []
            indices1 = torch.LongTensor(list(range(len(ent_s)))).to(device)
            for i in range(self.num_classes):
                _, indices2 = torch.sort(ent_s[y_hat == i])
                indices.append(indices1[y_hat == i][indices2][:filter_K])
            indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]

        return self.supports, self.labels

    @torch.enable_grad()
    def forward_and_adapt(self, z, supports, labels):
        # targets : pseudo labels, outputs: for prediction

        with torch.no_grad():
            targets, outputs = self.target_generation(z, supports, labels)

        self.optimizer.zero_grad()

        loss = None
        logits = self.compute_logits(z, supports, labels, self.mlps)

        for ens in range(self.num_ensemble):
            if loss is None:
                loss = F.kl_div(logits[ens].log_softmax(-1),
                                targets[ens],
                                reduction='batchmean')
            else:
                loss += F.kl_div(logits[ens].log_softmax(-1),
                                 targets[ens],
                                 reduction='batchmean')

        loss.backward()
        self.optimizer.step()

        return outputs  # outputs

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.mlps.reset()
        self.optimizer = torch.optim.Adam(self.mlps.parameters(), lr=self.lr)

        torch.cuda.empty_cache()

    # from https://jejjohnson.github.io/research_journal/snippets/numpy/euclidean/
    def euclidean_distance_einsum(self, X, Y):
        # X, Y [n, dim], [m, dim] use_featurer_cache > [n,m]
        XX = torch.einsum('nd, nd->n', X, X)[:, None]  # [n, 1]
        YY = torch.einsum('md, md->m', Y, Y)  # [m]
        XY = 2 * torch.matmul(X, Y.T)  # [n,m]
        return XX + YY - XY

    def cosine_distance_einsum(self, X, Y):
        # X, Y [n, dim], [m, dim] -> [n,m]
        X = F.normalize(X, dim=1)
        Y = F.normalize(Y, dim=1)
        XX = torch.einsum('nd, nd->n', X, X)[:, None]  # [n, 1]
        YY = torch.einsum('md, md->m', Y, Y)  # [m]
        XY = 2 * torch.matmul(X, Y.T)  # [n,m]
        return XX + YY - XY

    def target_generation(self, z, supports, labels):

        # retrieve k nearest neighbors. from "https://github.com/csyanbin/TPN/blob/master/train.py"
        dist = self.cosine_distance_einsum(z, supports)
        W = torch.exp(-dist)  # [B, N]

        temp_k = self.filter_K if self.filter_K != -1 else supports.size(
            0) // self.num_classes
        k = min(self.k, temp_k)

        values, indices = torch.topk(W, k, sorted=False)  # [B, k]
        topk_indices = torch.zeros_like(W).scatter_(
            1, indices, 1)  # [B, N] 1 for topk, 0 for else
        temp_labels = self.compute_logits(supports, supports, labels,
                                          self.mlps)  # [ens, N, C]
        temp_labels_targets = F.one_hot(
            temp_labels.argmax(-1),
            num_classes=self.num_classes).float()  # [ens, N, C]
        temp_labels_outputs = torch.softmax(temp_labels, -1)  # [ens, N, C]
        topk_indices = topk_indices.unsqueeze(0).repeat(
            self.num_ensemble, 1, 1)  # [ens, B, N]

        # targets for pseudo labels. we use one-hot class distribution
        targets = torch.bmm(topk_indices, temp_labels_targets)  # [ens, B, C]
        targets = targets / (targets.sum(2, keepdim=True) + 1e-12
                             )  # [ens, B, C]
        # targets = targets.mean(0)  # [B,C]

        # outputs for prediction

        outputs = torch.bmm(topk_indices, temp_labels_outputs)  # [ens, B, C]
        outputs = outputs / (outputs.sum(2, keepdim=True) + 1e-12
                             )  # [ens, B, C]
        outputs = outputs.mean(0)  # [B,C]

        return targets, outputs


class T3A(Algorithm):
    """
    Test Time Template Adjustments (T3A)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams,
                 algorithm):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = algorithm.featurizer
        self.classifier = algorithm.classifier

        warmup_supports = self.classifier.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(
            warmup_prob.argmax(1), num_classes=num_classes).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = hparams['filter_K']
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x, adapt=False):
        if not self.hparams['cached_loader']:
            z = self.featurizer(x)
        else:
            z = x
        if adapt:
            # online adaptation
            p = self.classifier(z)
            yhat = torch.nn.functional.one_hot(
                p.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(p)

            # prediction
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])

        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s)))).to(
                y_hat.device)

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).to(y_hat.device)
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat == i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]

        return self.supports, self.labels

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data


class TentFull(Algorithm):

    def __init__(self, input_shape, num_classes, num_domains, hparams,
                 algorithm):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.model, self.optimizer = self.configure_model_optimizer(
            algorithm, alpha=hparams['alpha'])
        self.steps = hparams['gamma']
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = False

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, adapt=False):
        if adapt:
            if self.episodic:
                self.reset()

            for _ in range(self.steps):
                if self.hparams['cached_loader']:
                    outputs = self.forward_and_adapt(x, self.model.classifier,
                                                     self.optimizer)
                else:
                    self.model.featurizer.eval()
                    outputs = self.forward_and_adapt(x, self.model,
                                                     self.optimizer)
                    self.model.featurizer.train()
        else:
            if self.hparams['cached_loader']:
                outputs = self.model.classifier(x)
            else:
                outputs = self.model(x)
        return outputs

    @torch.enable_grad(
    )  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        optimizer.zero_grad()
        outputs = model(x)
        # adapt
        loss = softmax_entropy(outputs).mean(0)
        loss.backward()
        optimizer.step()
        return outputs

    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        adapted_algorithm.featurizer = configure_model(
            adapted_algorithm.featurizer)
        params, param_names = collect_params(adapted_algorithm.featurizer)
        optimizer = torch.optim.Adam(
            params,
            lr=algorithm.hparams["lr"] * alpha,
            weight_decay=algorithm.hparams['weight_decay'])
        # adapted_algorithm.classifier.predict = lambda self, x: self(x)
        return adapted_algorithm, optimizer

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state,
                                 self.optimizer_state)


class TentNorm(TentFull):

    def forward(self, x, adapt=False):
        if self.hparams['cached_loader']:
            outputs = self.model.classifier(x)
        else:
            outputs = self.model(x)
        return outputs


class TentPreBN(TentFull):

    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        adapted_algorithm.classifier = PreBN(
            adapted_algorithm.classifier,
            adapted_algorithm.featurizer.n_outputs)
        adapted_algorithm.network = torch.nn.Sequential(
            adapted_algorithm.featurizer, adapted_algorithm.classifier)
        optimizer = torch.optim.Adam(
            adapted_algorithm.classifier.bn.parameters(),
            lr=algorithm.hparams["lr"] * alpha,
            weight_decay=algorithm.hparams['weight_decay'])
        return adapted_algorithm, optimizer


class TentClf(TentFull):

    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        optimizer = torch.optim.Adam(
            adapted_algorithm.classifier.parameters(),
            lr=algorithm.hparams["lr"] * alpha,
            weight_decay=algorithm.hparams['weight_decay'])
        adapted_algorithm.classifier.predict = lambda self, x: self(x)
        return adapted_algorithm, optimizer


class PreBN(torch.nn.Module):

    def __init__(self, m, num_features, **kwargs):
        super().__init__()
        self.m = m
        self.bn = torch.nn.BatchNorm1d(num_features, **kwargs)
        self.bn.requires_grad_(True)
        self.bn.track_running_stats = False
        self.bn.running_mean = None
        self.bn.running_var = None

    def forward(self, x):
        x = self.bn(x)
        return self.m(x)

    def predict(self, x):
        return self(x)


class PseudoLabel(Algorithm):

    def __init__(self, input_shape, num_classes, num_domains, hparams,
                 algorithm):
        """
        Hparams
        -------
        alpha (float) : learning rate coefficient
        beta (float) : threshold
        gamma (int) : number of updates
        """
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.model, self.optimizer = self.configure_model_optimizer(
            algorithm, alpha=hparams['alpha'])
        self.beta = hparams['beta']
        self.steps = hparams['gamma']
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = False

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, adapt=False):
        if adapt:
            if self.episodic:
                self.reset()

            for _ in range(self.steps):
                if self.hparams['cached_loader']:
                    outputs = self.forward_and_adapt(x, self.model.classifier,
                                                     self.optimizer)
                else:
                    self.model.featurizer.eval()
                    outputs = self.forward_and_adapt(x, self.model,
                                                     self.optimizer)
                    self.model.featurizer.train()
        else:
            if self.hparams['cached_loader']:
                outputs = self.model.classifier(x)
            else:
                outputs = self.model(x)
        return outputs

    @torch.enable_grad(
    )  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        optimizer.zero_grad()
        outputs = model(x)
        # adapt
        py, y_prime = F.softmax(outputs, dim=-1).max(1)
        flag = py > self.beta

        loss = F.cross_entropy(outputs[flag], y_prime[flag])
        loss.backward()
        optimizer.step()
        return outputs

    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        optimizer = torch.optim.Adam(
            adapted_algorithm.parameters(),
            lr=algorithm.hparams["lr"] * alpha,
            weight_decay=algorithm.hparams['weight_decay'])
        return adapted_algorithm, optimizer

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state,
                                 self.optimizer_state)


class PLClf(PseudoLabel):

    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        optimizer = torch.optim.Adam(
            adapted_algorithm.classifier.parameters(),
            lr=algorithm.hparams["lr"] * alpha,
            weight_decay=algorithm.hparams['weight_decay'])
        return adapted_algorithm, optimizer

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state,
                                 self.optimizer_state)


class SHOT(Algorithm):
    """
    "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation"
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams,
                 algorithm):
        """
        Hparams
        -------
        alpha (float) : learning rate coefficient
        beta (float) : threshold
        theta (float) : clf coefficient
        gamma (int) : number of updates
        """
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.model, self.optimizer = self.configure_model_optimizer(
            algorithm, alpha=hparams['alpha'])
        self.beta = hparams['beta']
        self.theta = hparams['theta']
        self.steps = hparams['gamma']
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = False

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, adapt=False):
        if adapt:
            if self.episodic:
                self.reset()

            for _ in range(self.steps):
                if self.hparams['cached_loader']:
                    outputs = self.forward_and_adapt(x, self.model.classifier,
                                                     self.optimizer)
                else:
                    self.model.featurizer.eval()
                    outputs = self.forward_and_adapt(x, self.model,
                                                     self.optimizer)
                    self.model.featurizer.train()
        else:
            if self.hparams['cached_loader']:
                outputs = self.model.classifier(x)
            else:
                outputs = self.model(x)
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        optimizer.zero_grad()
        outputs = model(x)

        loss = self.loss(outputs)
        loss.backward()
        optimizer.step()
        return outputs

    def loss(self, outputs):
        # (1) entropy
        ent_loss = softmax_entropy(outputs).mean(0)

        # (2) diversity
        softmax_out = F.softmax(outputs, dim=-1)
        msoftmax = softmax_out.mean(dim=0)
        ent_loss += torch.sum(msoftmax * torch.log(msoftmax + 1e-5))

        # (3) pseudo label
        # adapt
        py, y_prime = F.softmax(outputs, dim=-1).max(1)
        flag = py > self.beta
        clf_loss = F.cross_entropy(outputs[flag], y_prime[flag])

        loss = ent_loss + self.theta * clf_loss
        return loss

    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        optimizer = torch.optim.Adam(
            # adapted_algorithm.featurizer.parameters(),
            adapted_algorithm.classifier.parameters(),
            lr=algorithm.hparams["lr"] * alpha,
            weight_decay=algorithm.hparams['weight_decay'])
        return adapted_algorithm, optimizer

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state,
                                 self.optimizer_state)


class SHOTIM(SHOT):

    def loss(self, outputs):
        # (1) entropy
        ent_loss = softmax_entropy(outputs).mean(0)

        # (2) diversity
        softmax_out = F.softmax(outputs, dim=-1)
        msoftmax = softmax_out.mean(dim=0)
        ent_loss += torch.sum(msoftmax * torch.log(msoftmax + 1e-6))

        return ent_loss


class UniDG(Algorithm):

    def __init__(self, input_shape, num_classes, num_domains, hparams,
                 algorithm):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.model = copy.deepcopy(algorithm)
        self.model.train()
        self.lamb = hparams['lamb']
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams['lr'],
            weight_decay=algorithm.hparams['weight_decay'])
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

        self.pre_model = copy.deepcopy(algorithm)
        for param in self.pre_model.parameters():
            param.requires_grad = False

        warmup_supports = self.model.classifier.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.model.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(
            warmup_prob.argmax(1), num_classes=num_classes).float()
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data
        self.filter_K = hparams['filter_K']
        self.num_classes = num_classes
        self.sigma = 0.10
        self.softmax = torch.nn.Softmax(-1)

    @torch.enable_grad()
    def forward(self, x, adapt=False):
        if not self.hparams['cached_loader']:
            z = self.model.featurizer(x)
            with torch.no_grad():
                pre_z = self.pre_model.featurizer(x)
        else:
            z = x

        if adapt:
            # online adaptation
            self.optimizer.zero_grad()
            p = self.model.classifier(z)
            with torch.no_grad():
                pre_p = self.pre_model.classifier(pre_z)

            # Marginal Generalization
            loss_cons = torch.max(
                torch.pow(z - pre_z, 2) - self.sigma,
                torch.tensor(0.0)).mean()

            loss_ent = softmax_entropy(p).mean(0)
            loss = self.lamb * loss_cons + loss_ent
            loss.backward()
            self.optimizer.step()

            yhat = torch.nn.functional.one_hot(
                p.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(p)

            # Differentiable Memory Bank
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])

        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):

        # Memory Iteration
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        device = ent_s.device
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s)))).to(device)
        # Filter out Top-K
        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).to(device)
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat == i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]

        return self.supports, self.labels

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data
        load_model_and_optimizer(self.model, self.optimizer, self.model_state,
                                 self.optimizer_state)


class DeYO(Algorithm):

    def __init__(self, input_shape, num_classes, num_domains, hparams,
                 algorithm):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.lr = hparams["lr"]  
        self.steps = hparams["gamma"]
        self.deyo_margin = hparams["ent_thrshold"] 
        self.ent_margin = hparams["ent_margin"]  
        self.plpd_threshold = hparams["plpd_threshold"]  
        self.aug_type = hparams["aug_type"]  # ["occ", "patch", "pixel"]

        self.model, self.optimizer = self.configure_model_optimizer(algorithm)
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

        self.episodic = False
        self.reweight_ent, self.reweight_plpd = 1, 1
        self.occlusion_size = 112
        self.row_start = 56
        self.column_start = 56
        self.patch_len = 4

    def forward(self, x, adapt=False):
        if not self.hparams['cached_loader']:
            z = self.model.featurizer(x)
        else:
            z = x

        if adapt:
            if self.episodic:
                self.reset()

            for _ in range(self.steps):
                if self.hparams['cached_loader']:
                    outputs = self.forward_and_adapt(x, self.model.classifier,
                                                     self.optimizer)
                else:
                    self.model.featurizer.eval()
                    outputs = self.forward_and_adapt(x, self.model,
                                                     self.optimizer)
                    self.model.featurizer.train()
        else:
            if self.hparams['cached_loader']:
                outputs = self.model.classifier(x)
            else:
                outputs = self.model(x)
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model input data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        outputs = model(x)

        optimizer.zero_grad()
        entropys = softmax_entropy(outputs)
        if self.deyo_margin:
            filter_ids_1 = torch.where(
                (entropys < self.deyo_margin))  # hparams["ent_thrshold"] = 0.5
        else:
            filter_ids_1 = torch.where((entropys <= math.log(1000)))
        entropys = entropys[filter_ids_1]
        backward = len(entropys)
        if backward == 0:
            return outputs

        x_prime = x[filter_ids_1]
        x_prime = x_prime.detach()
        if self.aug_type == 'occ':
            first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1],
                                      -1).mean(dim=2)
            final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
            occlusion_window = final_mean.expand(-1, -1, self.occlusion_size,
                                                 self.occlusion_size)
            x_prime[:, :, self.row_start:self.row_start + self.occlusion_size,
                    self.column_start:self.column_start +
                    self.occlusion_size] = occlusion_window
        elif self.aug_type == 'patch':
            resize_t = torchvision.transforms.Resize(
                ((x.shape[-1] // self.patch_len) * self.patch_len,
                 (x.shape[-1] // self.patch_len) * self.patch_len))
            resize_o = torchvision.transforms.Resize(
                (x.shape[-1], x.shape[-1]))
            x_prime = resize_t(x_prime)
            x_prime = rearrange(x_prime,
                                'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w',
                                ps1=self.patch_len,
                                ps2=self.patch_len)
            perm_idx = torch.argsort(torch.rand(x_prime.shape[0],
                                                x_prime.shape[1]),
                                     dim=-1)
            x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),
                              perm_idx]
            x_prime = rearrange(x_prime,
                                'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)',
                                ps1=self.patch_len,
                                ps2=self.patch_len)
            x_prime = resize_o(x_prime)
        elif self.aug_type == 'pixel':
            x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
            x_prime = x_prime[:, :, torch.randperm(x_prime.shape[-1])]
            x_prime = rearrange(x_prime,
                                'b c (ps1 ps2) -> b c ps1 ps2',
                                ps1=x.shape[-1],
                                ps2=x.shape[-1])
        with torch.no_grad():
            outputs_prime = model(x_prime)

        prob_outputs = outputs[filter_ids_1].softmax(
            1)  # [len(filter_ids_1), num_classes]
        prob_outputs_prime = outputs_prime.softmax(
            1)  # [len(filter_ids_1), num_classes]

        cls1 = prob_outputs.argmax(dim=1)  # [len(filter_ids_1)]

        # Pseudo-Label Probability Difference
        plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(
            -1, 1)) - torch.gather(
                prob_outputs_prime, dim=1, index=cls1.reshape(-1, 1))
        plpd = plpd.reshape(-1)  # [len(filter_ids_1)]

        if self.plpd_threshold:
            filter_ids_2 = torch.where(plpd > self.plpd_threshold)
        else:
            filter_ids_2 = torch.where(plpd >= -2.0)
        entropys = entropys[filter_ids_2]
        final_backward = len(entropys)

        if final_backward == 0:
            del x_prime
            del plpd
            return outputs

        plpd = plpd[filter_ids_2]

        if self.reweight_ent or self.reweight_plpd:
            coeff = (self.reweight_ent * (1 / (torch.exp(
                ((entropys.clone().detach()) - self.ent_margin)))) +
                     self.reweight_plpd *
                     (1 / (torch.exp(-1. * plpd.clone().detach()))))
            entropys = entropys.mul(coeff)
        loss = entropys.mean(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        del x_prime
        del plpd
        return outputs

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def configure_model_optimizer(self, algorithm):
        adapted_algorithm = copy.deepcopy(algorithm)
        adapted_algorithm.featurizer = configure_model(
            adapted_algorithm.featurizer)
        params, param_names = collect_params(adapted_algorithm.featurizer)
        optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9)
        return adapted_algorithm, optimizer

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state,
                                 self.optimizer_state)


class DeYOClf(DeYO):

    def configure_model_optimizer(self, algorithm):
        adapted_algorithm = copy.deepcopy(algorithm)
        optimizer = torch.optim.SGD(adapted_algorithm.classifier.parameters(),
                                    lr=self.lr,
                                    momentum=0.9)
        adapted_algorithm.classifier.predict = lambda self, x: self(x)
        return adapted_algorithm, optimizer


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def configure_model(model):
    """Configure model for use with bn-based TTA."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
