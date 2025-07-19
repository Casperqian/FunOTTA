# The code is modified from domainbed.algorithms

import copy
import itertools
import math
import multiprocessing as mp
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import torch
import torchvision
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.cluster import KMeans
from torchvision import transforms

from domainbed.algorithms import Algorithm
from domainbed.kmeans_pytorch import kmeans
from domainbed.lib.augmenter import aug
from domainbed.lib.sam import SAM

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
    'MEMO',
    'MEMOBN',
    'UniDG',
    'DeYO',
    'DeYOClf',
    'SAR',
    'SARClf',
    'EATA',
    'EATAClf',
    'Ours',
]


te_transforms = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# original codes from "https://github.com/matsuolab/T3A"
def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
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
    
    
def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data
        
        
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


# motivated from "https://github.com/cs-giung/giung2/blob/c8560fd1b5/giung2/layers/linear.py"
class BatchEnsemble(nn.Module):

    def __init__(self, indim, outdim, ensemble_size, init_mode):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.in_features = indim
        self.out_features = outdim

        # register parameters
        self.register_parameter(
            "weight",
            nn.Parameter(torch.Tensor(self.out_features, self.in_features)))
        self.register_parameter("bias",
                                nn.Parameter(torch.Tensor(self.out_features)))

        self.register_parameter(
            "alpha_be",
            nn.Parameter(torch.Tensor(self.ensemble_size, self.in_features)))
        self.register_parameter(
            "gamma_be",
            nn.Parameter(torch.Tensor(self.ensemble_size, self.out_features)))

        use_ensemble_bias = True
        if use_ensemble_bias and self.bias is not None:
            delattr(self, "bias")
            self.register_parameter("bias", None)
            self.register_parameter(
                "ensemble_bias",
                nn.Parameter(
                    torch.Tensor(self.ensemble_size, self.out_features)))
        else:
            self.register_parameter("ensemble_bias", None)

        # initialize parameters
        self.init_mode = init_mode
        self.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = None
        if x.dim() == 2:
            B, D1 = x.size()
            k, dim = 1, 2
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            dim = 3
            B, k, D1 = x.size()
        else:
            raise ValueError(
                "Input must be either [batchsize, indim] or [batchsize, k, indim]"
            )

        r_x = x.view(1, B, k, D1).expand(self.ensemble_size, B, k,
                                         D1)  # [ensemble_size, B, k, D1]
        r_x = r_x.view(self.ensemble_size, B * k, D1)

        # Apply alpha
        r_x = r_x * self.alpha_be.view(self.ensemble_size, 1, D1)
        r_x = r_x.view(-1, D1)

        # Linear transformation
        w_r_x = nn.functional.linear(
            r_x, self.weight, self.bias)  # [ensemble_size * B * k, outdim]

        # Reshape the result back to [ensemble_size, B * k, outdim]
        _, D2 = w_r_x.size()
        s_w_r_x = w_r_x.view(self.ensemble_size, B * k, D2)

        # Apply gamma and ensemble bias
        s_w_r_x = s_w_r_x * self.gamma_be.view(self.ensemble_size, 1, D2)
        if self.ensemble_bias is not None:
            s_w_r_x = s_w_r_x + self.ensemble_bias.view(
                self.ensemble_size, 1, D2)

        s_w_r_x = s_w_r_x.view(self.ensemble_size, B, k, D2)
        s_w_r_x = s_w_r_x.view(-1, k, D2)

        if dim == 2:
            return s_w_r_x.squeeze()
        return s_w_r_x

    def reset(self):
        init_details = [0, 1]
        initialize_tensor(self.weight, self.init_mode, init_details)
        initialize_tensor(self.alpha_be, self.init_mode, init_details)
        initialize_tensor(self.gamma_be, self.init_mode, init_details)
        if self.ensemble_bias is not None:
            initialize_tensor(self.ensemble_bias, "zeros")
        if self.bias is not None:
            initialize_tensor(self.bias, "zeros")


def initialize_tensor(
    tensor: torch.Tensor,
    initializer: str,
    init_values: List[float] = [],
) -> None:
    if initializer == "zeros":
        nn.init.zeros_(tensor)

    elif initializer == "ones":
        nn.init.ones_(tensor)

    elif initializer == "uniform":
        nn.init.uniform_(tensor, init_values[0], init_values[1])

    elif initializer == "normal":
        nn.init.normal_(tensor, init_values[0], init_values[1])

    elif initializer == "random_sign":
        with torch.no_grad():
            tensor.data.copy_(
                2.0 * init_values[1] *
                torch.bernoulli(torch.zeros_like(tensor) + init_values[0]) -
                init_values[1])
    elif initializer == 'xavier_normal':
        torch.nn.init.xavier_normal_(tensor)

    elif initializer == 'kaiming_normal':
        torch.nn.init.kaiming_normal_(tensor)

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
        self.tau = hparams['tau']
        self._lambda1 = hparams['lambda1']
        self._lambda2 = hparams['lambda2']

        # modules and its optimizer
        self.mlps = BatchEnsemble(self.featurizer.n_outputs,
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
                self.ent = self.ent.to(z.device)
                self.supports = torch.cat([self.supports, z])
                self.labels = torch.cat([self.labels, yhat])
                self.ent = torch.cat([self.ent, ent])

        supports, labels = self.dynamic_filter()
        # supports, labels = self.select_supports()

        for _ in range(self.steps):
            p = self.forward_and_adapt(z, supports, labels)

        return p

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

    def dynamic_filter(self):
        filter_K = self.filter_K
        num_classes = self.num_classes
        y_hat = self.labels.argmax(dim=1).long()
        device = self.supports.device

        if filter_K >= 0:
            supports = self.supports.detach().cpu().numpy()
            # kmeans = KMeans(n_clusters=num_classes, random_state=0)
            # cluster_labels = kmeans.fit_predict(supports)
            # cluster_centers = kmeans.cluster_centers_
            # cluster_labels = torch.tensor(cluster_labels, device=device)
            # cluster_centers = torch.tensor(cluster_centers, device=device)
            cluster_labels, cluster_centers = kmeans(X=self.supports,
                                                     num_clusters=num_classes,
                                                     distance='euclidean',
                                                     device=device)
            dist = torch.cdist(self.supports, cluster_centers, p=2)
            indices = []
            indices1 = torch.LongTensor(list(range(len(dist)))).to(device)
            for i in range(num_classes):
                subset = (cluster_labels == i).nonzero(as_tuple=True)[0]
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
        self.ent = self.ent[indices]

        return self.supports, self.labels

    @torch.enable_grad()
    def forward_and_adapt(self, z, supports, labels):

        with torch.no_grad():
            targets, outputs = self.target_generation(z.squeeze(1), supports,
                                                      labels)

        self.optimizer.zero_grad()

        pre_logits = self.pre_classifier(z)
        classifier_logits = self.classifier(z)  # [B, num_classes]
        ent_s = softmax_entropy(classifier_logits)
        coeff = 1 / (torch.exp(ent_s))
        coeff = (coeff / coeff.sum()) * len(ent_s)
        prototype_logits = self.compute_logits(z, supports, labels,
                                               self.mlps)  # [ens, B, dim//4]

        embedding = self.mlps(z).view(self.num_ensemble, z.size(0),
                                      -1)  # [ens, B, dim//4]
        prototype = labels.T @ supports
        prototype_embedding = self.mlps(prototype).view(
            self.num_ensemble, self.num_classes, -1)  # [ens, C, dim//4]

        # yhat = torch.argmax(classifier_logits, dim=-1)  # [B]
        yhat = torch.argmax(prototype_logits, dim=-1)  # [ens, B]

        embedding = F.normalize(embedding, dim=-1)
        prototype_embedding = F.normalize(prototype_embedding, dim=-1)

        contrastive_loss = None
        conventional_loss = None
        for ens in range(self.num_ensemble):
            distances = F.pairwise_distance(
                embedding[ens].unsqueeze(1),
                prototype_embedding[ens].unsqueeze(0),
                p=2)  # [B, C]
            mask = torch.arange(
                self.num_classes,
                device=embedding.device).unsqueeze(0) == yhat[ens].unsqueeze(1)
            pos_distances = distances[mask].view(z.size(0), -1)  # [B, 1]
            neg_distances = distances[~mask].view(z.size(0), -1)

            contrastive_loss = (contrastive_loss or 0) + (F.cross_entropy(
                torch.cat([pos_distances, neg_distances], dim=1).float(),
                torch.zeros(z.size(0), device=z.device).long(),
                reduction='none') * coeff).mean(0) / self.num_ensemble

            conventional_loss = (conventional_loss or 0) + F.cross_entropy(
                prototype_logits[ens], targets[ens]) / self.num_ensemble

        # Compute the consistency loss, also unpdate the classifier
        classifier_loss = F.kl_div(classifier_logits.log_softmax(-1),
                                   prototype_logits.softmax(-1),
                                   reduction='batchmean')
        consistency_loss = F.kl_div(classifier_logits.log_softmax(-1),
                                    pre_logits.softmax(-1),
                                    reduction='batchmean')
        loss = 0.1 * self._lambda1 * contrastive_loss + conventional_loss + self._lambda2 * (
            classifier_loss + consistency_loss)
        loss.backward()
        self.optimizer.step()

        return outputs

    def target_generation(self, z, supports, labels):

        # retrieve k nearest neighbors. from "https://github.com/csyanbin/TPN/blob/master/train.py"
        dist = self.cosine_distance_einsum(z, supports)
        W = torch.exp(-dist)  # [B, N]

        temp_k = int(min(labels.sum(0)))
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
        targets = targets / (targets.sum(2, keepdim=True) + 1e-12)  # [ens, B, C]
        # targets = targets.mean(0)  # [B,C]

        # outputs for prediction
        outputs = torch.bmm(topk_indices, temp_labels_outputs)  # [ens, B, C]
        outputs = outputs / (outputs.sum(2, keepdim=True) + 1e-12)  # [ens, B, C]
        outputs = outputs.mean(0)  # [B,C]

        return targets, outputs

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
        self.pre_classifier.load_state_dict(self.clf_state_dict)
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
        self.mlps = BatchEnsemble(self.featurizer.n_outputs,
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


class TNN(Algorithm):
    """
    Non-Parametric Neighborhood Test-Time Generalization: Application to Medical Image Classification (TNN)
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
        self.k = hparams['k']
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
            prototype = labels @ supports.T

            p = self.cosine_distance_einsum(z, prototype)  # [B, C]
            # p = z @ self.prototype.T  # [B, C]
            indices = p.topk(self.k, dim=0, largest=True).indices
            new_prototypes = []
            for i in range(self.num_classes):
                mean_neighbor = z[indices[:, i], :].mean(dim=0)
                new_prototypes.append(mean_neighbor)

            classifier_weight = torch.stack(new_prototypes)  # [C, dim]

        return z @ classifier_weight.T

    def predict(self, x, adapt=False):
        return self(x, adapt)

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

    def reset(self):
        self.prototype = self.warmup_prototype.data


class EATA(Algorithm):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, algorithm):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.model, self.optimizer = self.configure_model_optimizer(
            algorithm, alpha=hparams['alpha'])
        self.steps = hparams['gamma']
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = False
        self.fisher_alpha = hparams['fisher_alpha'] # trade-off \beta for two losses (Eqn. 8), defualt: 1/2000.0
        self.e_margin = hparams['ent_margin'] # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = hparams['d_margin'] # hyper-parameter \epsilon for consine simlarity thresholding (Eqn. 5), default:0.05/0.4

        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples

        self.current_model_probs = None # the moving average of probability vector (Eqn. 4)

        self.fishers = None # to allow fully online adaptation
        # self.fishers = fishers # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, adapt=False):
        if adapt:
            if self.episodic:
                self.reset()

            for _ in range(self.steps):
                if self.hparams['cached_loader']:
                    outputs, num_counts_2, num_counts_1, updated_probs = self.forward_and_adapt(x, self.model.classifier, self.optimizer, fishers=self.fishers)
                else:
                    self.model.featurizer.eval()
                    outputs, num_counts_2, num_counts_1, updated_probs = self.forward_and_adapt(x, self.model, self.optimizer, fishers=self.fishers)
                    self.model.featurizer.train()
                self.num_samples_update_2 += num_counts_2
                self.num_samples_update_1 += num_counts_1
                self.reset_model_probs(updated_probs)
        else:
            if self.hparams['cached_loader']:
                outputs = self.model.classifier(x)
            else:
                outputs = self.model(x)
        return outputs
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer, fishers=None):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        Return: 
        1. model outputs; 
        2. the number of reliable and non-redundant samples; 
        3. the number of reliable samples;
        4. the moving average  probability vector over all previous samples
        """
        # forward
        outputs = model(x)
        # adapt
        entropys = softmax_entropy(outputs)
        # filter unreliable samples
        filter_ids_1 = torch.where(entropys < self.e_margin)
        ids1 = filter_ids_1
        ids2 = torch.where(ids1[0]>-0.1)
        entropys = entropys[filter_ids_1] 
        # filter redundant samples
        if self.current_model_probs is not None: 
            cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropys = entropys[filter_ids_2]
            ids2 = filter_ids_2
            updated_probs = self.update_model_probs(self.current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
        else:
            updated_probs = self.update_model_probs(self.current_model_probs, outputs[filter_ids_1].softmax(1))
        coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))
        # implementation version 1, compute loss, all samples backward (some unselected are masked)
        entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
        loss = entropys.mean(0)
        """
        # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
        # loss = 0
        # if x[ids1][ids2].size(0) != 0:
        #     loss = softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
        """
        if fishers is not None:
            ewc_loss = 0
            for name, param in model.named_parameters():
                if name in fishers:
                    ewc_loss += self.fisher_alpha * (fishers[name][0] * (param - fishers[name][1])**2).sum()
            loss += ewc_loss
        if x[ids1][ids2].size(0) != 0:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs
    
    def update_model_probs(self, current_model_probs, new_probs):
        if current_model_probs is None:
            if new_probs.size(0) == 0:
                return None
            else:
                with torch.no_grad():
                    return new_probs.mean(0)
        else:
            if new_probs.size(0) == 0:
                with torch.no_grad():
                    return current_model_probs
            else:
                with torch.no_grad():
                    return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs
    
    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        adapted_algorithm.featurizer = configure_model(
            adapted_algorithm.featurizer)
        params, param_names = collect_params(adapted_algorithm.featurizer)
        optimizer = torch.optim.SGD(params, momentum=0.9,
                                    lr=algorithm.hparams["lr"] * alpha,
                                    weight_decay=algorithm.hparams['weight_decay'])
        return adapted_algorithm, optimizer
        
        
class EATAClf(EATA):

    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        optimizer = torch.optim.SGD(adapted_algorithm.classifier.parameters(), 
                                    momentum=0.9,
                                    lr=algorithm.hparams["lr"] * alpha,
                                    weight_decay=algorithm.hparams['weight_decay'])
        adapted_algorithm.classifier.predict = lambda self, x: self(x)
        return adapted_algorithm, optimizer
    

class SAR(Algorithm):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, algorithm):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.model, self.optimizer = self.configure_model_optimizer(
            algorithm, alpha=hparams['alpha'])
        self.steps = hparams['gamma']
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = False
        self.margin_e0 = hparams['ent_margin']  # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.reset_constant_em = hparams['reset_constant_em']  # threshold e_m for model recovery scheme default:0.2
        self.ema = None
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, adapt=False):
        if adapt:
            if self.episodic:
                self.reset()

            for _ in range(self.steps):
                if self.hparams['cached_loader']:
                    outputs, ema, reset_flag = self.forward_and_adapt(x, self.model.classifier, self.optimizer)
                else:
                    self.model.featurizer.eval()
                    outputs, ema, reset_flag = self.forward_and_adapt(x, self.model, self.optimizer)
                    self.model.featurizer.train()
                if reset_flag:
                    self.reset()
                self.ema = ema
        else:
            if self.hparams['cached_loader']:
                outputs = self.model.classifier(x)
            else:
                outputs = self.model(x)
        return outputs
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model input data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        self.optimizer.zero_grad()
        # forward
        outputs = model(x)
        # adapt
        # filtering reliable samples/gradients for further adaptation; first time forward
        entropys = softmax_entropy(outputs)
        filter_ids_1 = torch.where(entropys < self.margin_e0)
        entropys = entropys[filter_ids_1]
        loss = entropys.mean(0)
        loss.backward()
        
        # 增加参数扰动
        self.optimizer.first_step(zero_grad=True) # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
        entropys2 = softmax_entropy(model(x))
        entropys2 = entropys2[filter_ids_1]  # second time forward  
        # loss_second_value = entropys2.clone().detach().mean(0)
        filter_ids_2 = torch.where(entropys2 < self.margin_e0)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
        loss_second = entropys2[filter_ids_2].mean(0)
        ema = self.ema
        if not np.isnan(loss_second.item()):
            ema = update_ema(ema, loss_second.item())  # record moving average loss values for model recovery

        # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
        loss_second.backward()
        self.optimizer.second_step(zero_grad=True)

        # perform model recovery
        reset_flag = False
        if ema is not None:
            if ema < self.reset_constant_em:
                # print(f"ema < {self.reset_constant_em}, now reset the model")
                reset_flag = True

        return outputs, ema, reset_flag

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.ema = None
        
    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        adapted_algorithm.featurizer = configure_model(
            adapted_algorithm.featurizer)
        params, param_names = collect_params(adapted_algorithm.featurizer)
        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, momentum=0.9, 
                        lr=algorithm.hparams["lr"] * alpha, 
                        weight_decay=algorithm.hparams['weight_decay'])
        return adapted_algorithm, optimizer
        
        
class SARClf(SAR):

    def configure_model_optimizer(self, algorithm, alpha):
        adapted_algorithm = copy.deepcopy(algorithm)
        base_optimizer = torch.optim.SGD
        optimizer = SAM(adapted_algorithm.classifier.parameters(), 
                        base_optimizer, momentum=0.9, 
                        lr=algorithm.hparams["lr"] * alpha, 
                        weight_decay=algorithm.hparams['weight_decay'])
        adapted_algorithm.classifier.predict = lambda self, x: self(x)
        return adapted_algorithm, optimizer
        
        
class MEMO(Algorithm):

    def __init__(self, input_shape, num_classes, num_domains, hparams,
                 algorithm):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.aug_batchsize = hparams['aug_batchsize']
        self.lr = hparams['lr']
        self.model, self.optimizer = self.configure_model_optimizer(algorithm)
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, adapt=False):
        if adapt:
            if self.hparams['cached_loader']:
                outputs = self.forward_and_adapt(x, self.model.classifier,
                                                 self.optimizer)
            else:
                outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        else:
            if self.hparams['cached_loader']:
                outputs = self.model.classifier(x)
            else:
                outputs = self.model(x)
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        batch_size = x.shape[0]
        logits_list = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            for i in range(batch_size):
                futures = [
                    executor.submit(aug, x[i])
                    for _ in range(self.aug_batchsize)
                ]
                inputs = [f.result() for f in futures]
                inputs = torch.stack(inputs).cuda()
                optimizer.zero_grad()
                logits = model(inputs)
                loss, logits = marginal_entropy(logits)
                loss.backward()
                optimizer.step()

                logits = self.test_single(model, x[i])
                logits_list.append(logits)
        outputs = torch.cat(logits_list, dim=0)
        return outputs

    def test_single(self, model, x):
        model.eval()
        inputs = te_transforms(x).unsqueeze(0)
        with torch.no_grad():
            outputs = model(inputs)
        return outputs

    def configure_model_optimizer(self, algorithm):
        algorithm = copy.deepcopy(algorithm)
        optimizer = torch.optim.Adam([{
            'params': algorithm.parameters(),
            'lr': self.lr
        }])
        return algorithm, optimizer

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state,
                                 self.optimizer_state)


class MEMOBN(MEMO):

    def configure_model_optimizer(self, algorithm):
        algorithm = copy.deepcopy(algorithm)
        params, _ = collect_params(algorithm)
        optimizer = torch.optim.Adam([{'params': params, 'lr': self.lr}])
        return algorithm, optimizer


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
        print('Learning Rate:', algorithm.hparams["lr"] * alpha)
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
        self.lr = hparams["lr"]  # [2.5e-4]
        self.steps = hparams["gamma"]
        self.deyo_margin = hparams["ent_thrshold"]  # [0.5]
        self.ent_margin = hparams["ent_margin"]  # [0.4]
        self.plpd_threshold = hparams["plpd_threshold"]  # [0.2, 0.3]
        self.aug_type = hparams["aug_type"]  # ["occ", "patch", "pixel"]

        # self.counts = [1e-6, 1e-6, 1e-6, 1e-6]
        # self.correct_counts = [0, 0, 0, 0]
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
                    outputs = self.forward_and_adapt(x, self.model.classifier, self.optimizer)
                else:
                    self.model.featurizer.eval()
                    outputs = self.forward_and_adapt(x, self.model, self.optimizer)
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
            first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
            final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
            occlusion_window = final_mean.expand(-1, -1, self.occlusion_size, self.occlusion_size)
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
            raise NotImplentError(
                "Must assign a plpd threshold for adaptation.")
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



