import numpy as np
import torch
from tqdm import tqdm


def initialize(x, num_clusters):
    c0_idx = int(np.random.uniform(0, len(x)))
    centroid = x[c0_idx].reshape(1, -1)  # 选择第一个簇中心
    k = 1
    n = x.shape[0]
    while k < num_clusters:
        d2 = []
        for i in range(n):
            subs = centroid - x[i, :]
            dimension2 = np.power(subs, 2)
            dimension_s = np.sum(dimension2, axis=1)  # sum of each row
            d2.append(np.min(dimension_s))

        # ---- 直接选择概率值最大的 ------
        # new_c_idx = np.argmax(d2)
        # ---- 依照概率分布进行选择 -----
        prob = np.array(d2) / np.sum(np.array(d2))
        new_c_idx = np.random.choice(n, p=prob)

        centroid = np.vstack([centroid, x[new_c_idx]])
        k += 1
    return centroid


def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans(X,
           num_clusters,
           init_state=None,
           distance='euclidean',
           tol=1e-4,
           max_iter=1000,
           device=torch.device('cpu')):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    # print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    if init_state is not None:
        initial_state = init_state
    else:
        initial_state = initialize(X, num_clusters)

    iteration = 0
    while True:
        dis = pairwise_distance_function(X, initial_state, device)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(
                choice_cluster == index).squeeze().to(device)
            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(torch.sum((initial_state - initial_state_pre)**2,
                                 dim=1)))

        # increment iteration
        iteration = iteration + 1

        if iteration >= max_iter:
            break

        if center_shift**2 < tol:
            break

    return choice_cluster, initial_state


def kmeans_predict(X,
                   cluster_centers,
                   distance='euclidean',
                   device=torch.device('cpu')):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers, device)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()


def pairwise_distance(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B)**2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis