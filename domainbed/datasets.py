# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import glob
import torch
import cv2
import random
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.datasets import MNIST, ImageFolder
from torch.utils.data import TensorDataset
from torchvision.transforms.functional import rotate
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from PIL import ImageOps, Image
from multiprocessing import Pool, cpu_count
from functools import partial
import logging

logger = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    "Debug",
    "TrainDataset",
    "TestDataset",
]


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class MyDataParallel(torch.nn.DataParallel):

    def __getattr__(self, name):
        return getattr(self.module, name)


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def rand_lvl(n):
    return np.random.uniform(low=0.1, high=n)


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


def load_image(path, transform=None):
    image = Image.open(path).convert('RGB')
    return transform(image)


class CropImageFromGray(object):

    def __init__(self, tol=7):
        self.tol = tol

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = torch.tensor(np.array(img), dtype=torch.float32)

        if img.ndim == 2:
            mask = img > self.tol
            img = img[np.ix_(mask.any(1), mask.any(0))]
        elif img.ndim == 3:
            gray_img = img.float().mean(dim=2)
            mask = gray_img > self.tol
            check_shape = img[:, :, 0][np.ix_(mask.any(1),
                                              mask.any(0))].shape[0]
            if check_shape == 0:
                return img
            else:
                img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
                img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
                img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
                img = torch.stack([img1, img2, img3], dim=-1)

        return Image.fromarray(np.uint8(img))


class AddWeightedBlur(object):

    def __init__(self, alpha=4, beta=-4, gamma=128, sigmaX=10):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigmaX = sigmaX

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        blurred = cv2.GaussianBlur(img, (0, 0), self.sigmaX)
        img = cv2.addWeighted(img, self.alpha, blurred, self.beta, self.gamma)
        return Image.fromarray(img)


class CircleCrop(object):

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        height, width, depth = img.shape
        x = int(width / 2)
        y = int(height / 2)
        r = np.amin((x, y))
        circle_img = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
        img = cv2.bitwise_and(img, img, mask=circle_img)
        return Image.fromarray(img)


class MultipleDomainDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 400  # Default, subclasses may override
    N_WORKERS = 1  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class MyDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        images,
        labels,
        transform=None,
    ):
        self.image_paths = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']


class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentDR(MultipleDomainDataset):
    N_STEPS = 1001
    CHECKPOINT_FREQ = 100
    N_WORKERS = 16
    ENVIRONMENTS = [
        'DR2015_subset',
        'Messidor_2_resized',
        'APTOS',
        'IDRID_resized',
        'DDR_resized',
        'DeepDRiD_resized',
        # 'FGADR_resized',
    ]

    def __init__(self,
                 root,
                 envs,
                 steps=N_STEPS,
                 environments=ENVIRONMENTS,
                 input_shape=(3, 224, 224),
                 num_classes=5,
                 denormalize=False):
        super().__init__()
        self.dir = os.path.join(root, "MultipleEnvironmentDR/")
        if self.dir is None:
            raise ValueError('Data directory not specified!')

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.steps = steps
        self.datasets = []

        denormalize_transform = transforms.Compose([
            CropImageFromGray(tol=7),
            transforms.Resize(input_shape[1:]),
            CircleCrop(),
            AddWeightedBlur(alpha=4, beta=-4, gamma=128, sigmaX=10),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            CropImageFromGray(tol=7),
            transforms.Resize(input_shape[1:]),
            CircleCrop(),
            AddWeightedBlur(alpha=4, beta=-4, gamma=128, sigmaX=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        augment_transform = transforms.Compose([
            CropImageFromGray(tol=7),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            CircleCrop(),
            AddWeightedBlur(alpha=4, beta=-4, gamma=128, sigmaX=10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-360, 360)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        for dataset_idx, dataset_name in enumerate(environments):
            # Directories for each class
            class_dirs = [
                os.path.join(self.dir, dataset_name, f'{i}/')
                for i in range(num_classes)
            ]

            # Load images and create labels
            imgs = []
            labels = []

            for i, class_dir in enumerate(class_dirs):
                imgs_path = glob.glob(os.path.join(class_dir, '*'))
                imgs += imgs_path
                labels.extend([i] * len(imgs_path))
            labels = torch.tensor(labels, dtype=torch.long)

            shuffle_indices = torch.randperm(len(imgs)).tolist()
            imgs = [imgs[i] for i in shuffle_indices]
            labels = labels[shuffle_indices]

            # Create dataset and append to datasets list
            if dataset_idx in envs:  # train envs
                # if dataset_idx not in envs:  # test envs
                self.datasets.append(
                    MyDataset(imgs, labels, transform=augment_transform))
            else:
                if not denormalize:
                    self.datasets.append(
                        MyDataset(imgs, labels, transform=transform))
                else:
                    self.datasets.append(
                        MyDataset(imgs,
                                  labels,
                                  transform=denormalize_transform))

            print(f'{len(imgs)} images from {dataset_name} dataset.')


class Glaucoma(MultipleDomainDataset):
    N_STEPS = 1001
    CHECKPOINT_FREQ = 100
    N_WORKERS = 16
    ENVIRONMENTS = [
        'SOURCE', 'ORIGA', 'BEH', 'FIVES', 'G1020', 'sjchoi86-HRF', 'REFUGE1',
        'PAPILA'
    ]

    def __init__(self,
                 root,
                 envs,
                 steps=N_STEPS,
                 environments=ENVIRONMENTS,
                 input_shape=(3, 224, 224),
                 num_classes=2,
                 denormalize=False):
        super().__init__()
        self.dir = os.path.join(root, "Glaucoma/")
        if self.dir is None:
            raise ValueError('Data directory not specified!')

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.steps = steps
        self.datasets = []

        denormalize_transform = transforms.Compose([
            CropImageFromGray(tol=7),
            transforms.Resize(input_shape[1:]),
            CircleCrop(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            CropImageFromGray(tol=7),
            transforms.Resize(input_shape[1:]),
            CircleCrop(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        augment_transform = transforms.Compose([
            CropImageFromGray(tol=7),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            CircleCrop(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-360, 360)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        for dataset_idx, dataset_name in enumerate(environments):
            # Directories for each class
            class_dirs = [
                os.path.join(self.dir, dataset_name, f'{i}/')
                for i in range(num_classes)
            ]

            # Load images and create labels
            imgs = []
            labels = []

            for i, class_dir in enumerate(class_dirs):
                imgs_path = glob.glob(os.path.join(class_dir, '*'))
                imgs += imgs_path
                labels.extend([i] * len(imgs_path))
            labels = torch.tensor(labels, dtype=torch.long)

            shuffle_indices = torch.randperm(len(imgs)).tolist()
            imgs = [imgs[i] for i in shuffle_indices]
            labels = labels[shuffle_indices]

            if dataset_idx in envs:  # train envs
                self.datasets.append(
                    MyDataset(imgs, labels, transform=augment_transform))
            else:
                if not denormalize:
                    self.datasets.append(
                        MyDataset(imgs, labels, transform=transform))
                else:
                    self.datasets.append(
                        MyDataset(imgs,
                                  labels,
                                  transform=denormalize_transform))

            print(f'{len(imgs)} images from {dataset_name} dataset.')


class MultipleEnvironmentImageFolder(MultipleDomainDataset):

    def __init__(self, root, envs, return_positive):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        pos_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if i in envs:  # train envs
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = MyImageFolder(path, transform, return_positive,
                                        pos_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (
            3,
            224,
            224,
        )
        self.num_classes = len(self.datasets[-1].classes)
