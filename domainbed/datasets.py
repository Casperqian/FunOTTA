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

    def __init__(self, images, labels, transform=None, return_path=False):
        self.image_paths = images
        self.labels = labels
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        image_path = self.image_paths[idx]
        if self.return_path:
            return image, label, image_path
        else:
            return image, label


class MyImageFolder(ImageFolder):

    def __init__(self,
                 root,
                 transform=None,
                 return_positive=False,
                 aug_transform=None):
        super().__init__(root, transform)
        self.return_positive = return_positive
        self.aug_transform = aug_transform

    def __getitem__(self, index):
        path, label = self.imgs[index]
        image = Image.open(path).convert('RGB')
        if self.return_positive and self.aug_transform:
            positive_pair = self.aug_transform(image)
        if self.transform:
            image = self.transform(image)
        if self.return_positive:
            return image, positive_pair, label
        return image, label


class Debug(MultipleDomainDataset):

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 5
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(torch.randn(100, *self.INPUT_SHAPE),
                              torch.randint(0, self.num_classes, (100, ))))


class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']


class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentDR(MultipleDomainDataset):
    N_STEPS = 10001
    CHECKPOINT_FREQ = 100
    N_WORKERS = 16
    ENVIRONMENTS = [
        'DR2015',
        'Messidor_2_resized',
        'APTOS',
        'IDRID_resized',
        'DDR_resized',
        'DeepDRiD_resized',
    ]

    def __init__(self,
                 root,
                 envs,
                 steps=N_STEPS,
                 environments=ENVIRONMENTS,
                 input_shape=(3, 224, 224),
                 num_classes=5,
                 denormalize=False,
                 return_path=False):
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
                    MyDataset(imgs,
                              labels,
                              transform=augment_transform,
                              return_path=return_path))
            else:
                if not denormalize:
                    self.datasets.append(
                        MyDataset(imgs,
                                  labels,
                                  transform=transform,
                                  return_path=return_path))
                else:
                    self.datasets.append(
                        MyDataset(imgs,
                                  labels,
                                  transform=denormalize_transform,
                                  return_path=return_path))

            print(f'{len(imgs)} images from {dataset_name} dataset.')


class Glaucoma(MultipleDomainDataset):
    N_STEPS = 1001
    CHECKPOINT_FREQ = 100
    N_WORKERS = 16
    ENVIRONMENTS = [
        'SOURCE', 'ORIGA', 'BEH', 'FIVES', 'sjchoi86-HRF', 'REFUGE1', 'PAPILA'
    ]

    def __init__(self,
                 root,
                 envs,
                 steps=N_STEPS,
                 environments=ENVIRONMENTS,
                 input_shape=(3, 224, 224),
                 num_classes=2,
                 denormalize=False,
                 return_path=False):
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
                    MyDataset(imgs,
                              labels,
                              transform=augment_transform,
                              return_path=return_path))
            else:
                if not denormalize:
                    self.datasets.append(
                        MyDataset(imgs,
                                  labels,
                                  transform=transform,
                                  return_path=return_path))
                else:
                    self.datasets.append(
                        MyDataset(imgs,
                                  labels,
                                  transform=denormalize_transform,
                                  return_path=return_path))

            print(f'{len(imgs)} images from {dataset_name} dataset.')


class Fetal8(MultipleDomainDataset):
    N_STEPS = 3001
    CHECKPOINT_FREQ = 100
    N_WORKERS = 16
    ENVIRONMENTS = ['Aloka', 'Voluson']

    def __init__(self,
                 root,
                 envs,
                 steps=N_STEPS,
                 environments=ENVIRONMENTS,
                 input_shape=(3, 224, 224),
                 num_classes=8,
                 denormalize=False,
                 return_path=False):
        super().__init__()
        self.dir = os.path.join(root, "Fetal8/")
        if self.dir is None:
            raise ValueError('Data directory not specified!')

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.steps = steps
        self.datasets = []

        denormalize_transform = transforms.Compose([
            transforms.Resize(input_shape[1:]),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.Resize(input_shape[1:]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        augment_transform = transforms.Compose([
            transforms.Resize(input_shape[1:]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
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
                    MyDataset(imgs,
                              labels,
                              transform=augment_transform,
                              return_path=return_path))
            else:
                if not denormalize:
                    self.datasets.append(
                        MyDataset(imgs,
                                  labels,
                                  transform=transform,
                                  return_path=return_path))
                else:
                    self.datasets.append(
                        MyDataset(imgs,
                                  labels,
                                  transform=denormalize_transform,
                                  return_path=return_path))

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


class PACS(MultipleEnvironmentImageFolder):
    N_STEPS = 1000
    CHECKPOINT_FREQ = 100
    ENVIRONMENTS = ["A", "C", "P", "S"]

    def __init__(self, root, envs, steps=N_STEPS, return_positive=False):
        self.dir = os.path.join(root, "PACS/")
        self.steps = steps
        super().__init__(self.dir, envs, return_positive)


class MultipleEnvironmentMNIST(MultipleDomainDataset):

    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat(
            (original_dataset_tr.data, original_dataset_te.data))

        original_labels = torch.cat(
            (original_dataset_tr.targets, original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(
                dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class RotatedMNIST(MultipleEnvironmentMNIST):
    N_STEPS = 5000
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, envs, steps=N_STEPS, denormalize=False):
        super().__init__(root, [0, 15, 30, 45, 60, 75], self.rotate_dataset, (
            1,
            28,
            28,
        ), 10)
        self.steps = steps

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(
                x, angle, fill=(0, ), interpolation=Image.BICUBIC)),
            transforms.ToTensor()
        ])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                           self.color_dataset, (
                                               2,
                                               28,
                                               28,
                                           ), 2)

        self.input_shape = (
            2,
            28,
            28,
        )
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(
            labels, self.torch_bernoulli_(environment, len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))),
               (1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()
