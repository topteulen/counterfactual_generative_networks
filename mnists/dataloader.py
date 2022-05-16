import numpy as np
from PIL import Image, ImageColor
from pathlib import Path

import torch
import torch.nn.functional as F

from torch import tensor
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils import transform_to_one_color

import glob
from collections import OrderedDict

class ColoredMNIST(Dataset):
    def __init__(self, train=True, counterfactual=False, rotate=0, translate=None, scale=None, shear=None):
        # get the colored mnist
        self.data_path = 'mnists/data/colored_mnist/mnist_10color_double_testsets_jitter_var_0.02_0.025.npy'
        data_dic = np.load(self.data_path, encoding='latin1', allow_pickle=True).item()

        transform = [
            transforms.ToPILImage(),
            transforms.Resize((32, 32), Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ]

        if train:
            self.ims = data_dic['train_image']
            self.labels = tensor(data_dic['train_label'], dtype=torch.long)
        elif not counterfactual:
            self.ims = data_dic['test_image']
            self.labels = tensor(data_dic['test_label'], dtype=torch.long)
        else:
            self.ims = data_dic['counterfactual_image']
            self.labels = tensor(data_dic['counterfactual_label'], dtype=torch.long)
            transform += [
                transforms.RandomAffine(degrees=rotate, translate=translate, scale=scale, shear=shear),
                transform_to_one_color(),
            ]

        self.transform = transforms.Compose(transform)


    def __getitem__(self, idx):
        ims, labels = self.transform(self.ims[idx]), self.labels[idx]

        ret = {
            'ims': ims,
            'labels': labels,
        }

        return ret

    def __len__(self):
        return self.ims.shape[0]

class DoubleColoredMNIST(Dataset):

    def __init__(self, train=True):
        self.train = train
        self.mnist_sz = 32

        # get mnist
        mnist = datasets.MNIST('mnists/data', train=True, download=True)
        if train:
            ims, labels = mnist.data[:50000], mnist.targets[:50000]
        else:
            ims, labels = mnist.data[50000:], mnist.targets[50000:]

        self.ims_digit = torch.stack([ims, ims, ims], dim=1)
        self.labels = labels

        # colors generated by https://mokole.com/palette.html
        colors1 = [
            'darkgreen', 'darkblue', '#b03060',
            'orangered', 'yellow', 'burlywood', 'lime',
            'aqua', 'fuchsia', '#6495ed',
        ]
        # shift colors by X
        colors2 = [colors1[i-6] for i in range(len(colors1))]

        def get_rgb(x):
            t = torch.tensor(ImageColor.getcolor(x, "RGB"))/255.
            return t.view(-1, 1, 1)

        self.background_colors = list(map(get_rgb, colors1))
        self.object_colors = list(map(get_rgb, colors2))

        self.T = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        i = self.labels[idx] if self.train else np.random.randint(10)
        back_color = self.background_colors[i].clone()
        back_color += torch.normal(0, 0.01, (3, 1, 1))

        i = self.labels[idx] if self.train else np.random.randint(10)
        obj_color = self.object_colors[i].clone()
        obj_color += torch.normal(0, 0.01, (3, 1, 1))

        # get digit
        im_digit = (self.ims_digit[idx]/255.).to(torch.float32)
        im_digit = F.interpolate(im_digit[None,:], (self.mnist_sz, self.mnist_sz)).squeeze()
        im_digit = (im_digit > 0.1).to(int)  # binarize

        # plot digit onto the texture
        ims = im_digit*(obj_color) + (1 - im_digit)*back_color

        ret = {
            'ims': self.T(ims),
            'labels': self.labels[idx],
        }
        return ret

    def __len__(self):
        return self.labels.shape[0]

class WildlifeMNIST(Dataset):
    def __init__(self, train=True):
        self.train = train
        self.mnist_sz = 32
        inter_sz = 150

        # get mnist
        mnist = datasets.MNIST('mnists/data', train=True, download=True)
        if train:
            ims, labels = mnist.data[:50000], mnist.targets[:50000]
        else:
            ims, labels = mnist.data[50000:], mnist.targets[50000:]

        self.ims_digit = torch.stack([ims, ims, ims], dim=1)
        self.labels = labels

        # texture paths
        background_dir = Path('.') / 'mnists' / 'data' / 'textures' / 'background'
        self.background_textures = sorted([im for im in background_dir.glob('*.jpg')])
        object_dir = Path('.') / 'mnists' / 'data' / 'textures' / 'object'
        self.object_textures = sorted([im for im in object_dir.glob('*.jpg')])

        self.T_texture = transforms.Compose([
            transforms.Resize((inter_sz, inter_sz), Image.NEAREST),
            transforms.RandomCrop(self.mnist_sz, padding=3, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        # get textures
        i = self.labels[idx] if self.train else np.random.randint(10)
        back_text = Image.open(self.background_textures[i])
        back_text = self.T_texture(back_text)

        i = self.labels[idx] if self.train else np.random.randint(10)
        obj_text = Image.open(self.object_textures[i])
        obj_text = self.T_texture(obj_text)

        # get digit
        im_digit = (self.ims_digit[idx]/255.).to(torch.float32)
        im_digit = F.interpolate(im_digit[None, :], (self.mnist_sz, self.mnist_sz)).squeeze()
        im_digit = (im_digit > 0.1).to(int)  # binarize

        # plot digit onto the texture
        ims = im_digit*(obj_text) + (1 - im_digit)*back_text

        ret = {
            'ims': ims,
            'labels': self.labels[idx],
        }
        return ret

    def __len__(self):
        return self.labels.shape[0]

def get_dataloaders(dataset, batch_size, workers):
    if dataset == 'colored_MNIST':
        MNIST = ColoredMNIST
    elif dataset == 'double_colored_MNIST':
        MNIST = DoubleColoredMNIST
    elif dataset == 'wildlife_MNIST':
        MNIST = WildlifeMNIST
    else:
        raise TypeError(f"Unknown dataset: {dataset}")

    ds_train = MNIST(train=True)
    ds_test = {"test" :                               MNIST(train=False, counterfactual=False),
               "test_counterfactual":                 MNIST(train=False, counterfactual=True, rotate=0,   translate=None,       scale=None,       shear=None),
               "test_counterfactual_rot":             MNIST(train=False, counterfactual=True, rotate=180, translate=(0.1, 0.1), scale=None,       shear=None),
               "test_counterfactual_rot_scale":       MNIST(train=False, counterfactual=True, rotate=180, translate=(0.1, 0.1), scale=(0.5, 1.5), shear=None),
               "test_counterfactual_rot_scale_shear": MNIST(train=False, counterfactual=True, rotate=180, translate=(0.1, 0.1), scale=(0.5, 1.5), shear=30)}

    dl_train = DataLoader(ds_train, batch_size=batch_size,
                          shuffle=True, num_workers=workers)
    dl_test = {name: DataLoader(ds, batch_size=batch_size*2, shuffle=False, num_workers=workers) for name, ds in ds_test.items()}

    return dl_train, dl_test

TENSOR_DATASETS = ['colored_MNIST',
                   'double_colored_MNIST',
                   'wildlife_MNIST']
TENSOR_DATASETS = [d for dataset in TENSOR_DATASETS for d in [f'{dataset}_counterfactual',
                                                              f'{dataset}_counterfactual_rot',
                                                              f'{dataset}_counterfactual_rot_scale',
                                                              f'{dataset}_counterfactual_rot_scale_shear']]

def get_tensor_dataloaders(dataset, batch_size=64):
    assert dataset in TENSOR_DATASETS, f"Unknown datasets {dataset}"

    if 'counterfactual' in dataset:
        tensor = torch.load(f'mnists/data/{dataset}.pth')
        ds_train = TensorDataset(*tensor[:2])
        dataset = dataset.replace(dataset.split('MNIST')[1], '')
    else:
        ds_train = TensorDataset(*torch.load(f'mnists/data/{dataset}_train.pth'))

    # load test data
    ds_test = {}
    for name in glob.glob(f'mnists/data/{dataset}_test*.pth'):
        key = name.split("colored_MNIST_")[1].split(".")[0]
        ds_test[key] = TensorDataset(*torch.load(name))

    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=4,
                          shuffle=True, pin_memory=True)
    dl_test = {name: DataLoader(ds, batch_size=batch_size*10, num_workers=4, shuffle=False, pin_memory=True) for name, ds in ds_test.items()}
    dl_test = OrderedDict(sorted(dl_test.items()))

    return dl_train, dl_test
