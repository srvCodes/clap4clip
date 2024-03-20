import os
import pickle
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class cifar10(Dataset):

    filename = "cifar-10-python.tar.gz"

    train_list=[

    ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
    ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
    ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
    ['data_batch_4', '634d18415352ddfa80567beed471001a'],
    ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list=[

    ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    
    meta ={
    'filename':'batches.meta',
    'key': 'label_names',
    'md5':'5ff9c542aee3614f3951f8cda6e48888',
    }

    templates=[
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo-of a {}.',
    'a high contrast photb of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',]

    def __init__(self,root, transform=None, train=True):
        self.root = root
        self.train = train
        self.transform = transform
        self.base_folder = 'cifar-10-batches-py'
        if self.train:
            downloaded_list = self.train_list 
        else:
            downloaded_list = self.test_list

        self.data =[]
        self.targets =[]
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1)) # convert to HwC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def prompts(self,mode='single'):
        if mode == 'single':
            prompts = [[self.templates[0].format(label)] for label in self.classes]
            return prompts
        elif mode == 'ensemble':
            prompts = [[template.format(label) for template in self.templates] for label in self.classes]
            return prompts

    def get_labels(self):
        return np.array(self.targets)

    def get_classes(self):
        return self.classes


class cifar100(Dataset):
    # base_folder = 'cifar-100-python'
    train_list=[
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list=[
        ['test', 'foef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta={
        'filename':'meta',
        'key': 'fine_label_names',
        'md5':'7973b15100ade9c7d40fb424638fde48'
    }

    templates=[
        'a photo of a {}.',
        'a blurry photo of a {}.',
        'a black and white photo of a {}.',
        'a low contrast photo of a {}.',
        'a high contrast photo of a {}.',
        'a bad photo of a {}.',
        'a good photo of a {}.',
        'a photo of a small {}.',
        'a photo of a big {}.',
        'a photo of the {}.',
        'a blurry photo of the {}.',
        'a black and white photo of the {}.',
        'a low contrast photo of the {}.',
        'a high contrast photo of the {}.',
        'a bad photo of the {}.',
        'a good photo of the {}.',
        'a photo of the small {}.',
        'a photo of the big {}.',
    ]


    def __init__(self, root, transform=None, train=True):
        self.root = root
        self.train = train
        self.transform = transform
        self.base_folder = 'cifar-100-python'

        if self.train:
            downloaded_list = self.train_list 
        else:
            downloaded_list = self.test_list

        self.data =[]
        self.targets =[]
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))# convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta[ 'key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img,target,int(index)

    def __len__(self):
        return len(self.data)

    def prompts(self, mode='single'):
        if mode =='single':
            prompts = [[self.templates[0].format(label)] for label in self.classes]
            return prompts
        elif mode == 'ensemble':
            prompts = [[template.format(label) for template in self.templates] for label in self.classes]
            return prompts

    def get_labels(self):
        return np.array(self.targets)

    def get_classes(self):
        return self.classes
