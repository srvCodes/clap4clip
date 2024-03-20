import os
import random
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from .cifar import cifar10, cifar100
from .imagenet import imagenet
from .mnist import mnist
import pdb

def get_transforms(n_px=224, transform_mode='origin'):

    if transform_mode == 'origin':
        transform = transforms.Compose([
            transforms.Resize(n_px, interpolation=BICUBIC),
            # transforms.Resize((n_px,n_px), interpolation=BICUBIC),
            transforms.CenterCrop(n_px),
            # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5))], p=0.3),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954,0.26130258, 0.27577711)),
            ])

    elif transform_mode == 'flip':
        transform = transforms.Compose([
            transforms.Resize(n_px, interpolation=BICUBIC),
            transforms.CenterCrop(n_px),
            # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5))], p=0.3), #modifed
            lambda image: image.convert("RGB"),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            #modifed
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954,0.26130258, 0.27577711)),
            ])
    return transform

class FewShotDatasetWrapper(Dataset):
    def __init__(self, db, select_labels):
        self.db=db
        self.select_labels = select_labels
    
    def __getitem__(self, index):
        db_idx = self.select_labels[index]
        return self.db.__getitem__(db_idx)

    def __len__(self):
        return len(self.select_labels)

    def prompts(self, mode='single'):
        return self.db.prompts(mode)

    def get_labels(self):
        return self.db.get_labels()

    def get_classes(self):
        return self.db.get_classes()

def get_split_index(labels, n_shot, n_val=0, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    all_label_list = np.unique(labels)
    train_idx_list =[]
    val_idx_list =[]
    # pdb.set_trace()
    for label in all_label_list:
        label_collection = np.where(labels == label)[0]
        random.shuffle(label_collection)
        selected_idx = label_collection[:n_shot+n_val]
        train_idx_list.extend(selected_idx[:n_shot])
        val_idx_list.extend(selected_idx[n_shot:])
    # pdb.set_trace()
    return train_idx_list, val_idx_list

def build_dataset(db_name, root, n_shot=-1, n_val=0, transform_mode='origin'):
    root = os.path.join(root, db_name)
    transform = get_transforms(transform_mode=transform_mode)
    test_transform = get_transforms(transform_mode='origin')

    db_func={
        'cifar10':cifar10,
        'cifar100':cifar100,
        'imagenet': imagenet,
        'mnist':mnist,
    }

    train_db = db_func[db_name](root, transform, train=True)
    test_db = db_func[db_name](root, test_transform, train=False)

    return train_db, test_db

def build_dataset_fs(db_name, root, n_shot=1, n_val=0, transform_mode='origin'):
    root = os.path.join(root,db_name)
    transform = get_transforms(transform_mode=transform_mode)
    test_transform = get_transforms(transform_mode='origin')

    db_func ={
        'cifar10':cifar10,
        'cifar100':cifar100,
        'imagenet': imagenet,
        'mnist':mnist,
        }

    train_db = db_func[db_name](root, transform, train=True)
    val_db = db_func[db_name](root, test_transform, train=True)
    test_db = db_func[db_name](root, test_transform, train=False)

    if n_shot >0:
        labels = train_db.get_labels()
        train_index, val_index = get_split_index(labels, n_shot, n_val)
        train_db = FewShotDatasetWrapper(train_db, train_index)
        val_db = FewShotDatasetWrapper(val_db, val_index)
        return train_db, val_db, test_db
    else:
        return train_db, test_db