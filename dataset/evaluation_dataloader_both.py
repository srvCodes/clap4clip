import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Sampler
# from torchvision import datasets, transforms
from .imagenet import imagenet
from .imagenet100 import imagenet100
from .cifar import *
# from .CUB200 import Cub2011
import torchvision.datasets as dset
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import pdb

# from idatasets.imagenet import ImageNet
# from idatasets.CUB200 import Cub2011
# from idatasets.omniglot import Omniglot
# from idatasets.celeb_1m import MS1M
import collections
from utils.cutout import Cutout

class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices, shuffle):
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self):
        if(self.shuffle):
            return (self.indices[i] for i in torch.randperm(len(self.indices)))
        else:
            return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
    
    

class IncrementalDataset:

    def __init__(
        self,
        dataset1_name,
        dataset2_name,
        args,
        random_order=False,
        shuffle=True,
        workers=8,
        batch_size=128,
        seed=1,
        increment=10,
        validation_split=0.
    ):
        self.dataset1_name = dataset1_name.lower().strip() #小写，减去空白
        datasets_1 = _get_datasets(dataset1_name)
        self.dataset2_name = dataset2_name.lower().strip() #小写，减去空白
        datasets_2 = _get_datasets(dataset2_name)
        self.train_transforms_1 = datasets_1[0].train_transforms 
        self.common_transforms_1 = datasets_1[0].common_transforms
        self.train_transforms_2 = datasets_2[0].train_transforms 
        self.common_transforms_2 = datasets_2[0].common_transforms
        try:
            self.meta_transforms_1 = datasets_1[0].meta_transforms
            self.meta_transforms_2 = datasets_2[0].meta_transforms
        except:
            self.meta_transforms_1 = datasets_1[0].train_transforms
            self.meta_transforms_2 = datasets_2[0].train_transforms
        self.args = args
        
        self._setup_data(
            datasets_1,
            datasets_2,
            args.root_1,
            args.root_2,
            random_order=random_order,
            seed=seed,
            increment=increment,
            validation_split=validation_split
        )
        

        self._current_task = 0

        self._batch_size = batch_size
        self._workers = workers
        self._shuffle = shuffle
        self.sample_per_task_testing = {}
    # import pdb;pdb.set_trace()
    @property
    def n_tasks(self):
        return len(self.increments)
    
    def get_same_index(self, target, label, mode="train", memory=None):
        label_indices = []
        label_targets = []

        for i in range(len(target)):
            if int(target[i]) in label:
                label_indices.append(i)
                label_targets.append(target[i])
        for_memory = (label_indices.copy(),label_targets.copy()) #for_memory[0]:图片序号 for_memory[1]:图片标签
        
#         if(self.args.overflow and not(mode=="test")):
#             memory_indices, memory_targets = memory
#             return memory_indices, memory
            
        if memory is not None:
            memory_indices, memory_targets = memory
            memory_indices2 = np.tile(memory_indices, (1,))
            all_indices = np.concatenate([memory_indices2,label_indices])
        else:
            all_indices = label_indices
            
        return all_indices, for_memory
    
    def get_same_index_test_chunk(self, target, label, mode="test", memory=None):
        label_indices = []
        label_targets = []
        
        # import pdb;pdb.set_trace()

        np_target = np.array(target, dtype="uint32")   # label of all inputs  ; label: max_class
        np_indices = np.array(list(range(len(target))), dtype="uint32") #0:9999

        for t in range(len(label)//self.args.class_per_task):  #挑选当前任务图片
            task_idx = []
            for class_id in label[t*self.args.class_per_task: (t+1)*self.args.class_per_task]:
                idx = np.where(np_target==class_id)[0]
                task_idx.extend(list(idx.ravel()))
            task_idx = np.array(task_idx, dtype="uint32")
            task_idx.ravel()
            random.shuffle(task_idx)

            label_indices.extend(list(np_indices[task_idx]))  #图片位置
            label_targets.extend(list(np_target[task_idx]))   #图片标签
            if(t not in self.sample_per_task_testing.keys()):
                self.sample_per_task_testing[t] = len(task_idx)
        label_indices = np.array(label_indices, dtype="uint32")
        label_indices.ravel()
        return list(label_indices), label_targets
    

    def new_task(self, ses, memory=None):
        # import pdb;pdb.set_trace()
        self._current_task = ses
        print(self._current_task)
        print(self.increments)      #classed per task
        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])
        pdb.set_trace()
#         if(self.args.overflow):
#             min_class = 0
#             max_class = sum(self.increments)
        # self.train_dataset_targets = self.train_dataset_2.targets + self.train_dataset_1.targets
        # self.test_dataset_targets = self.test_dataset_1.targets + self.test_dataset_2.targets
        train_indices_1, for_memory = self.get_same_index(self.train_dataset_1.targets, list(range(min_class, max_class)), mode="train", memory=memory)
        test_indices_1, _ = self.get_same_index_test_chunk(self.test_dataset_1.targets, list(range(max_class)), mode="test")
        if max_class<sum(self.increments)/2 :
            train_indices_2 = None
            test_indices_2 = None
        else:
            train_indices_2, for_memory = self.get_same_index(self.train_dataset_2.targets, list(range(min_class-int(sum(self.increments)/2), max_class-int(sum(self.increments)/2))), mode="train", memory=memory)
            test_indices_2, _ = self.get_same_index_test_chunk(self.test_dataset_2.targets, list(range(max_class-int(sum(self.increments)/2))), mode="test")
        

        
        pdb.set_trace()
        class_name_1 = self.train_dataset_1.get_classes()
        class_name_2 = self.train_dataset_2.get_classes()
        class_name = class_name_1 + class_name_2
        test_class = class_name[:max_class]
        class_name = class_name[min_class:max_class]
        

        self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset_1, batch_size=self._batch_size,shuffle=False,num_workers=8, sampler=SubsetRandomSampler(train_indices_1, True))
        self.test_data_loader_1 = torch.utils.data.DataLoader(self.test_dataset_1, batch_size=self.args.test_batch,shuffle=False,num_workers=8, sampler=SubsetRandomSampler(test_indices_1, False))
        if max_class<sum(self.increments)/2 :
           self.test_data_loader_2 = None
        else:
            self.test_data_loader_2 = torch.utils.data.DataLoader(self.test_dataset_2, batch_size=self.args.test_batch,shuffle=False,num_workers=8, sampler=SubsetRandomSampler(test_indices_2, False))
        
        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": len(train_indices_1)+len(train_indices_2),
            "n_test_data": len(test_indices_1)+len(test_indices_2)
        }

        self._current_task += 1

        return task_info, self.train_data_loader, class_name, test_class, self.test_data_loader_1, self.test_data_loader_2, for_memory
        
    # for verification   
    def get_galary(self, task, batch_size=10):
        indexes = []
        dict_ind = {}
        seen_classes = []
        for i, t in enumerate(self.train_dataset.targets):
            if not(t in seen_classes) and (t< (task+1)*self.args.class_per_task and (t>= (task)*self.args.class_per_task)):
                seen_classes.append(t)
                dict_ind[t] = i
                
        od = collections.OrderedDict(sorted(dict_ind.items()))
        for k, v in od.items(): 
            indexes.append(v)
            
        data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(indexes, False))
    
        return data_loader 
    def get_custom_loader_idx(self, indexes, mode="train", batch_size=10, shuffle=True):
     
        if(mode=="train"):
            data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(indexes, True))
        else: 
            data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(indexes, False))
    
        return data_loader   
    def get_custom_loader_class(self, class_id, mode="train", batch_size=10, shuffle=False):
        
        if(mode=="train"):
            train_indices, for_memory = self.get_same_index(self.train_dataset.targets, class_id, mode="train", memory=None)
            data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(train_indices, True))
        else: 
            test_indices, _ = self.get_same_index(self.test_dataset.targets, class_id, mode="test")
            data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(test_indices, False))
            
        return data_loader

    def _setup_data(self, datasets_1, datasets_2, path_1, path_2, random_order=False, seed=1, increment=10, validation_split=0.):
        self.increments = []
        self.class_order = []
        
        trsf_train_1 = transforms.Compose(self.train_transforms_1)
        trsf_train_2 = transforms.Compose(self.train_transforms_2)
        # try:
        #     trsf_mata = transforms.Compose(self.meta_transforms)
        # except:
        #     trsf_mata = transforms.Compose(self.train_transforms)
            
        trsf_test_1 = transforms.Compose(self.common_transforms_1)
        trsf_test_2 = transforms.Compose(self.common_transforms_2)
        
        current_class_idx = 0  # When using multiple datasets
        for dataset_1 in datasets_1:
            if(self.dataset1_name=="imagenet" or self.dataset1_name=="imagenet100"):
                train_dataset_1 = dataset_1.base_dataset(root=path_1, train=True, transform=trsf_train_1)
                test_dataset_1 = dataset_1.base_dataset(root=path_1, train=False, transform=trsf_test_1)
                
            elif(self.dataset1_name=="cub200" or self.dataset1_name=="cifar100" or self.dataset1_name=="mnist"  or self.dataset1_name=="caltech101"  or self.dataset1_name=="omniglot"  or self.dataset1_name=="celeb"):
                # pdb.set_trace()
                train_dataset_1 = dataset_1.base_dataset(root=path_1, train=True, transform=trsf_train_1)
                test_dataset_1 = dataset_1.base_dataset(root=path_1, train=False, transform=trsf_test_1)

        for dataset_2 in datasets_2:
            if(self.dataset1_name=="imagenet" or self.dataset1_name=="imagenet100"):
                train_dataset_2 = dataset_2.base_dataset(root=path_2, train=True, transform=trsf_train_2)
                test_dataset_2 = dataset_2.base_dataset(root=path_2, train=False, transform=trsf_test_2)
                
            elif(self.dataset1_name=="cub200" or self.dataset1_name=="cifar100" or self.dataset1_name=="mnist"  or self.dataset1_name=="caltech101"  or self.dataset1_name=="omniglot"  or self.dataset1_name=="celeb"):
                # pdb.set_trace()
                train_dataset_2 = dataset_2.base_dataset(root=path_2, train=True, transform=trsf_train_2)
                test_dataset_2 = dataset_2.base_dataset(root=path_2, train=False, transform=trsf_test_2)
                
        order = [i for i in range(self.args.num_class)]
        if random_order:
            # pdb.set_trace()
            random.seed(seed)  
            random.shuffle(order)
        elif dataset_1.class_order is not None:
            # pdb.set_trace()
            order_1 = dataset_1.class_order
            order_2 = dataset_2.class_order
            self.class_order.append(order_1)
            self.class_order.append(order_2)
            
        for i,t in enumerate(train_dataset_1.targets):
            train_dataset_1.targets[i] = order[t]
        for i,t in enumerate(test_dataset_1.targets):
            test_dataset_1.targets[i] = order[t]
        for i,t in enumerate(train_dataset_2.targets):
            train_dataset_2.targets[i] = order[t]
        for i,t in enumerate(test_dataset_2.targets):
            test_dataset_2.targets[i] = order[t]
        
        # pdb.set_trace()
        self.increments = [increment for _ in range(len(order)// increment)]

        self.train_dataset_1 = train_dataset_1
        self.train_dataset_2 = train_dataset_2
        self.test_dataset_1 = test_dataset_1
        self.test_dataset_2 = test_dataset_2


    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: order.index(x), y)))
    
    
    def get_memory(self, memory, for_memory, seed=1):
        random.seed(seed)
        # pdb.set_trace()
        memory_per_task = self.args.memory // ((self.args.sess+1)*self.args.class_per_task)
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        mu = 1
        
        #update old memory
        if(memory is not None):
            data_memory, targets_memory = memory
            data_memory = np.array(data_memory, dtype="int32")
            targets_memory = np.array(targets_memory, dtype="int32")
            for class_idx in range(self.args.class_per_task*(self.args.sess)):
                idx = np.where(targets_memory==class_idx)[0][:memory_per_task]   #若task=2 取前100/class训练集图片
                self._data_memory = np.concatenate([self._data_memory, np.tile(data_memory[idx], (mu,))   ])
                self._targets_memory = np.concatenate([self._targets_memory, np.tile(targets_memory[idx], (mu,))    ])
                
                
        #add new classes to the memory
        new_indices, new_targets = for_memory

        new_indices = np.array(new_indices, dtype="int32")
        new_targets = np.array(new_targets, dtype="int32")
        for class_idx in range(self.args.class_per_task*(self.args.sess),self.args.class_per_task*(1+self.args.sess)):
            idx = np.where(new_targets==class_idx)[0][:memory_per_task]
            self._data_memory = np.concatenate([self._data_memory, np.tile(new_indices[idx],(mu,))   ])
            self._targets_memory = np.concatenate([self._targets_memory, np.tile(new_targets[idx],(mu,))    ])
            
        print(len(self._data_memory))
        return list(self._data_memory.astype("int32")), list(self._targets_memory.astype("int32"))
    
def _get_datasets(dataset1_names):
    return [_get_dataset(dataset1_name) for dataset1_name in dataset1_names.split("-")]


def _get_dataset(dataset1_name):
    dataset1_name = dataset1_name.lower().strip()

    if dataset1_name == "cifar10":
        return iCIFAR10
    elif dataset1_name == "cifar100":
        return iCIFAR100
    elif dataset1_name == "imagenet":
        return iIMAGENET
    elif dataset1_name == "imagenet100":
        return iIMAGENET100
    # elif dataset1_name == "cub200":
    #     return iCUB200
    # elif dataset1_name == "mnist":
    #     return iMNIST

    
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset1_name))

class DataHandler:
    base_dataset = None
    train_transforms = []
    mata_transforms = [transforms.ToTensor()]
    common_transforms = [transforms.ToTensor()]
    class_order = None

class iCIFAR10(DataHandler):
    base_dataset = cifar10
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]

class iCIFAR100(DataHandler):
    base_dataset = cifar100
    train_transforms = [
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        lambda image: image.convert("RGB"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954,0.26130258, 0.27577711)),
    ]
    
    common_transforms = [
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954,0.26130258, 0.27577711)),
    ]
    
class iIMAGENET(DataHandler):
    base_dataset = imagenet
    train_transforms = [
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        lambda image: image.convert("RGB"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954,0.26130258, 0.27577711)),
    ]
    common_transforms = [
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954,0.26130258, 0.27577711)),
    ]


class iIMAGENET100(DataHandler):
    base_dataset = imagenet100
    train_transforms = [
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        lambda image: image.convert("RGB"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954,0.26130258, 0.27577711)),
    ]
    common_transforms = [
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954,0.26130258, 0.27577711)),
    ]

# class iCUB200(DataHandler):
#     base_dataset = Cub2011
#     train_transforms = [
#         transforms.Resize(230),
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ColorJitter(brightness=63 / 255),
#         transforms.ToTensor(),
        
#     ]
#     common_transforms = [
#         transforms.Resize(230),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#     ]

    
# class iMNIST(DataHandler):
#     base_dataset = dset.MNIST
#     train_transforms = [ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ]
#     common_transforms = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]

