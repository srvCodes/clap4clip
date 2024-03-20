import os
import copy 
import yaml
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from classifier.utils import SubsetRandomSampler
# from torchvision import datasets, transforms
from .imagenet import imagenet
from .imagenetr import imagenetR
from .cifar import *
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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
    

class IncrementalDataset:

    def __init__(
        self,
        dataset_name,
        args,
        random_order=False,
        shuffle=True,
        workers=8,
        batch_size=128,
        seed=1,
        increment=10,
        validation_split=0.,
        exemplar_selector=None, 
        offset = 0
    ):
        if type(dataset_name) is not list:
            dataset_name = [dataset_name]
        self.dataset_names = [name.lower().strip() for name in dataset_name]
        datasets = _get_datasets(self.dataset_names)
        self.train_transforms = datasets[0].train_transforms 
        self.common_transforms = datasets[0].common_transforms
        try:
            self.meta_transforms = datasets[0].meta_transforms
        except:
            self.meta_transforms = datasets[0].train_transforms
        self.args = args
        self.offset = offset # offset for labels
        self._setup_data(
            datasets,
            args.root,
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
        self.new_task_memory_start_point = 0 
        self.prompt_templates = self.get_prompts(datasets=datasets)
        self.exemplar_selector = exemplar_selector
        self.test_data_loaders = []
        self.test_indices_len = []
        self.past_memory_dataset = None 
    # import pdb;pdb.set_trace()

    @staticmethod
    def get_prompts(datasets):
        prompt_templates = [each for dataset in datasets for each in dataset.base_dataset.templates]
        return list(set(prompt_templates))
    
    @property
    def n_tasks(self):
        return len(self.increments)
    
    def get_same_index(self, target, label, mode="train", memory=None):
        label_indices = []
        label_targets = []

        np_target = np.array(target, dtype="uint32")   # label of all inputs  ; label: max_class
        # pdb.set_trace()
        if self.args.fscil and self.args.sess > 0:
            """
            Only select k number of instances without replacement for each label.
            Also, to be aligned with previous works, the label order is permuted at random.
            Source: https://github.com/xyutao/fscil/blob/master/dataloader/dataloader.py#L76C13-L76C87
            """
            select_index = list()
            np.random.seed(0)
            label = np.random.choice(label,size=len(label),replace=False)
            for cls_id in label:
                ind = list(np.where(target==cls_id)[0])
                np.random.seed(1)
                random_ind = np.random.choice(ind,self.args.k_shot,replace=False)
                label_indices.extend(random_ind)
                label_targets.extend([cls_id]*len(random_ind))
        else:
            for i in range(len(target)):
                if int(target[i]) in label:
                    label_indices.append(i)
                    label_targets.append(target[i])

        for_memory = (label_indices.copy(),label_targets.copy()) 
       
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
        task_idx = []
        for class_id in label:
            idx = np.where(np_target==class_id)[0]
            task_idx.extend(list(idx.ravel()))
        task_idx = np.array(task_idx, dtype="uint32")
        task_idx.ravel()
        random.shuffle(task_idx)

        label_indices.extend(list(np_indices[task_idx]))  
        label_targets.extend(list(np_target[task_idx]))   
        t = self.args.sess + 1
        if(t not in self.sample_per_task_testing.keys()):
            self.sample_per_task_testing[t] = len(task_idx)
        label_indices = np.array(label_indices, dtype="uint32")
        label_indices.ravel()
        return list(label_indices), label_targets
    
    def get_same_index_test_chunk_v0(self, target, label, mode="test", memory=None):
        label_indices = []
        label_targets = []
        
        # import pdb;pdb.set_trace()

        np_target = np.array(target, dtype="uint32")   # label of all inputs  ; label: max_class
        np_indices = np.array(list(range(len(target))), dtype="uint32") #0:9999

        for t in range(len(label)//self.args.class_per_task):  
            task_idx = []
            for class_id in label[t*self.args.class_per_task: (t+1)*self.args.class_per_task]:
                idx = np.where(np_target==class_id)[0]
                task_idx.extend(list(idx.ravel()))
            task_idx = np.array(task_idx, dtype="uint32")
            task_idx.ravel()
            random.shuffle(task_idx)

            label_indices.extend(list(np_indices[task_idx]))  
            label_targets.extend(list(np_target[task_idx]))   
            if(t not in self.sample_per_task_testing.keys()):
                self.sample_per_task_testing[t] = len(task_idx)
        label_indices = np.array(label_indices, dtype="uint32")
        label_indices.ravel()
        return list(label_indices), label_targets

    def get_past_dataset_loader(self):
        return self.past_dataset_mem_loader
    
    def get_future_tasks_test_loader(self):
        min_class = 0
        future_test_loader = None
        if self.args.eval_ood_score and (self._current_task + 1 < len(self.increments)):
            # check if there are future tasks
            i = self._current_task
            # for i, _ in enumerate(self.increments[self._current_task+1:]):
            min_class = sum(self.increments[:i+1]) + self.offset # min class is the next task's min
            max_class = sum(self.increments) + self.offset # max class possible
            test_indices, _ = self.get_same_index_test_chunk(self.test_dataset.targets, list(range(min_class, max_class)), mode="test")
            future_test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.test_batch,
                                                                shuffle=False,num_workers=8, 
                                                                sampler=SubsetRandomSampler(test_indices, False),
                                                                worker_init_fn=seed_worker,
                                                                generator=g,)
        return future_test_loader
                
    def new_task(self, memory=None, past_dataset_memory=None):
        # import pdb;pdb.set_trace()
        print(f"Cur sess: {self.args.sess}, Cur task: {self._current_task}")
        print("Increments: ", self.increments)      #classed per task
        min_class = sum(self.increments[:self._current_task]) + self.offset
        max_class = sum(self.increments[:self._current_task + 1]) + self.offset
        train_indices, for_memory = self.get_same_index(self.train_dataset.targets, list(range(min_class, max_class)), mode="train", memory=memory)
        test_indices, _ = self.get_same_index_test_chunk(self.test_dataset.targets, list(range(min_class, max_class)), mode="test")
        # pdb.set_trace()
        class_name = [self.train_dataset.get_classes()[i-self.offset] for i in self.class_order]
        # class_name = self.train_dataset.get_classes()
        test_class = class_name[:max_class-self.offset]
        class_name = class_name[min_class-self.offset:max_class-self.offset]
        
        
        
        self.test_data_loaders.append(torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.test_batch,
                                                                  shuffle=False,num_workers=8, 
                                                                  sampler=SubsetRandomSampler(test_indices, False),
                                                                  worker_init_fn=seed_worker,
                                                                    generator=g,))
        ood_test_loader = self.get_future_tasks_test_loader()
        if past_dataset_memory is not None:
            if self.past_memory_dataset is None:
                # past dataset memory indices will remain constant so retrieve them only once here
                past_dataset_memory_indices, _ = past_dataset_memory
                self.past_memory_dataset = torch.utils.data.Subset(self.prev_train_dataset, past_dataset_memory_indices)
            
            # This does not work, surprisingly! Not sure what's wrong.
            # total_train_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.prev_train_dataset])
            # all_indices = np.concatenate([train_indices,  past_dataset_memory_indices])
            # self.train_data_loader = torch.utils.data.DataLoader(total_train_dataset, batch_size=self._batch_size,
            #                                                  shuffle=False,num_workers=8, 
            #                                                  sampler=SubsetRandomSampler(all_indices, True), 
            #                                                  worker_init_fn=seed_worker,
            #                                                  generator=g)
            # n_train_data = len(all_indices)

            # once retrieved, combine the past dataset memory samples with the current dataset training samples
            curr_train_dataset = torch.utils.data.Subset(self.train_dataset, train_indices)
            total_train_dataset = torch.utils.data.ConcatDataset([curr_train_dataset, self.past_memory_dataset])
            self.train_data_loader = torch.utils.data.DataLoader(total_train_dataset, batch_size=self._batch_size,
                                                             shuffle=True,num_workers=8, 
                                                             worker_init_fn=seed_worker,
                                                             generator=g)
            n_train_data = len(train_indices) + len(self.past_memory_dataset)
        else:
            self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self._batch_size,
                                                             shuffle=False,num_workers=8, 
                                                             sampler=SubsetRandomSampler(train_indices, True), 
                                                             worker_init_fn=seed_worker,
                                                             generator=g)
            n_train_data = len(train_indices)
        self.test_indices_len.append(len(test_indices))
        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": n_train_data,
            "n_test_data": sum(self.test_indices_len),
            "memory size": len(self.past_memory_dataset) if self.past_memory_dataset is not None else 0
        }
        self._current_task += 1
        return task_info, self.train_data_loader, class_name, test_class, self.test_data_loaders, for_memory, ood_test_loader
    
    def get_memory_loader(self, memory):
        memory_indices, _ = memory
        if self.past_memory_dataset is not None:
            curr_train_dataset = torch.utils.data.Subset(self.train_dataset, memory_indices)
            # this is the class-balanced dataset to be used for finetuning
            total_train_dataset = torch.utils.data.ConcatDataset([curr_train_dataset, self.past_memory_dataset])
            print(f"Class-balanced finetuning dataset size: {len(total_train_dataset) // (self.args.sess + 1)} per task over {(self.args.sess + 1)} tasks")
            memory_loader = torch.utils.data.DataLoader(total_train_dataset, batch_size=self._batch_size,
                                                             shuffle=True,num_workers=8, 
                                                             worker_init_fn=seed_worker,
                                                             generator=g)
        else:
            print(f"Class-balanced finetuning dataset size: {len(memory_indices) // (self.args.sess + 1)} per task over {(self.args.sess + 1)} tasks")
            memory_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self._batch_size,
                                                        shuffle=False,num_workers=8, 
                                                        sampler=SubsetRandomSampler(memory_indices, True), 
                                                        worker_init_fn=seed_worker,
                                                        generator=g,)
        return memory_loader

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
            
        data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, 
                                                  num_workers=4, sampler=SubsetRandomSampler(indexes, False), 
                                                  worker_init_fn=seed_worker,
                                                             generator=g)
    
        return data_loader
    
    
    def get_custom_loader_idx(self, indexes, mode="train", batch_size=10, shuffle=True):
     
        if(mode=="train"):
            data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(indexes, True), worker_init_fn=seed_worker,
                                                             generator=g)
        else: 
            data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(indexes, False), worker_init_fn=seed_worker,
                                                             generator=g)
    
        return data_loader
    
    
    def get_custom_loader_class(self, class_id, mode="train", batch_size=10, shuffle=False):
        
        if(mode=="train"):
            train_indices, for_memory = self.get_same_index(self.train_dataset.targets, class_id, mode="train", memory=None)
            data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(train_indices, True))
        else: 
            test_indices, _ = self.get_same_index(self.test_dataset.targets, class_id, mode="test")
            data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(test_indices, False))
            
        return data_loader

    def load_class_order(self, random=False, num_classes=None):
        if random:
            seed_val = self.args.seed if self.args.fscil else 1993 + self.args.num_run
            np.random.seed(seed_val)
            labels = np.arange(num_classes, dtype=int)
            np.random.shuffle(labels)
            return labels
        else:
            with open(f"dataset/class_order/{self.dataset_names[-1]}_order1.yaml", "r") as stream:
                text = yaml.safe_load(stream)
                return text['class_order']

    def _setup_data(self, datasets, path, random_order=False, seed=1, increment=10, validation_split=0.):
        self.increments = []
        self.class_order = []
        
        trsf_train = transforms.Compose(self.train_transforms)
        trsf_test = transforms.Compose(self.common_transforms)
        
        current_class_idx = 0  # When using multiple datasets
        for i, dataset in enumerate(datasets):
            if(self.dataset_names[i]=="imagenet"or self.dataset_names[i]=="imagenet100" or self.dataset_names[i]=="imagenet-r"):
                train_dataset = dataset.base_dataset(root=path, train=True, transform=trsf_train)
                test_dataset = dataset.base_dataset(root=path, train=False, transform=trsf_test)
                # traindir = os.path.join(path, 'imagenet100/train')
                # validdir = os.path.join(path, 'imagenet100/val')
                # train_dataset = dset.ImageFolder(traindir, transform=trsf_train)
                # test_dataset = dset.ImageFolder(validdir, transform=trsf_test)
                # print(test_dataset.labels); exit(1)
            
            elif(self.dataset_names[i]=="cub200" or self.dataset_names[i]=="vtab" or self.dataset_names[i]=="cifar100" \
                        or self.dataset_names[i]=="mnist"  or self.dataset_names[i]=="caltech101"  
                        or self.dataset_names[i]=="omniglot"  or self.dataset_names[i]=="celeb"):
                # pdb.set_trace()
                train_dataset = dataset.base_dataset(root=path, train=True, transform=trsf_train)
                test_dataset = dataset.base_dataset(root=path, train=False, transform=trsf_test)


            elif(self.dataset_names[i]=="svhn"):
                train_dataset = dataset.base_dataset(root=path, split='train', transform=trsf_train)
                test_dataset = dataset.base_dataset(root=path, split='test', transform=trsf_test)
                train_dataset.targets = train_dataset.labels
                test_dataset.targets = test_dataset.labels

            # current_class_idx = len(set(train_dataset.targets))
            if self.dataset_names[i] == 'vtab':
                order = np.arange(self.args.num_class, dtype=int)  
            else:
                order = self.load_class_order(random=True, num_classes= dataset.num_classes)

            sorted_class_order = np.asarray(order, dtype=np.int32).argsort()
            train_dataset.targets = sorted_class_order[train_dataset.targets]
            test_dataset.targets = sorted_class_order[test_dataset.targets]
            if len(datasets) == 2:
                if i == 0:
                    # this is true when two datasets are passed simultaneously
                    # here the first dataset is saved for the memory rehearsal purpose
                    self.prev_train_dataset = train_dataset
                    self.prev_class_order = order
                else:
                    train_dataset.targets = list(map(lambda x:x+self.offset, train_dataset.targets) )
                    test_dataset.targets = list(map(lambda x:x+self.offset, test_dataset.targets) )
            order = list(map(lambda x:x+self.offset, order) )
            self.class_order = order
            if not self.args.fscil:
                self.increments = [increment for _ in range(len(order) // increment)]
            else:
                self.increments = self.get_variable_increment_num()

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    @staticmethod
    def get_variable_increment_num():
        num_tasks = 9
        base_task_cls_num = 60
        incr_task_cls_num = 5
        task_split =  [base_task_cls_num]
        task_split.extend([incr_task_cls_num for _ in range(num_tasks-1)])
        return task_split

    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: order.index(x), y)))
    
    def flush_curr_memory(self):
        self._data_memory, self._targets_memory = np.array([]), np.array([])

    @torch.no_grad()
    def get_memory(self, model, memory, for_memory, seed=1):
        random.seed(seed)
        self.flush_curr_memory()
        if self.args.memory_type == 'fix_total':
            effective_sess = (self.args.sess+1)
            if self.offset > 0:
                effective_sess -= self.args.start_sess
                
            memory_per_cls = self.args.memory // (effective_sess*self.args.class_per_task)
        elif self.args.memory_type == 'fix_per_cls':
            memory_per_cls = 20 
        
        self.exemplar_selector.set_dataset_and_transform(self.train_dataset, self.common_transforms)

        self._data_memory, self._targets_memory = self.exemplar_selector.select_indices(model, memory_per_cls, 
                                                                                        memory, for_memory)


        return list(self._data_memory.astype("int32")), list(self._targets_memory.astype("int32"))

    def get_memory_v0(self, memory, for_memory, seed=1):
        random.seed(seed)
        # pdb.set_trace()
        if self.args.memory_type == 'fix_total':
            memory_per_task = self.args.memory // ((self.args.sess+1)*self.args.class_per_task)
        elif self.args.memory_type == 'fix_per_cls':
            memory_per_task = 20 #* ((self.args.sess+1)*self.args.class_per_task)
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        mu = 1
        
        #update old memory
        if(memory is not None):
            data_memory, targets_memory = memory
            data_memory = np.array(data_memory, dtype="int32")
            targets_memory = np.array(targets_memory, dtype="int32")
            for class_idx in range(self.args.class_per_task*(self.args.sess)):
                idx = np.where(targets_memory==class_idx)[0][:memory_per_task]   
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
            
        return list(self._data_memory.astype("int32")), list(self._targets_memory.astype("int32"))
    
def _get_datasets(dataset_names):
    return [_get_dataset(dataset_name) for dataset_name in dataset_names]


def _get_dataset(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "cifar10":
        return iCIFAR10
    elif dataset_name == "cifar100":
        return iCIFAR100
    elif dataset_name == "imagenet":
        return iIMAGENET
    elif dataset_name == "imagenet-r":
        return iIMAGENETR

    
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))

def build_transform(is_train, args):
    input_size = 224

    t=[  
        transforms.Resize((224,224),transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ]
    return t

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
    num_classes = 100
    # class_order = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39]
    # class_order = [58, 30, 93, 69, 21, 77, 3, 78, 12, 71, 65, 40, 16, 49, 89, 46, 24, 66, 19, 41, 5, 29, 15, 73, 11, 70, 90, 63, 67, 25, 59, 72, 80, 94, 54, 33, 18, 96, 2, 10, 43, 9, 57, 81, 76, 50, 32, 6, 37, 7, 68, 91, 88, 95, 85, 4, 60, 36, 22, 27, 39, 42, 34, 51, 55, 28, 53, 48, 38, 17, 83, 86, 56, 35, 45, 79, 99, 84, 97, 82, 98, 26, 47, 44, 62, 13, 31, 0, 75, 14, 52, 74, 8, 20, 1, 92, 87, 23, 64, 61]
    # class_order = [71, 54, 45, 32, 4, 8, 48, 66, 1, 91, 28, 82, 29, 22, 80, 27, 86, 23, 37, 47, 55, 9, 14, 68, 25, 96, 36, 90, 58, 21, 57, 81, 12, 26, 16, 89, 79, 49, 31, 38, 46, 20, 92, 88, 40, 39, 98, 94, 19, 95, 72, 24, 64, 18, 60, 50, 63, 61, 83, 76, 69, 35, 0, 52, 7, 65, 42, 73, 74, 30, 41, 3, 6, 53, 13, 56, 70, 77, 34, 97, 75, 2, 17, 93, 33, 84, 99, 51, 62, 87, 5, 15, 10, 78, 67, 44, 59, 85, 43, 11]
    
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

class iIMAGENETR(DataHandler):
    base_dataset = imagenetR
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
    num_classes = 200

    
# class iMNIST(DataHandler):
#     base_dataset = mnist
#     train_transforms = [ transforms.Resize(224),transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ]
#     common_transforms = [transforms.Resize(224),transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]

