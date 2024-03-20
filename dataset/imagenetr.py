from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from torchvision import datasets
import torch
from shutil import move, rmtree

def prepare_imagenet_r(fpath="/home/srv/Documents/mammoth_datasets/imagenet-r/"):
    if not os.path.exists(fpath + '/train') and not os.path.exists(fpath + '/test'):
        dataset = datasets.ImageFolder(fpath, transform=None)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train, val = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_idx, val_idx = train.indices, val.indices

        train_file_list = [dataset.imgs[i][0] for i in train_idx]
        test_file_list = [dataset.imgs[i][0] for i in val_idx]

            
        train_folder = fpath + '/train'
        test_folder = fpath + '/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        for c in dataset.classes:
            if not os.path.exists(os.path.join(train_folder, c)):
                os.mkdir(os.path.join(os.path.join(train_folder, c)))
            if not os.path.exists(os.path.join(test_folder, c)):
                os.mkdir(os.path.join(os.path.join(test_folder, c)))
        
        for path in train_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(train_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)

        for path in test_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(test_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)
        
        for c in dataset.classes:
            path = os.path.join(fpath, c)
            rmtree(path)

class imagenetR(Dataset):

    templates = [
        'a photo of a {}.',
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',
        ]
        
    new_classes = ['goldfish', 'great_white_shark', 'hammerhead', 'stingray', 'hen', 'ostrich', 'goldfinch', 
               'junco', 'bald_eagle', 'vulture', 'newt', 'axolotl', 'tree_frog', 'iguana', 'African_chameleon',
                'cobra', 'scorpion', 'tarantula', 'centipede', 'peacock', 'lorikeet', 'hummingbird', 'toucan', 
                'duck', 'goose', 'black_swan', 'koala', 'jellyfish', 'snail', 'lobster', 'hermit_crab', 'flamingo',
                'american_egret', 'pelican', 'king_penguin', 'grey_whale', 'killer_whale', 'sea_lion', 'chihuahua',
                'shih_tzu', 'afghan_hound', 'basset_hound', 'beagle', 'bloodhound', 'italian_greyhound', 'whippet',
                'weimaraner', 'yorkshire_terrier', 'boston_terrier', 'scottish_terrier', 
                'west_highland_white_terrier', 'golden_retriever', 'labrador_retriever', 'cocker_spaniels',
                'collie', 'border_collie', 'rottweiler', 'german_shepherd_dog', 'boxer', 'french_bulldog', 
                'saint_bernard', 'husky', 'dalmatian', 'pug', 'pomeranian', 'chow_chow', 
                'pembroke_welsh_corgi', 'toy_poodle', 'standard_poodle', 'timber_wolf', 'hyena', 'red_fox',
                'tabby_cat', 'leopard', 'snow_leopard', 'lion', 'tiger', 'cheetah', 'polar_bear', 'meerkat',
                'ladybug', 'fly', 'bee', 'ant', 'grasshopper', 'cockroach', 'mantis', 'dragonfly', 
                'monarch_butterfly', 'starfish', 'wood_rabbit', 'porcupine', 'fox_squirrel', 'beaver', 'guinea_pig',
                'zebra', 'pig', 'hippopotamus', 'bison', 'gazelle', 'llama', 'skunk', 'badger', 'orangutan', 
                'gorilla', 'chimpanzee', 'gibbon', 'baboon', 'panda', 'eel', 'clown_fish', 'puffer_fish', 
                'accordion', 'ambulance', 'assault_rifle', 'backpack', 'barn', 'wheelbarrow', 'basketball', 'bathtub',
                'lighthouse', 'beer_glass', 'binoculars', 'birdhouse', 'bow_tie', 'broom', 'bucket', 'cauldron', 'candle', 
                'cannon', 'canoe', 'carousel', 'castle', 'mobile_phone', 'cowboy_hat', 'electric_guitar', 'fire_engine', 'flute', 
                'gasmask', 'grand_piano', 'guillotine', 'hammer', 'harmonica', 'harp', 'hatchet', 'jeep', 'joystick', 'lab_coat',
                'lawn_mower', 'lipstick', 'mailbox', 'missile', 'mitten', 'parachute', 'pickup_truck', 'pirate_ship', 'revolver', 
                'rugby_ball', 'sandal', 'saxophone', 'school_bus', 'schooner', 'shield', 'soccer_ball', 'space_shuttle', 'spider_web',
                'steam_locomotive', 'scarf', 'submarine', 'tank', 'tennis_ball', 'tractor', 'trombone', 'vase', 'violin',
                'military_aircraft', 'wine_bottle', 'ice_cream', 'bagel', 'pretzel', 'cheeseburger', 'hotdog', 'cabbage', 'broccoli',
                'cucumber', 'bell_pepper', 'mushroom', 'Granny_Smith', 'strawberry', 'lemon', 'pineapple', 'banana', 'pomegranate', 
                'pizza', 'burrito', 'espresso', 'volcano', 'baseball_player', 'scuba_diver', 'acorn']
    


    def __init__(self, root, transform=None,train=True):
        split = 'train' if train else 'test'
        self.split = split
        self.classes = self.new_classes
        self.root = root
        self.datadir = os.path.join(root, f'imagenet-r/{split}')
        self.transform = transform
        # self.prepare_files_()
        self._load_meta()

    def prepare_files_(self):
        metadata_path = os.path.join(self.root, "imagenet-r/README.txt")
        self.data, self.targets = [], []
        with open(metadata_path) as f:
            lines = [line for line in f.readlines()[13:]]
        f.close()
        dir_names = ["".join(line.split(" ")[0]) for line in lines]
        relative_path_and_labels = []
        for class_id, dir in enumerate(dir_names):
            for dirpath, dnames, fnames in os.walk(f"{self.datadir}/{dir}"):
                for fname in fnames:
                    relative_fpath = f"{self.split}/{dir}/{fname}"
                    to_write = f"{relative_fpath} {class_id}\n"
                    relative_path_and_labels.append(to_write)
        with open(f"imagenet_split/imagenetr_{self.split}.txt", "w") as f:
            f.writelines(relative_path_and_labels)    
        

    def _load_meta(self):
        metadata_path = f"imagenet_split/imagenetr_{self.split}.txt"
        self.data, self.targets = [], []
        with open(metadata_path) as f:
            for line in f:
                path, target = line.strip().split(" ")
                self.data.append(os.path.join(self.root, f'imagenet-r/{path}'))
                self.targets.append(int(target))
        self.data = np.array(self.data)


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.open(img).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img,target,int(index)
    
    def __len__(self):
        return len(self.data)
    
    def prompts(self,mode='single'):
        if mode == 'single':
            prompts = [[self.imagenet_templates[0].format(label)] for label in self.new_classes]
            return prompts
        elif mode == 'ensemble':
            prompts = [[template.format(label) for template in self.imagenet_templates] for label in self.new_classes]
            return prompts

    def get_labels(self):
        return np.array(self.targets)

    def get_classes(self):
        return self.new_classes
    
# first download the imagenet-r.tar file
# then uncomment this to prepare the train and test datasets
# prepare_imagenet_r()