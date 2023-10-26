import sys
import os.path as osp
import os
import logging

base_path = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(base_path)

import clip
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, ConcatDataset, Subset

from PIL import ImageFile
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageTextData(object):
    def __init__(self, root, client_name, task_id, preprocess, prompt="a picture of a"):
        client_data_path = osp.join(root, client_name, f"task{task_id}")
        
        if len(os.listdir(client_data_path)) > 1:
            logging.info(f"loading data in {client_data_path} with {os.listdir(client_data_path)}")
            data = []
            for domain in os.listdir(client_data_path):
                data.append(datasets.ImageFolder(osp.join(client_data_path, domain),transform=ImageTextData._TRANSFORM))
            data = ConcatDataset(data)
        else:
            logging.info(f"loading data in {osp.join(client_data_path, os.listdir(client_data_path)[0])}")
            data = datasets.ImageFolder(
                osp.join(client_data_path, os.listdir(client_data_path)[0]),
                transform=self._TRANSFORM,
            )

        self.data = data
        self.labels = data.classes

        if prompt:
            self.labels = [prompt + " " + x for x in self.labels]

        self.preprocess = preprocess
        self.text = clip.tokenize(self.labels)

    def __getitem__(self, index):
        image, label = self.data.imgs[index]
        if self.preprocess is not None:
            image = self.preprocess(Image.open(image))
        text_enc = self.text[label]
        return image, text_enc, label

    def __len__(self):
        return len(self.data)

    _TRANSFORM = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class DomainDataset:
    def __init__(self, args, preprocess):
        self.domains = args.domains
        self.root_dir = args.root_dir
        self.preprocess = preprocess
        self.random_seed = args.seed
        self.batch_size = args.batch
        self.train_percentage = args.datapercent
        self.set_labels()


    def get_dataloader(self, task_id):
        train_datasets, test_datasets = [], []
        train_dataloaders, test_dataloaders = [], []
        
        for _, client_name in enumerate(os.listdir(self.root_dir)):
            data = ImageTextData(self.root_dir, client_name, task_id, self.preprocess)
            l = len(data)
            index = np.arange(l)
            np.random.seed(self.random_seed)
            np.random.shuffle(index)
            l1, l2 = int(l * self.train_percentage), int(l * (1 - self.train_percentage))
            train_datasets.append(Subset(data, index[:l1]))
            test_datasets.append(Subset(data, index[l1 : l1 + l2 + 1]))
            train_dataloaders.append(
                DataLoader(
                    train_datasets[-1], batch_size=self.batch_size, shuffle=True))
            test_dataloaders.append(
                DataLoader(
                    test_datasets[-1], batch_size=self.batch_size, shuffle=False)) # build dataloader for each clients
        
        return train_dataloaders, test_dataloaders, self.labels
    
    def set_labels():
        raise NotImplementedError
    
class DoaminNet(DomainDataset): 
    def __init__(self, args, preprocess) -> None:
        super().__init__(args, preprocess)
          
    def set_labels(self):
        self.labels = sorted(['aircraft_carrier', 'chandelier', 'harp', 'palm_tree', 'spider', 'airplane', 'church', 'hat', 'panda', 'spoon', 'alarm_clock', 'circle', 'headphones', 'pants', 'spreadsheet', 'ambulance', 'clarinet', 'hedgehog', 'paper_clip', 'square', 'angel', 'clock', 'helicopter', 'parachute', 'squiggle', 'animal_migration', 'cloud', 'helmet', 'parrot', 'squirrel', 'ant', 'coffee_cup', 'hexagon', 'passport', 'stairs', 'anvil', 'compass', 'hockey_puck', 'peanut', 'star', 'apple', 'computer', 'hockey_stick', 'pear', 'steak', 'arm', 'cookie', 'horse', 'peas', 'stereo', 'asparagus', 'cooler', 'hospital', 'pencil', 'stethoscope', 'axe', 'couch', 'hot_air_balloon', 'penguin', 'stitches', 'backpack', 'cow', 'hot_dog', 'piano', 'stop_sign', 'banana', 'crab', 'hot_tub', 'pickup_truck', 'stove', 'bandage', 'crayon', 'hourglass', 'picture_frame', 'strawberry', 'barn', 'crocodile', 'house', 'pig', 'streetlight', 'baseball', 'crown', 'house_plant', 'pillow', 'string_bean', 'baseball_bat', 'cruise_ship', 'hurricane', 'pineapple', 'submarine', 'basket', 'cup', 'ice_cream', 'pizza', 'suitcase', 'basketball', 'diamond', 'jacket', 'pliers', 'sun', 'bat', 'dishwasher', 'jail', 'police_car', 'swan', 'bathtub', 'diving_board', 'kangaroo', 'pond', 'sweater', 'beach', 'dog', 'key', 'pool', 'swing_set', 'bear', 'dolphin', 'keyboard', 'popsicle', 'sword', 'beard', 'donut', 'knee', 'postcard', 'syringe', 'bed', 'door', 'knife', 'potato', 'table', 'bee', 'dragon', 'ladder', 'power_outlet', 'teapot', 'belt', 'dresser', 'lantern', 'purse', 'teddy-bear', 'bench', 'drill', 'laptop', 'rabbit', 'telephone', 'bicycle', 'drums', 'leaf', 'raccoon', 'television', 'binoculars', 'duck', 'leg', 'radio', 'tennis_racquet', 'bird', 'dumbbell', 'light_bulb', 'rain', 'tent', 'birthday_cake', 'ear', 'lighter', 'rainbow', 'The_Eiffel_Tower', 'blackberry', 'elbow', 'lighthouse', 'rake', 'The_Great_Wall_of_China', 'blueberry', 'elephant', 'lightning', 'remote_control', 'The_Mona_Lisa', 'book', 'envelope', 'line', 'rhinoceros', 'tiger', 'boomerang', 'eraser', 'lion', 'rifle', 'toaster', 'bottlecap', 'eye', 'lipstick', 'river', 'toe', 'bowtie', 'eyeglasses', 'lobster', 'roller_coaster', 'toilet', 'bracelet', 'face', 'lollipop', 'rollerskates', 'tooth', 'brain', 'fan', 'mailbox', 'sailboat', 'toothbrush', 'bread', 'feather', 'map', 'sandwich', 'toothpaste', 'bridge', 'fence', 'marker', 'saw', 'tornado', 'broccoli', 'finger', 'matches', 'saxophone', 'tractor', 'broom', 'fire_hydrant', 'megaphone', 'school_bus', 'traffic_light', 'bucket', 'fireplace', 'mermaid', 'scissors', 'train', 'bulldozer', 'firetruck', 'microphone', 'scorpion', 'tree', 'bus', 'fish', 'microwave', 'screwdriver', 'triangle', 'bush', 'flamingo', 'monkey', 'sea_turtle', 'trombone', 'butterfly', 'flashlight', 'moon', 'see_saw', 'truck', 'cactus', 'flip_flops', 'mosquito', 'shark', 'trumpet', 'cake', 'floor_lamp', 'motorbike', 'sheep', 't-shirt', 'calculator', 'flower', 'mountain', 'shoe', 'umbrella', 'calendar', 'flying_saucer', 'mouse', 'shorts', 'underwear', 'camel', 'foot', 'moustache', 'shovel', 'van', 'camera', 'fork', 'mouth', 'sink', 'vase', 'camouflage', 'frog', 'mug', 'skateboard', 'violin', 'campfire', 'frying_pan', 'mushroom', 'skull', 'washing_machine', 'candle', 'garden', 'nail', 'skyscraper', 'watermelon', 'cannon', 'garden_hose', 'necklace', 'sleeping_bag', 'waterslide', 'canoe', 'giraffe', 'nose', 'smiley_face', 'whale', 'car', 'goatee', 'ocean', 'snail', 'wheel', 'carrot', 'golf_club', 'octagon', 'snake', 'windmill', 'castle', 'grapes', 'octopus', 'snorkel', 'wine_bottle', 'cat', 'grass', 'onion', 'snowflake', 'wine_glass', 'ceiling_fan', 'guitar', 'oven', 'snowman', 'wristwatch', 'cello', 'hamburger', 'owl', 'soccer_ball', 'yoga', 'cell_phone', 'hammer', 'paintbrush', 'sock', 'zebra', 'chair', 'hand', 'paint_can', 'speedboat', 'zigzag'])




class Officehome(DomainDataset):
    def __init__(self, args, preprocess) -> None:
        super().__init__(args, preprocess)
        
    def set_labels(self):
        self.labels = sorted(['Alarm_Clock', 'Chair', 'File_Cabinet', 'Knives', 'Pan', 'Scissors', 'ToothBrush', 'Backpack', 'Clipboards', 'Flipflops', 'Lamp_Shade', 'Paper_Clip', 'Screwdriver', 'Toys', 'Batteries', 'Computer', 'Flowers', 'Laptop', 'Pen', 'Shelf', 'Trash_Can', 'Bed', 'Couch', 'Folder', 'Marker', 'Pencil', 'Sink', 'TV', 'Bike', 'Curtains', 'Fork', 'Monitor', 'Postit_Notes', 'Sneakers', 'Webcam', 'Bottle', 'Desk_Lamp', 'Glasses', 'Mop', 'Printer', 'Soda', 'Bucket', 'Drill', 'Hammer', 'Mouse', 'Push_Pin', 'Speaker', 'Calculator', 'Eraser', 'Helmet', 'Mug', 'Radio', 'Spoon', 'Calendar', 'Exit_Sign', 'Kettle', 'Notebook', 'Refrigerator', 'Table', 'Candles', 'Fan', 'Keyboard', 'Oven', 'Ruler', 'Telephone'])
        
        


class PACS(DomainDataset):
    def __init__(self, args, model):
        super().__init__(args, model)

    def get_dataloader(self, client):
        pass


def get_data(dataset) -> DomainDataset:
    datalist = {"officehome": Officehome, "domainnet":DoaminNet}
    return datalist[dataset]
