import sys
import os.path as osp
import os
import logging

base_path = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(base_path)

import clip
import torchvision.datasets as datasets
from .folder import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split

from PIL import ImageFile
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from functools import reduce
import pathlib
SELECTED_CLASSES = [ "tree","golf_club","squirrel","dog","whale","spreadsheet","snowman","tiger","table","shoe","windmill","submarine","truck","feather","bird","spider","strawberry","nail","beard","bread","train","watermelon","zebra","sheep","elephant","teapot","eye","mushroom","sea_turtle","sword","streetlight","lighthouse","bridge","owl","horse","penguin","pond","sock","snorkel","helicopter","snake","butterfly","umbrella","river","fish","van","grapes","hot_air_balloon","wine_glass","teddy-bear","speedboat","sun","swan","bicycle","brain","bracelet","tornado","flower","stairs","cup","steak","vase","tractor","wristwatch","stethoscope","suitcase","triangle","parrot","zigzag","ice_cream","mug","beach","cat","raccoon","garden","monkey","shark","animal_migration","lion","saxophone","asparagus","tent","firetruck","The_Eiffel_Tower","hand","spoon","squiggle","palm_tree","octopus","toaster","skateboard","dumbbell","headphones","mountain","bottlecap","hexagon","pig","toilet","washing_machine","frog"]
ALL_CLASSES = ['aircraft_carrier', 'chandelier', 'harp', 'palm_tree', 'spider', 'airplane', 'church', 'hat', 'panda', 'spoon', 'alarm_clock', 'circle', 'headphones', 'pants', 'spreadsheet', 'ambulance', 'clarinet', 'hedgehog', 'paper_clip', 'square', 'angel', 'clock', 'helicopter', 'parachute', 'squiggle', 'animal_migration', 'cloud', 'helmet', 'parrot', 'squirrel', 'ant', 'coffee_cup', 'hexagon', 'passport', 'stairs', 'anvil', 'compass', 'hockey_puck', 'peanut', 'star', 'apple', 'computer', 'hockey_stick', 'pear', 'steak', 'arm', 'cookie', 'horse', 'peas', 'stereo', 'asparagus', 'cooler', 'hospital', 'pencil', 'stethoscope', 'axe', 'couch', 'hot_air_balloon', 'penguin', 'stitches', 'backpack', 'cow', 'hot_dog', 'piano', 'stop_sign', 'banana', 'crab', 'hot_tub', 'pickup_truck', 'stove', 'bandage', 'crayon', 'hourglass', 'picture_frame', 'strawberry', 'barn', 'crocodile', 'house', 'pig', 'streetlight', 'baseball', 'crown', 'house_plant', 'pillow', 'string_bean', 'baseball_bat', 'cruise_ship', 'hurricane', 'pineapple', 'submarine', 'basket', 'cup', 'ice_cream', 'pizza', 'suitcase', 'basketball', 'diamond', 'jacket', 'pliers', 'sun', 'bat', 'dishwasher', 'jail', 'police_car', 'swan', 'bathtub', 'diving_board', 'kangaroo', 'pond', 'sweater', 'beach', 'dog', 'key', 'pool', 'swing_set', 'bear', 'dolphin', 'keyboard', 'popsicle', 'sword', 'beard', 'donut', 'knee', 'postcard', 'syringe', 'bed', 'door', 'knife', 'potato', 'table', 'bee', 'dragon', 'ladder', 'power_outlet', 'teapot', 'belt', 'dresser', 'lantern', 'purse', 'teddy-bear', 'bench', 'drill', 'laptop', 'rabbit', 'telephone', 'bicycle', 'drums', 'leaf', 'raccoon', 'television', 'binoculars', 'duck', 'leg', 'radio', 'tennis_racquet', 'bird', 'dumbbell', 'light_bulb', 'rain', 'tent', 'birthday_cake', 'ear', 'lighter', 'rainbow', 'The_Eiffel_Tower', 'blackberry', 'elbow', 'lighthouse', 'rake', 'The_Great_Wall_of_China', 'blueberry', 'elephant', 'lightning', 'remote_control', 'The_Mona_Lisa', 'book', 'envelope', 'line', 'rhinoceros', 'tiger', 'boomerang', 'eraser', 'lion', 'rifle', 'toaster', 'bottlecap', 'eye', 'lipstick', 'river', 'toe', 'bowtie', 'eyeglasses', 'lobster', 'roller_coaster', 'toilet', 'bracelet', 'face', 'lollipop', 'rollerskates', 'tooth', 'brain', 'fan', 'mailbox', 'sailboat', 'toothbrush', 'bread', 'feather', 'map', 'sandwich', 'toothpaste', 'bridge', 'fence', 'marker', 'saw', 'tornado', 'broccoli', 'finger', 'matches', 'saxophone', 'tractor', 'broom', 'fire_hydrant', 'megaphone', 'school_bus', 'traffic_light', 'bucket', 'fireplace', 'mermaid', 'scissors', 'train', 'bulldozer', 'firetruck', 'microphone', 'scorpion', 'tree', 'bus', 'fish', 'microwave', 'screwdriver', 'triangle', 'bush', 'flamingo', 'monkey', 'sea_turtle', 'trombone', 'butterfly', 'flashlight', 'moon', 'see_saw', 'truck', 'cactus', 'flip_flops', 'mosquito', 'shark', 'trumpet', 'cake', 'floor_lamp', 'motorbike', 'sheep', 't-shirt', 'calculator', 'flower', 'mountain', 'shoe', 'umbrella', 'calendar', 'flying_saucer', 'mouse', 'shorts', 'underwear', 'camel', 'foot', 'moustache', 'shovel', 'van', 'camera', 'fork', 'mouth', 'sink', 'vase', 'camouflage', 'frog', 'mug', 'skateboard', 'violin', 'campfire', 'frying_pan', 'mushroom', 'skull', 'washing_machine', 'candle', 'garden', 'nail', 'skyscraper', 'watermelon', 'cannon', 'garden_hose', 'necklace', 'sleeping_bag', 'waterslide', 'canoe', 'giraffe', 'nose', 'smiley_face', 'whale', 'car', 'goatee', 'ocean', 'snail', 'wheel', 'carrot', 'golf_club', 'octagon', 'snake', 'windmill', 'castle', 'grapes', 'octopus', 'snorkel', 'wine_bottle', 'cat', 'grass', 'onion', 'snowflake', 'wine_glass', 'ceiling_fan', 'guitar', 'oven', 'snowman', 'wristwatch', 'cello', 'hamburger', 'owl', 'soccer_ball', 'yoga', 'cell_phone', 'hammer', 'paintbrush', 'sock', 'zebra', 'chair', 'hand', 'paint_can', 'speedboat', 'zigzag']


ImageFile.LOAD_TRUNCATED_IMAGES = True



class ImageTextData(object):
    def __init__(self, root, client_name, task_id, preprocess, prompt="a picture of a"):
        if (client_name == "test"):
            client_data_path = osp.join(root, client_name, task_id)
        else:
            client_data_path = osp.join(root, client_name, f"task{task_id}")
        if len(os.listdir(client_data_path)) > 1:
            logging.info(f"loading data in {client_data_path} with {os.listdir(osp.join(root, client_name))}")
            data = ImageFolder(client_data_path ,transform=ImageTextData._TRANSFORM, allow_empty=True)
        else:
            logging.info(f"loading data in {osp.join(client_data_path, os.listdir(client_data_path)[0])}")
            data = ImageFolder(
                osp.join(client_data_path, os.listdir(client_data_path)[0]),
                allow_empty=True,
            )
        self.data = data
        self.labels = self.data.classes
        

        
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
        # self.train_percentage = args.datapercent
        self.set_labels()


    def get_dataloader(self, task_id):
        train_datasets, test_datasets = [], []
        train_dataloaders, test_dataloaders = [], []
        
        for _, client_name in enumerate(sorted(os.listdir(self.root_dir))):
            if client_name == 'test':
                for domain in os.listdir(osp.join(self.root_dir, client_name)):
                    test_datasets.append(ImageTextData(self.root_dir, client_name, domain, self.preprocess))
                    test_dataloaders.append(DataLoader(test_datasets[-1], batch_size=self.batch_size, shuffle=True))
                    test_dataloaders[-1].domain = domain
                continue
            data = ImageTextData(self.root_dir, client_name, task_id, self.preprocess)
            l = len(data)
            index = np.arange(l)
            np.random.seed(self.random_seed)
            np.random.shuffle(index)
            if ((self.frac != 1) and (client_name != "test")):
                split_size = int(self.frac * len(data))
                temp = Subset(data, index[:split_size])
                train_datasets.append(temp)
            else:
                train_datasets.append(data)
            # l1, l2 = int(l * self.train_percentage), int(l * (1 - self.train_percentage))
            # train_datasets.append(Subset(data, index[:l1]))
            # test_datasets.append(Subset(data, index[l1 : l1 + l2 + 1]))
            train_dataloaders.append(
                DataLoader(
                    train_datasets[-1], batch_size=self.batch_size, shuffle=True))
        
        return train_dataloaders, test_dataloaders, self.labels
    
    def set_labels():
        raise NotImplementedError
    

class DoaminNet(DomainDataset): 
    def __init__(self, args, preprocess) -> None:
        super().__init__(args, preprocess)
          
    def set_labels(self):
        self.labels = sorted(ALL_CLASSES)
        # self.labels = sorted(['aircraft_carrier', 'chandelier', 'harp', 'palm_tree', 'spider', 'airplane', 'church', 'hat', 'panda', 'spoon', 'alarm_clock', 'circle', 'headphones', 'pants', 'spreadsheet', 'ambulance', 'clarinet', 'hedgehog', 'paper_clip', 'square', 'angel', 'clock', 'helicopter', 'parachute', 'squiggle', 'animal_migration', 'cloud', 'helmet', 'parrot', 'squirrel', 'ant', 'coffee_cup', 'hexagon', 'passport', 'stairs', 'anvil', 'compass', 'hockey_puck', 'peanut', 'star', 'apple', 'computer', 'hockey_stick', 'pear', 'steak', 'arm', 'cookie', 'horse', 'peas', 'stereo', 'asparagus', 'cooler', 'hospital', 'pencil', 'stethoscope', 'axe', 'couch', 'hot_air_balloon', 'penguin', 'stitches', 'backpack', 'cow', 'hot_dog', 'piano', 'stop_sign', 'banana', 'crab', 'hot_tub', 'pickup_truck', 'stove', 'bandage', 'crayon', 'hourglass', 'picture_frame', 'strawberry', 'barn', 'crocodile', 'house', 'pig', 'streetlight', 'baseball', 'crown', 'house_plant', 'pillow', 'string_bean', 'baseball_bat', 'cruise_ship', 'hurricane', 'pineapple', 'submarine', 'basket', 'cup', 'ice_cream', 'pizza', 'suitcase', 'basketball', 'diamond', 'jacket', 'pliers', 'sun', 'bat', 'dishwasher', 'jail', 'police_car', 'swan', 'bathtub', 'diving_board', 'kangaroo', 'pond', 'sweater', 'beach', 'dog', 'key', 'pool', 'swing_set', 'bear', 'dolphin', 'keyboard', 'popsicle', 'sword', 'beard', 'donut', 'knee', 'postcard', 'syringe', 'bed', 'door', 'knife', 'potato', 'table', 'bee', 'dragon', 'ladder', 'power_outlet', 'teapot', 'belt', 'dresser', 'lantern', 'purse', 'teddy-bear', 'bench', 'drill', 'laptop', 'rabbit', 'telephone', 'bicycle', 'drums', 'leaf', 'raccoon', 'television', 'binoculars', 'duck', 'leg', 'radio', 'tennis_racquet', 'bird', 'dumbbell', 'light_bulb', 'rain', 'tent', 'birthday_cake', 'ear', 'lighter', 'rainbow', 'The_Eiffel_Tower', 'blackberry', 'elbow', 'lighthouse', 'rake', 'The_Great_Wall_of_China', 'blueberry', 'elephant', 'lightning', 'remote_control', 'The_Mona_Lisa', 'book', 'envelope', 'line', 'rhinoceros', 'tiger', 'boomerang', 'eraser', 'lion', 'rifle', 'toaster', 'bottlecap', 'eye', 'lipstick', 'river', 'toe', 'bowtie', 'eyeglasses', 'lobster', 'roller_coaster', 'toilet', 'bracelet', 'face', 'lollipop', 'rollerskates', 'tooth', 'brain', 'fan', 'mailbox', 'sailboat', 'toothbrush', 'bread', 'feather', 'map', 'sandwich', 'toothpaste', 'bridge', 'fence', 'marker', 'saw', 'tornado', 'broccoli', 'finger', 'matches', 'saxophone', 'tractor', 'broom', 'fire_hydrant', 'megaphone', 'school_bus', 'traffic_light', 'bucket', 'fireplace', 'mermaid', 'scissors', 'train', 'bulldozer', 'firetruck', 'microphone', 'scorpion', 'tree', 'bus', 'fish', 'microwave', 'screwdriver', 'triangle', 'bush', 'flamingo', 'monkey', 'sea_turtle', 'trombone', 'butterfly', 'flashlight', 'moon', 'see_saw', 'truck', 'cactus', 'flip_flops', 'mosquito', 'shark', 'trumpet', 'cake', 'floor_lamp', 'motorbike', 'sheep', 't-shirt', 'calculator', 'flower', 'mountain', 'shoe', 'umbrella', 'calendar', 'flying_saucer', 'mouse', 'shorts', 'underwear', 'camel', 'foot', 'moustache', 'shovel', 'van', 'camera', 'fork', 'mouth', 'sink', 'vase', 'camouflage', 'frog', 'mug', 'skateboard', 'violin', 'campfire', 'frying_pan', 'mushroom', 'skull', 'washing_machine', 'candle', 'garden', 'nail', 'skyscraper', 'watermelon', 'cannon', 'garden_hose', 'necklace', 'sleeping_bag', 'waterslide', 'canoe', 'giraffe', 'nose', 'smiley_face', 'whale', 'car', 'goatee', 'ocean', 'snail', 'wheel', 'carrot', 'golf_club', 'octagon', 'snake', 'windmill', 'castle', 'grapes', 'octopus', 'snorkel', 'wine_bottle', 'cat', 'grass', 'onion', 'snowflake', 'wine_glass', 'ceiling_fan', 'guitar', 'oven', 'snowman', 'wristwatch', 'cello', 'hamburger', 'owl', 'soccer_ball', 'yoga', 'cell_phone', 'hammer', 'paintbrush', 'sock', 'zebra', 'chair', 'hand', 'paint_can', 'speedboat', 'zigzag'])

class DoaminNetSub(DomainDataset): 
    def __init__(self, args, preprocess) -> None:
        self.frac = 0.25
        super().__init__(args, preprocess)
          
    def set_labels(self):
        self.labels = sorted(SELECTED_CLASSES)
        # self.labels = sorted(['aircraft_carrier', 'chandelier', 'harp', 'palm_tree', 'spider', 'airplane', 'church', 'hat', 'panda', 'spoon', 'alarm_clock', 'circle', 'headphones', 'pants', 'spreadsheet', 'ambulance', 'clarinet', 'hedgehog', 'paper_clip', 'square', 'angel', 'clock', 'helicopter', 'parachute', 'squiggle', 'animal_migration', 'cloud', 'helmet', 'parrot', 'squirrel', 'ant', 'coffee_cup', 'hexagon', 'passport', 'stairs', 'anvil', 'compass', 'hockey_puck', 'peanut', 'star', 'apple', 'computer', 'hockey_stick', 'pear', 'steak', 'arm', 'cookie', 'horse', 'peas', 'stereo', 'asparagus', 'cooler', 'hospital', 'pencil', 'stethoscope', 'axe', 'couch', 'hot_air_balloon', 'penguin', 'stitches', 'backpack', 'cow', 'hot_dog', 'piano', 'stop_sign', 'banana', 'crab', 'hot_tub', 'pickup_truck', 'stove', 'bandage', 'crayon', 'hourglass', 'picture_frame', 'strawberry', 'barn', 'crocodile', 'house', 'pig', 'streetlight', 'baseball', 'crown', 'house_plant', 'pillow', 'string_bean', 'baseball_bat', 'cruise_ship', 'hurricane', 'pineapple', 'submarine', 'basket', 'cup', 'ice_cream', 'pizza', 'suitcase', 'basketball', 'diamond', 'jacket', 'pliers', 'sun', 'bat', 'dishwasher', 'jail', 'police_car', 'swan', 'bathtub', 'diving_board', 'kangaroo', 'pond', 'sweater', 'beach', 'dog', 'key', 'pool', 'swing_set', 'bear', 'dolphin', 'keyboard', 'popsicle', 'sword', 'beard', 'donut', 'knee', 'postcard', 'syringe', 'bed', 'door', 'knife', 'potato', 'table', 'bee', 'dragon', 'ladder', 'power_outlet', 'teapot', 'belt', 'dresser', 'lantern', 'purse', 'teddy-bear', 'bench', 'drill', 'laptop', 'rabbit', 'telephone', 'bicycle', 'drums', 'leaf', 'raccoon', 'television', 'binoculars', 'duck', 'leg', 'radio', 'tennis_racquet', 'bird', 'dumbbell', 'light_bulb', 'rain', 'tent', 'birthday_cake', 'ear', 'lighter', 'rainbow', 'The_Eiffel_Tower', 'blackberry', 'elbow', 'lighthouse', 'rake', 'The_Great_Wall_of_China', 'blueberry', 'elephant', 'lightning', 'remote_control', 'The_Mona_Lisa', 'book', 'envelope', 'line', 'rhinoceros', 'tiger', 'boomerang', 'eraser', 'lion', 'rifle', 'toaster', 'bottlecap', 'eye', 'lipstick', 'river', 'toe', 'bowtie', 'eyeglasses', 'lobster', 'roller_coaster', 'toilet', 'bracelet', 'face', 'lollipop', 'rollerskates', 'tooth', 'brain', 'fan', 'mailbox', 'sailboat', 'toothbrush', 'bread', 'feather', 'map', 'sandwich', 'toothpaste', 'bridge', 'fence', 'marker', 'saw', 'tornado', 'broccoli', 'finger', 'matches', 'saxophone', 'tractor', 'broom', 'fire_hydrant', 'megaphone', 'school_bus', 'traffic_light', 'bucket', 'fireplace', 'mermaid', 'scissors', 'train', 'bulldozer', 'firetruck', 'microphone', 'scorpion', 'tree', 'bus', 'fish', 'microwave', 'screwdriver', 'triangle', 'bush', 'flamingo', 'monkey', 'sea_turtle', 'trombone', 'butterfly', 'flashlight', 'moon', 'see_saw', 'truck', 'cactus', 'flip_flops', 'mosquito', 'shark', 'trumpet', 'cake', 'floor_lamp', 'motorbike', 'sheep', 't-shirt', 'calculator', 'flower', 'mountain', 'shoe', 'umbrella', 'calendar', 'flying_saucer', 'mouse', 'shorts', 'underwear', 'camel', 'foot', 'moustache', 'shovel', 'van', 'camera', 'fork', 'mouth', 'sink', 'vase', 'camouflage', 'frog', 'mug', 'skateboard', 'violin', 'campfire', 'frying_pan', 'mushroom', 'skull', 'washing_machine', 'candle', 'garden', 'nail', 'skyscraper', 'watermelon', 'cannon', 'garden_hose', 'necklace', 'sleeping_bag', 'waterslide', 'canoe', 'giraffe', 'nose', 'smiley_face', 'whale', 'car', 'goatee', 'ocean', 'snail', 'wheel', 'carrot', 'golf_club', 'octagon', 'snake', 'windmill', 'castle', 'grapes', 'octopus', 'snorkel', 'wine_bottle', 'cat', 'grass', 'onion', 'snowflake', 'wine_glass', 'ceiling_fan', 'guitar', 'oven', 'snowman', 'wristwatch', 'cello', 'hamburger', 'owl', 'soccer_ball', 'yoga', 'cell_phone', 'hammer', 'paintbrush', 'sock', 'zebra', 'chair', 'hand', 'paint_can', 'speedboat', 'zigzag'])

class Adaptiope(DomainDataset):
    def __init__(self, args, preprocess) -> None:
        self.frac = 1
        super().__init__(args, preprocess)
    
    def set_labels(self):
        self.labels = sorted(['dart', 'laptop', 'umbrella', 'cellphone', 'magic lamp', 'compass', 'ice skates', 'file cabinet', 'wheelchair', 'nail clipper', 'telescope', 'desk lamp', 'bicycle', 'rc car', 'chainsaw', 'hoverboard', 'speakers', 'trash can', 'smoking pipe', 'stethoscope', 'bookcase', 'hot glue gun', 'skeleton', 'baseball bat', 'boxing gloves', 'in-ear headphones', 'hair dryer', 'pogo stick', 'rubber boat', 'binoculars', 'razor', 'glasses', 'keyboard', 'power drill', 'handcuffs', 'toilet brush', 'projector', 'mug', 'sword', 'screwdriver', 'knife', 'quadcopter', 'tape dispenser', 'vr goggles', 'computer', 'usb stick', 'vacuum cleaner', 'hand mixer', 'comb', 'rifle', 'stand mixer', 'ladder', 'letter tray', 'fighter jet', 'stroller', 'sewing machine', 'puncher', 'corkscrew', 'pen', 'ruler', 'electric guitar', 'fire extinguisher', 'scooter', 'skateboard', 'acoustic guitar', 'power strip', 'hard-wired fixed phone', 'motorbike helmet', 'fan', 'syringe', 'drum set', 'over-ear headphones', 'handgun', 'golf club', 'webcam', 'grill', 'monitor', 'wallet', 'ice cube tray', 'tyrannosaurus', 'purse', 'pipe wrench', 'watering can', 'printer', 'bottle', 'car jack', 'spatula', 'coat hanger', 'diving fins', 'game controller', 'computer mouse', 'calculator', 'electric shaver', 'pikachu', 'bicycle helmet', 'axe', 'wristwatch', 'ring binder', 'hat', 'tank', 'brachiosaurus', 'crown', 'microwave', 'shower head', 'notepad', 'cordless fixed phone', 'flat iron', 'roller skates', 'mixing console', 'tent', 'smartphone', 'toothbrush', 'scissors', 'stapler', 'backpack', 'sleeping bag', 'office chair', 'hourglass', 'phonograph', 'network switch', 'lawn mower', 'helicopter', 'snow shovel'])

class Officehome(DomainDataset):
    def __init__(self, args, preprocess) -> None:
        self.frac = 0.5
        super().__init__(args, preprocess)
        
    def set_labels(self):
        self.labels = sorted(['Alarm_Clock', 'Chair', 'File_Cabinet', 'Knives', 'Pan', 'Scissors', 'ToothBrush', 'Backpack', 'Clipboards', 'Flipflops', 'Lamp_Shade', 'Paper_Clip', 'Screwdriver', 'Toys', 'Batteries', 'Computer', 'Flowers', 'Laptop', 'Pen', 'Shelf', 'Trash_Can', 'Bed', 'Couch', 'Folder', 'Marker', 'Pencil', 'Sink', 'TV', 'Bike', 'Curtains', 'Fork', 'Monitor', 'Postit_Notes', 'Sneakers', 'Webcam', 'Bottle', 'Desk_Lamp', 'Glasses', 'Mop', 'Printer', 'Soda', 'Bucket', 'Drill', 'Hammer', 'Mouse', 'Push_Pin', 'Speaker', 'Calculator', 'Eraser', 'Helmet', 'Mug', 'Radio', 'Spoon', 'Calendar', 'Exit_Sign', 'Kettle', 'Notebook', 'Refrigerator', 'Table', 'Candles', 'Fan', 'Keyboard', 'Oven', 'Ruler', 'Telephone'])
        
        
class PACS(DomainDataset):
    def __init__(self, args, model):
        super().__init__(args, model)

    def get_dataloader(self, client):
        pass

class Cifar100(DomainDataset):
    def __init__(self, args, preprocess) -> None:
        self.frac = 1
        super().__init__(args, preprocess)
    def set_labels(self):
        self.labels = sorted(['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'])

class Miniimagenet(DomainDataset):
    def __init__(self, args, preprocess) -> None:
        self.frac = 1
        super().__init__(args, preprocess)
    def set_labels(self):
        self.labels = sorted(['African_hunting_dog', 'Arctic_fox', 'French_bulldog', 'Gordon_setter', 'Ibizan_hound', 'Newfoundland', 'Saluki', 'Tibetan_mastiff', 'Walker_hound', 'aircraft_carrier', 'ant', 'ashcan', 'barrel', 'beer_bottle', 'black-footed_ferret', 'bolete', 'bookshop', 'boxer', 'cannon', 'carousel', 'carton', 'catamaran', 'chime', 'cliff', 'clog', 'cocktail_shaker', 'combination_lock', 'consomme', 'coral_reef', 'crate', 'cuirass', 'dalmatian', 'dishrag', 'dome', 'dugong', 'ear', 'electric_guitar', 'file', 'fire_screen', 'frying_pan', 'garbage_truck', 'golden_retriever', 'goose', 'green_mamba', 'hair_slide', 'harvestman', 'holster', 'horizontal_bar', 'hotdog', 'hourglass', 'house_finch', 'iPod', 'jellyfish', 'king_crab', 'komondor', 'ladybug', 'lion', 'lipstick', 'malamute', 'meerkat', 'miniature_poodle', 'miniskirt', 'missile', 'mixing_bowl', 'nematode', 'oboe', 'orange', 'organ', 'parallel_bars', 'pencil_box', 'photocopier', 'poncho', 'prayer_rug', 'reel', 'rhinoceros_beetle', 'robin', 'rock_beauty', 'school_bus', 'scoreboard', 'slot', 'snorkel', 'solar_dish', 'spider_web', 'stage', 'street_sign', 'tank', 'theater_curtain', 'three-toed_sloth', 'tile_roof', 'tobacco_shop', 'toucan', 'triceratops', 'trifle', 'unicycle', 'upright', 'vase', 'white_wolf', 'wok', 'worm_fence', 'yawl'])


def get_data(dataset) -> DomainDataset:
    datalist = {"officehome": Officehome, "domainnet":DoaminNet, "domainnetsub":DoaminNetSub, "adaptiope":Adaptiope, "cifar100":Cifar100, "miniimagenet":Miniimagenet}
    return datalist[dataset]
