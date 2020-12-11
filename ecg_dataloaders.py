import torchvision
import torch
from torch import nn
import numpy as np
from torchvision import transforms
import torch.utils.data as data
import os
from PIL import Image
import numpy as np

class sv_ve_nm_loader(data.Dataset):

    def __init__(self, datafile):
        self.specs = []
        fp = open(datafile, 'r')
        paths = fp.readlines()
        fp.close()
        for path in paths:
            basename = os.path.basename(path)
            
            label = 1
            if basename.startswith("nm"):
                label = 0
            #if basename.startswith("sv"):
            #    label = 1
            #elif basename.startswith('ve'):
            #    label = 2
            path = path.strip("\n")
            self.specs.append((path, label))
        self.preprocess = transforms.Compose([
                              transforms.Resize(256),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                          ])

    def __len__(self):
        return len(self.specs)


    def __getitem__(self, index):
        path, label = self.specs[index]
        img = Image.open(path).convert('RGB')
        img = self.preprocess(img)
        return img, label
    


class ar_binclass_loader(data.Dataset):

    def __init__(self,datadir):
        self.specs = []
        #normal_dir = normal
        #arrythmia_dir  = arrythmia

        for fname in os.listdir(datadir):
            label = 0 if fname.split('_')[0] == 'nm' else 1
            #print(datadir, fname)
            self.specs.append((os.path.join(datadir, fname), label))
            

        self.preprocess = transforms.Compose([
                              transforms.Resize(256),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                          ])

    def __len__(self):
        return len(self.specs)


    def __getitem__(self, index):
        path, label = self.specs[index]
        img = Image.open(path).convert('RGB')
        #print("IMG", img)
        img = self.preprocess(img)
        return img, label



