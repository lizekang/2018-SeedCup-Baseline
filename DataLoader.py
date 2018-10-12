import re
import torch
import numpy as np
from word_idx import word_idx
from class_idx import class_idx
from config import Config
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, source_file, word_idx, class_idx):
        self.word_idx = word_idx
        self.class_idx = class_idx
        self.pattern = '.*?\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\n'
        with open(source_file, 'r') as f:
            data = f.read()
            data = re.findall(self.pattern, data)[1:]
            self.inputs = [(data_item[1]+','+data_item[3]).split(',')
                           for data_item in data]
            self.targets = [[data_item[4], data_item[5], data_item[6]]
                            for data_item in data]

    def __getitem__(self, idx):
        inputs = self.word_idx.word_to_idx(self.inputs[idx])
        targets = self.class_idx.class_to_idx(inputs=self.targets[idx])
        return (inputs, targets)

    def __len__(self):
        return len(self.inputs)


class MyTestSet(Dataset):
    def __init__(self, word_idx, class_idx, test_file='./data/test_a.txt'):
        self.word_idx = word_idx
        self.class_idx = class_idx
        self.pattern = '(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t\t\t\n'
        with open(test_file, 'r') as f:
            data = f.read()
            data = re.findall(self.pattern, data)
            self.inputs = [(data_item[2]+','+data_item[4]).split(',')
                           for data_item in data]
            self.ids = [data_item[0] for data_item in data]

    def __getitem__(self, idx):
        inputs = self.word_idx.word_to_idx(self.inputs[idx])
        ids = self.ids[idx]
        return (inputs, ids)

    def __len__(self):
        return len(self.inputs)
