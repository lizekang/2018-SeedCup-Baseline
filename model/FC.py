import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from config import Config


class FC(nn.Module):
    def __init__(self, opt):
        super(FC, self).__init__()
        self.encoder = nn.Embedding(opt.VOCAB_SIZE, opt.EMBEDDING_DIM)
        self.fc_1 = nn.Sequential(
            nn.Linear(opt.SENT_LEN*opt.EMBEDDING_DIM, opt.LINER_HID_SIZE),
            nn.BatchNorm1d(opt.LINER_HID_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(opt.LINER_HID_SIZE, opt.NUM_CLASS_1)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(opt.SENT_LEN*opt.EMBEDDING_DIM, opt.LINER_HID_SIZE),
            nn.BatchNorm1d(opt.LINER_HID_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(opt.LINER_HID_SIZE, opt.NUM_CLASS_2)
        )
        self.fc_3 = nn.Sequential(
            nn.Linear(opt.SENT_LEN*opt.EMBEDDING_DIM, opt.LINER_HID_SIZE),
            nn.BatchNorm1d(opt.LINER_HID_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(opt.LINER_HID_SIZE, opt.NUM_CLASS_3)
        )

    def forward(self, x):
        outputs = self.encoder(x)
        outputs = outputs.view(outputs.size()[0], -1)
        output_1 = self.fc_1(outputs)
        output_2 = self.fc_2(outputs)
        output_3 = self.fc_3(outputs)
        return (output_1, output_2, output_3)

