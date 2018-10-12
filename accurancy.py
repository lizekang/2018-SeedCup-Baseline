import torch
import numpy as np


class F1():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.output = np.zeros(self.num_classes)
        self.target = np.zeros(self.num_classes)
        self.TT = np.zeros(self.num_classes)
        self.precision = np.zeros(self.num_classes)
        self.recall = np.zeros(self.num_classes)
        self.answer = np.zeros(self.num_classes)

    def save_data(self, outputs, targets):
        outputs = torch.zeros(outputs.size()).scatter_(
            1, (torch.max(outputs, dim=1)[1]).cpu().view(-1, 1), 1.)
        targets = torch.zeros(outputs.size()).scatter_(
            1, targets.cpu().view(-1, 1), 1.)
        outputs, targets = outputs.int().numpy(), targets.int().numpy()
        self.output += outputs.sum(axis=0)
        self.target += targets.sum(axis=0)
        self.TT += (outputs*targets).sum(axis=0)

    def caculate_f1(self):
        for i in range(self.num_classes):
            if self.output[i] != 0:
                self.precision[i] = self.TT[i]/self.output[i]
            if self.target[i] != 0:
                self.recall[i] = self.TT[i]/self.target[i]
            if (self.precision[i]+self.recall[i]) != 0:
                self.answer[i] = self.precision[i]*self.recall[i] * \
                    2/(self.precision[i]+self.recall[i])
        return np.mean(self.answer)
