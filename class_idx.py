import re
import numpy as np


class class_idx():
    def __init__(self, opt, source_file='./data/train_a.txt'):
        self.opt = opt
        self.pattern = '.*?\t.*?\t.*?\t.*?\t.*?\t(.*?)\t(.*?)\t(.*?)\n'
        with open(source_file) as f:
            data = f.read()
            data = re.findall(self.pattern, data)[1:]
            self.class_1, self.class_2, self.class_3 = [item[0] for item in data], [
                item[1] for item in data], [item[2] for item in data]
            self.class_1, self.class_2, self.class_3 = list(
                set(self.class_1)), list(set(self.class_2)), list(set(self.class_3))
            self.class_1.sort()
            self.class_2.sort()
            self.class_3.sort()
            self.class1_to_idx = dict(
                zip(self.class_1, [i for i in range(self.opt.NUM_CLASS_1)]))
            self.class2_to_idx = dict(
                zip(self.class_2, [i for i in range(self.opt.NUM_CLASS_2)]))
            self.class3_to_idx = dict(
                zip(self.class_3, [i for i in range(self.opt.NUM_CLASS_3)]))
            self.idx_to_class1 = dict(
                zip([i for i in range(10)], self.class_1))
            self.idx_to_class2 = dict(
                zip([i for i in range(64)], self.class_2))
            self.idx_to_class3 = dict(
                zip([i for i in range(125)], self.class_3))

    def class_to_idx(self, inputs):
        outputs = [self.class1_to_idx[inputs[0]],
                   self.class2_to_idx[inputs[1]], self.class3_to_idx[inputs[2]]]
        outputs = np.array(outputs)
        return outputs

    def idx_to_class(self, inputs):
        outputs = []
        outputs = [self.idx_to_class1[inputs[0]],
                   self.idx_to_class2[inputs[1]], self.idx_to_class3[inputs[2]]]
        return outputs

