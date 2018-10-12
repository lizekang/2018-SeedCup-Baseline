import re
import numpy as np


class word_idx():
    def __init__(self, opt, source_file='./data/train_a.txt'):
        self.opt = opt
        self.pattern = '.*?\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\n'
        with open(source_file) as f:
            data = f.read()
            data = re.findall(self.pattern, data)[1:]
            title_words = [data_item[1] for data_item in data]
            describe_words = [data_item[3] for data_item in data]
            words = []
            for i in range(len(data)):
                words += title_words[i].split(',')+describe_words[i].split(',')
            self.words = set(words)
            words = list(set(words))
            words.sort()
            self.word_to_idx_dict = {word: i+1 for i, word in enumerate(words)}

    def word_to_idx(self, inputs):
        outputs = []
        for item in inputs:
            if item in self.words:
                outputs.append(self.word_to_idx_dict[item])
        if(len(outputs) < self.opt.SENT_LEN):
            outputs += [0]*(self.opt.SENT_LEN-len(outputs))
        else:
            outputs = outputs[:self.opt.SENT_LEN]
        outputs = np.array(outputs, dtype=int)
        return outputs


def test():
    word_with_idx = word_idx()
    print(word_with_idx.word_to_idx(['w1', 'wa']))
# test()
