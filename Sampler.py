from torch.utils.data.sampler import Sampler
import numpy as np
import random
import torch


class BalanceSampler(Sampler):
    def __init__(self, intervals, GSize=2):

        class_len = len(intervals)
        list_sp = []

        # find the max interval
        interval_list = [torch.arange(b[0], b[1]).long() for b in intervals]
        len_max = max([b[1] - b[0] for b in intervals])

        if len_max > 3000: len_max = 3000

        # exact division
        if len_max % GSize != 0:
            len_max = len_max + (GSize - len_max % GSize)

        #print("лист интервалов")
        #print(interval_list)
        for l in interval_list:
            if l.shape[0] < len_max:
                #print(l.shape[0], len_max)
                l_ext = np.random.choice(l, len_max - l.shape[0])
                l_ext = np.concatenate((l, l_ext), axis=0)
                l_ext = np.random.permutation(l_ext)
            elif l.shape[0] > len_max:
                l_ext = np.random.choice(l, len_max, replace=False)
                l_ext = np.random.permutation(l_ext)
            elif l.shape[0] == len_max:
                l_ext = np.random.permutation(l)

            list_sp.append(l_ext)

        random.shuffle(list_sp)
        self.idx = np.vstack(list_sp).reshape((GSize * class_len, -1)).T.reshape((1, -1)).flatten().tolist()
        print(self.idx)

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


class BalanceSampler2(Sampler):
    def __init__(self, intervals, GSize=2):
        class_len = len(intervals)
        list_sp = []

        # find the max interval
        interval_list = [torch.arange(b[0], b[1]).long() for b in intervals]
        len_max = max([b[1] - b[0] for b in intervals])

        if len_max > 3000: len_max = 3000

        list_sp = []
        for l in interval_list:
            len_l = l.size(0)
            list_l = l.tolist()
            if len_l > len_max:
                list_l = random.sample(list_l, len_max)
            # exact division
            if len_l % GSize != 0:
                len_l = len_l + (GSize - len_l % GSize)
                l_ext = random.choices(list_l, k=len_l)
                l_ext = torch.LongTensor(l_ext)
                l_ext = l_ext.view(-1, 2).tolist()
            else:
                l_ext = list_l
                random.shuffle(l_ext)
                l_ext = torch.LongTensor(l_ext)
                l_ext = l_ext.view(-1, 2).tolist()

            list_sp += l_ext

        random.shuffle(list_sp)

        self.idx = torch.LongTensor(list_sp).view(-1).tolist()

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)
