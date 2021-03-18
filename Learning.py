import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import normalize
from torch.autograd import Variable
from torchvision import transforms

import Utils as K
from Net import KNet
from LabeledImageReader import LabeledImageReader
from Sampler import BalanceSampler

PHASE = ['training', 'validation']
RGBmean, RGBstd = [0.429, 0.495, 0.259], [0.218, 0.224, 0.171]


class Learning:
    def __init__(self, src, dst, data_dict):
        self.src = src
        self.dst = dst
        self.data_dict = data_dict
        self.gpu_id = [0, 1]  # number of gpu cores
        self.mp = False  # True for gpu parallel computing

        self.batch_size = 20
        self.num_workers = 20

        self.init_lr = 0.001
        self.decay_time = [False, False]
        self.decay_rate = 0.1

        self.num_features = 11
        self.criterion = nn.CrossEntropyLoss()
        self.record = {p: [] for p in PHASE}

    def run(self, num_epochs):
        if not self.setSystem():
            return
        self.num_epochs = num_epochs
        self.loadData()
        self.setModel()
        self.train(num_epochs)

    def setSystem(self):
        if not os.path.exists(self.src):
            print('src directory does not exist')
            return False
        if torch.cuda.is_available():
            dev = "cuda"
            torch.cuda.set_device(0)
        else:
            dev = "cpu"

        self.device = torch.device(dev)
        print('Current device is: ' + str(self.device))
        if not os.path.exists(self.dst):
            os.makedirs(self.dst)
        return True

    def loadData(self):
        data_transforms = {PHASE[0]: transforms.Compose([
            transforms.Resize(224 * 4),
            transforms.RandomCrop(224 * 3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(RGBmean, RGBstd)]),
            PHASE[1]: transforms.Compose([
                transforms.Resize(224 * 4),
                transforms.CenterCrop(224 * 3),
                transforms.ToTensor(),
                transforms.Normalize(RGBmean, RGBstd)])}

        self.dsets = {p: LabeledImageReader(self.data_dict[p], data_transforms[p]) for p in PHASE}
        self.intervals = self.dsets[PHASE[0]].intervals
        self.classSize = len(self.intervals)

    def setModel(self):
        Kmodel = KNet(self.num_features, self.classSize)

        if self.mp:
            print('Training on GPU')
            self.batch_size = self.batch_size * len(self.gpu_id)
            self.model = torch.nn.DataParallel(Kmodel, device_ids=self.gpu_id).cuda()
        else:
            print('Training on CPU')
            self.model = Kmodel.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        return

    def lr_scheduler(self, epoch):
        if epoch >= 0.6 * self.num_epochs and not self.decay_time[0]:
            self.decay_time[0] = True
            lr = self.init_lr * self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        if epoch >= 0.9 * self.num_epochs and not self.decay_time[1]:
            self.decay_time[1] = True
            lr = self.init_lr * self.decay_rate * self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return

    def train(self, num_epochs):
        starting_time = time.time()
        self.best_accuracy = 0.0
        self.best_epoch = 0

        for epoch in range(num_epochs):
            print('Epoch {}/{} \n'.format(epoch+1, num_epochs) + '-' * 40)
            for phase in PHASE:
                accuracyMat = np.zeros((self.classSize, self.classSize))
                running_loss = 0.0
                N_A = 0

                # Adjust the model for different phase
                if phase == PHASE[0]:
                    dataLoader = torch.utils.data.DataLoader(self.dsets[phase], batch_size=self.batch_size,
                                                             sampler=BalanceSampler(self.intervals, GSize=1),
                                                             num_workers=self.num_workers)
                    self.lr_scheduler(epoch)
                    if self.mp:
                        self.model.module.train(True)  # Set model to training mode
                        if epoch < int(num_epochs * 0.3):
                            self.model.module.R.d_rate(0.2)
                        elif int(num_epochs * 0.3) <= epoch < int(num_epochs * 0.6):
                            self.model.module.R.d_rate(0.1)
                        elif int(num_epochs * 0.6) <= epoch < int(num_epochs * 0.8):
                            self.model.module.R.d_rate(0.05)
                        elif epoch >= int(num_epochs * 0.8):
                            self.model.module.R.d_rate(0)
                    else:
                        self.model.train(True)
                        if epoch < int(num_epochs * 0.3):
                            print('тут')
                            self.model.R.d_rate(0.1)
                        elif int(num_epochs * 0.3) <= epoch < int(num_epochs * 0.6):
                            self.model.R.d_rate(0.1)
                        elif int(num_epochs * 0.6) <= epoch < int(num_epochs * 0.8):
                            self.model.R.d_rate(0.05)
                        elif epoch >= int(num_epochs * 0.8):
                            self.model.R.d_rate(0)
                else:
                    dataLoader = torch.utils.data.DataLoader(self.dsets[phase], batch_size=self.batch_size,
                                                             shuffle=False, num_workers=self.num_workers)

                    if self.mp:
                        self.model.module.train(False)  # Set model to evaluate mode
                        self.model.module.R.d_rate(0)
                    else:
                        self.model.train(False)
                        self.model.R.d_rate(0)

                # iterate batch
                for data in dataLoader:
                    inputs_batch, labels_batch = data
                    self.optimizer.zero_grad()

                    if torch.cuda.is_available():
                        outputs = self.model(Variable(inputs_batch.cuda()))
                    else:
                        outputs = self.model(Variable(inputs_batch))

                    _, preds_bt = torch.max(outputs.data, 1)
                    preds_bt = preds_bt.cpu().view(-1)

                    if torch.cuda.is_available():
                        loss = self.criterion(outputs, Variable(labels_batch.cuda()))
                    else:
                        loss = self.criterion(outputs, Variable(labels_batch))

                    if phase == PHASE[0]:
                        loss.backward()
                        self.optimizer.step()

                    if loss.data.dim != 0:
                        running_loss += loss.data.item()
                    N_A += len(labels_batch)
                    for i in range(len(labels_batch)):
                        accuracyMat[labels_batch[i], preds_bt[i]] += 1

                normedAccuracyMat = normalize(accuracyMat.astype(np.float64), axis=1, norm='l1')
                K.matrixPlot(normedAccuracyMat, self.dst + 'epoch/', phase + str(epoch))
                epoch_accuracy = np.trace(normedAccuracyMat)
                epoch_loss = running_loss / N_A

                print(accuracyMat)
                print(normedAccuracyMat)

                self.record[phase].append((epoch, epoch_loss, epoch_accuracy))

                if type(epoch_loss) != float:
                    epoch_loss = epoch_loss[0]
                print('{:5}:\n Loss: {:.4f}. Accuracy: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))

                if phase == PHASE[1]:
                    if epoch_accuracy > self.best_accuracy:
                        self.best_accuracy = epoch_accuracy
                        self.best_epoch = epoch
                        self.best_model = copy.deepcopy(self.model)
                        torch.save(self.best_model, self.dst + 'model.pth')

        time_elapsed = time.time() - starting_time
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f} in epoch: {}'.format(self.best_accuracy, self.best_epoch))
        torch.save(self.record, self.dst + str(self.best_epoch) + 'record.pth')
        return
