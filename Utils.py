import matplotlib
import numpy as np
import torch
import os
import random
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.utils import make_grid
from torchvision import transforms
from glob import glob
from sklearn.preprocessing import normalize

matplotlib.use('Agg')


def combDict1D(dict_main, dict_in):
    if dict_main is None:
        dict_main = dict_in
    else:
        for A in dict_in:
            if A in dict_main:
                dict_main[A] += dict_in[A]
            else:
                dict_main[A] = dict_in[A]
    return dict_main


def combDict2D(dict_main, dict_in):
    if dict_main == None:
        dict_main = dict_in
    else:
        for A in dict_in:
            if A in dict_main:
                for B in dict_in[A]:
                    if B in dict_main[A]:
                        dict_main[A][B] += dict_in[A][B]
                    else:
                        dict_main[A][B] = dict_in[A][B]
            else:
                dict_main[A] = dict_in[A]
    return dict_main


def combDict3D(dict_main, dict_in):
    if dict_main is None:
        dict_main = dict_in
    else:
        for A in dict_in:
            if A in dict_main:
                for B in dict_in[A]:
                    if B in dict_main[A]:
                        for C in dict_in[A][B]:
                            if C in dict_main[A][B]:
                                dict_main[A][B][C] += dict_in[A][B][C]
                            else:
                                dict_main[A][B][C] = dict_in[A][B][C]
                    else:
                        dict_main[A][B] = dict_in[A][B]
            else:
                dict_main[A] = dict_in[A]
    return dict_main


def invDict(dict_in):
    values = sorted(set([v for k, v in dict_in.items()]))
    dict_out = {v: [] for v in values}
    for k, v in dict_in.items(): dict_out[v].append(k)
    return dict_out


def matrixPlot(mat, dst, figname):
    mat = normalize(mat.astype(np.float64), axis=1, norm='l1')
    plt.figure()
    plt.imshow(mat, cmap='hot', vmin=0, vmax=1)
    plt.colorbar()
    plt.xticks(np.arange(0, mat.shape[0], 1))
    plt.xlabel('pre')
    plt.yticks(np.arange(0, mat.shape[0], 1))
    plt.ylabel('lab')
    plt.axes().xaxis.set_ticks_position('top')
    plt.title(figname, y=1.05)
    if not os.path.exists(dst): os.makedirs(dst)
    plt.savefig(dst + figname, dpi=600)
    plt.close("all")


def recordPlot(record, dst):
    if not os.path.exists(dst): os.makedirs(dst)
    PHASE = [p for p in record]

    data = {p: np.asarray(record[p]) for p in PHASE}  # [i, loss, acc]

    plt.figure()
    for phase in record: plt.plot(data[phase][:, 1], label='{:5} loss'.format(phase))
    plt.axis([0, len(data[PHASE[0]][:, 1]) - 1, 0, max([np.max(data[PHASE[0]][:, 1]), np.max(data[PHASE[1]][:, 1])])])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend(bbox_to_anchor=(0.95, 0.05), loc=4)
    plt.savefig(dst + 'loss.jpg', dpi=600)
    plt.close()

    plt.figure()
    for phase in record: plt.plot(data[phase][:, 2], label='{:5} acc'.format(phase))

    plt.xticks(np.arange(0, len(data[PHASE[0]][:, 1]), 2))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.axis([0, len(data[PHASE[0]][:, 1]) - 1, 0, 1])
    plt.axes()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.legend(bbox_to_anchor=(0.95, 0.05), loc=4)
    plt.savefig(dst + 'acc.jpg', dpi=600)
    plt.close()
    return


def show(img, size, figname='montage'):
    npimg = img.numpy()
    fig = plt.figure(figsize=size)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    fig.savefig(figname + '.JPG', bbox_inches='tight', dpi=175)
    return


def montage(imglist, figname, scale=0.4):
    imlist, imcomb = [], []
    for imgdir in imglist:
        img = Image.open(imgdir)
        w, h = img.size
        w, h = int(w * scale), int(h * scale)
        rto = round(w / h, 1)
        img.thumbnail((w, h), Image.ANTIALIAS)
        img = transforms.ToTensor()(img)
        if len(imlist) == 8:
            imlist.append(img)
            imcomb.append(torch.stack(imlist))
            imlist = []
        else:
            imlist.append(img)

    if len(imlist) != 0: imcomb.append(torch.stack(imlist))
    print('total row: {}'.format(len(imcomb)))

    if len(imcomb) == 0:
        print('no images')
        return
    elif len(imcomb) <= 8:
        size = (24 * rto, 24 / 8 * len(imcomb))
        show(make_grid(torch.cat(imcomb, 0), padding=40), size, figname)
    else:
        N = len(imcomb) // 8
        R = len(imcomb) % 8
        for i in range(N):
            size = (24 * rto, 24)
            print('processing row @ {}'.format(i * 8))
            show(make_grid(torch.cat(imcomb[i:i + 8], 0), padding=40), size, figname + str(i))

        if R != 0:
            size = (24 * rto, 24 / 8 * (R))
            show(make_grid(torch.cat(imcomb[N * 8:], 0), padding=40), size, figname + str(N))
    return


def folderView(src, dst, figname='Montage'):
    if not os.path.exists(dst): os.makedirs(dst)
    filelist = [f for f in glob(src + '*.JPG')]
    print('Folder {} contains {} images'.format(src, len(filelist)))
    if len(filelist) > 64:
        print('too many images! the result will show montage with 64 ramdonly picked images')
        filelist = [filelist[i] for i in sorted(random.sample(range(len(filelist)), 64))]

    montage(filelist, dst + figname)
    print('done')
    return


def folderViewL(src, dst):
    for phase in ['train', 'val']:
        for dir_label in glob(os.path.join(src + phase + '/', '*')):
            figname = 'montage_' + phase + '_' + os.path.basename(dir_label)
            folderView(dir_label + '/', dst, figname)
    return
