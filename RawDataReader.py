import os
import random
import pandas as pd

PHASE = ['training', 'validation']
PCT = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


class RawDataReader:
    def __init__(self, src, dst, csv_files):
        if not os.path.exists(dst):
            os.makedirs(dst)
        self.src = src
        self.dst = dst
        self.csv = csv_files
        self.plot = None
        self.data_dict = {p: {x: [] for x in PCT} for p in PHASE}

    def run(self):
        self.loadfile()
        self.spPlots()
        self.report()
        return self.data_dict

    def loadfile(self):
        plot_id = []
        for file in self.csv['17']:
            filePlots = pd.Series.to_dict(pd.read_csv(file)['plot_id'])
            plot_id += [v for k, v in filePlots.items()]
        self.plot = sorted(set(plot_id))

        plot_files = {p: {pct: [] for pct in PCT} for p in self.plot}
        for file in self.csv['17']:
            print('Loading file: {}'.format(file))
            df = pd.read_csv(file)
            fileNames = pd.Series.to_dict(df['image_file_name'])
            filePlots = pd.Series.to_dict(df['plot_id'])
            filePCThd = pd.Series.to_dict(df['PCTHEAD'])
            files_cam = pd.Series.to_dict(df['camera_sn'])  # dict type Camera #5: 0671720638(nadir)

            for idx, plot in filePlots.items():
                if files_cam[idx] == 'CAM_0671720638':
                    continue
                for pct in PCT:
                    if pct - 4 < filePCThd[idx] < pct + 4:
                        plot_files[plot][pct].append(self.src + fileNames[idx])
        self.plot_files = plot_files

    def spPlots(self):
        random.shuffle(self.plot)
        plot_sep = {PHASE[0]: self.plot[:-100], PHASE[1]: self.plot[-100:]}
        for phase in PHASE:
            for plot in plot_sep[phase]:
                for pct in PCT:
                    for img in self.plot_files[plot][pct]:
                        if os.path.isfile(img):
                            self.data_dict[phase][pct].append(img)
                        else:
                            print('File not found: ' + img)

    def report(self):
        print('-' * 100)
        print('numbers of plots: {}'.format(len(self.plot)))
        print('-' * 100)
        for phase in PHASE:
            for pct in PCT:
                print('{}-{:03}: {}'.format(phase, pct, len(self.data_dict[phase][pct])))
