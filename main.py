from Learning import Learning
from RawDataReader import RawDataReader


src = 'project/focus/hong/datasets/KSU_wheat/17ASH_AM-PANEL_KEY_PCTHEAD_20170419/'
dst = '_result/'

if __name__ == '__main__':
    csv_file_pct_A17 = ['keyfile/17ASH_AM-PANEL_KEY_PCTHEAD_20170419.csv']
    labels = {'17': csv_file_pct_A17}
    data_dict = RawDataReader(src, dst, labels).run()
    Learning(src, dst, data_dict).run(3)
