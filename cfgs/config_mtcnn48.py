import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.name = '48landmark/mtcnn'
cfg.img_size_12 = 12
cfg.img_size_24 = 24
cfg.img_size_48 = 48

# cfg.net_dir_neg = '12/neg'
# cfg.net_dir_par = '12/par'
# cfg.net_dir_pos = '12/pos'

cfg.train_list = [cfg.name + "_new_train.txt"]
cfg.test_list = cfg.name + "_new_val.txt"
# cfg.train_list = [cfg.name + "_test_train.txt"]
# cfg.test_list = cfg.name + "_test_val.txt"




# cfg.hflip = False

# cfg.BATCH_SIZE = 128

# cfg.CLS_OHEM = True
# cfg.CLS_OHEM_RATIO = 0.7
# cfg.BBOX_OHEM = False
# cfg.BBOX_OHEM_RATIO = 0.7

# cfg.EPS = 1e-14
# cfg.LR_EPOCH = [8, 14]




cfg.weight_decay = 4e-3
# cfg.weight_decay = 0

