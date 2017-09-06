import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.name = 'train_file/mtcnn'
cfg.img_size_12 = 12
cfg.img_size_24 = 24
cfg.img_size_48 = 48

# cfg.net_dir_neg = '12/neg'
# cfg.net_dir_par = '12/par'
# cfg.net_dir_pos = '12/pos'

# cfg.train_list = [cfg.name + "_train.txt"]
# cfg.test_list = cfg.name + "_val.txt"
cfg.train_list = [cfg.name + "_test_train.txt"]
cfg.test_list = cfg.name + "_test_val.txt"




# cfg.hflip = False

# cfg.BATCH_SIZE = 128

# cfg.CLS_OHEM = True
# cfg.CLS_OHEM_RATIO = 0.7
# cfg.BBOX_OHEM = False
# cfg.BBOX_OHEM_RATIO = 0.7

# cfg.EPS = 1e-14
# cfg.LR_EPOCH = [8, 14]


cfg.channels_12 = [10, 16, 32]
cfg.kernel_size_12 = [3, 3, 3]

cfg.channels_24 = [28, 48, 64]
cfg.kernel_size_24 = [3, 3, 2]

cfg.channels_48 = [32, 64, 64, 128]
cfg.kernel_size_48 = [3, 3, 3, 2]


# cfg.weight_decay = 5e-4
cfg.weight_decay = 0

