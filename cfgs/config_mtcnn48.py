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

# cfg.train_list = [cfg.name + "_new_train.txt"]
cfg.test_list = cfg.name + "_new_val.txt"


#cfg.train_list = [cfg.name + "_train_test_100.txt"]
#cfg.test_list = cfg.name + "_val.txt"




# cfg.hflip = False

# cfg.BATCH_SIZE = 128

# cfg.CLS_OHEM = True
# cfg.CLS_OHEM_RATIO = 0.7
# cfg.BBOX_OHEM = False
# cfg.BBOX_OHEM_RATIO = 0.7

# cfg.EPS = 1e-14
# cfg.LR_EPOCH = [8, 14]


cfg.nrof_classes = 10575#num of class label

cfg.train_list = "train_files.txt"
#inception_resnet_v1
cfg.image_size = 160
#resnet
#cfg.image_size = 224

cfg.center_loss_alfa = 0.9#center loss's learning rate

cfg. center_loss_factor = 1e-2#center loss's weight

cfg.weight_decay = 5e-5

cfg.random_crop = False

cfg.random_flip = True

cfg.shuffle = True

cfg.keep_probability = 0.8#drop out, keep rate

#inception_resnet_v1
cfg.feature_length = 128#last full connected layer output for inception_resnet_v1
#resnet
#cfg.feature_length = 1024#last full connected layer output for inception_resnet_v1

cfg.validate = True# if validate on lfw else cfg.validate=False while train model



# cfg.weight_decay = 4e-3
cfg.weight_decay = 5e-5

