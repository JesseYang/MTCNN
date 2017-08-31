import os, sys
import pdb
import pickle
import numpy as np
from scipy import misc
import random
import six
from six.moves import urllib, range
import copy
import logging
import cv2

from tensorpack import *
from cfgs.config import cfg

def get_img_list(text_file):
    with open(text_file) as f:
        content = f.readlines()
    ret = [record.strip().split(' ') for record in content]
    filter_ret = []
    for idx, ele in enumerate(ret):
    	if int(ele[1]) == -1:
    		im_path = os.path.join(cfg.net_dir_par, ele[0])
    		flage = -1
    		label = np.asarray([float(e) for e in ele[2:]])
    	elif int(ele[1]) == 1:
    		im_path = os.path.join(cfg.net_dir_pos, ele[0])
    		flage = 1
    		label = np.asarray([float(e) for e in ele[2:]])
    	elif int(ele[1])  == 0:
    		im_path = os.path.join(cfg.net_dir_neg, ele[0])
    		flage = 0
    		label = np.asarray([float(0) for e in range(0,4)])
    	filter_ret.append([im_path, flage, label])
    return filter_ret

class Data(RNGDataFlow):

    def __init__(self, filename_list, shuffle=True):
        self.filename_list = filename_list

        if isinstance(filename_list, list) == False:
            filename_list = [filename_list]

        self.imglist = []
        for filename in filename_list:
            self.imglist.extend(get_img_list(filename))
        self.shuffle = shuffle


    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            img_path, flage, label = self.imglist[k]
            # print(flage)

            img = misc.imread(img_path, mode='RGB')        
            # img = cv2.imread(img_path)
            yield [img, flage, label]

if __name__ == '__main__':
    ds = Data(cfg.train_list)
    ds.reset_state()
    g = ds.get_data()
    dp = next(g)
    # import pdb
    # pdb.set_trace()
