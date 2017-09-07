#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
import ntpath
import numpy as np
from scipy import misc
import argparse
import json
import cv2
from scipy import misc
from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator
from tensorflow.python import debug as tf_debug
import uuid
import tensorflow as tf
from tensorpack import *
from train import Model
from reader import *
from cfgs.config import cfg

def process_resutl(pre_bbox, input_image, im=None):
    if input_image != None:
        im = misc.imread(input_image, mode='RGB')
    ori_h, ori_w, _ = im.shape

    # w_rate = ori_w / float(cfg.img_size_12)
    # h_rate = ori_w / float(cfg.img_size_12)
    pre_bbox = [float(e) for e in pre_bbox]

    x1 = pre_bbox[0] * ori_w
    y1 = pre_bbox[1] * ori_h
    x2 = pre_bbox[2] * ori_w + ori_w
    y2 = pre_bbox[3] * ori_h + ori_h

    xmin = np.max([x1, 0])
    ymin = np.max([y1, 0])
    xmax = np.min([y2, ori_w])
    ymax = np.min([y2, ori_h])

    return [int(xmin), int(ymin), int(xmax), int(ymax)]
def get_pred_func(args):
    sess_init = SaverRestore(args.model_path)
    model = Model(args.net_format)
    predcit_config = PredictConfig(session_init = sess_init,
                                    model = model,
                                    input_names = ["input", "label"],
                                    output_names = ["LABELS", "BBOXS"])
    predict_func = OfflinePredictor(predcit_config)
    return predict_func

def predict_on_testimage(file_path, predict_func):
    if not os.path.exists(file_path):
        print("file does not exists")
        return
    with open(file_path, 'r')  as f:
        content = f.readlines()

    count = 0
    for item in content:
        image = item.split(' ')
        coor = image[2:] if int(image[1]) != 0 else None
        predict_image(image[0], predict_func, coor)
        count += 1
        if count == 300: #after 500 images tested break
            break

def predict_image(input_image, predict_func, coor=None):
    img = misc.imread(input_image, mode='RGB')
    image = cv2.resize(img, (cfg.img_size_12, cfg.img_size_12))
    image = np.expand_dims(image, axis=0)
  
    label = np.array([1,1], dtype=np.int32)
    predcit_result = predict_func([image, label])
    # print(predcit_result)
    pre_label = predcit_result[0][0][0][0]
    pre_bbox = predcit_result[1][0][0][0]
    # print(pre_label)
    print(pre_bbox)
    bbox = process_resutl(pre_bbox, None, img)
    print(bbox)
    if coor != None:
        ori_coor = process_resutl(coor, None, img)
        cv2.rectangle(img, (ori_coor[0], ori_coor[1]), (ori_coor[2], ori_coor[3]), (0, 0, 255), 1)
        
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
        cv2.putText(img, str(np.max(pre_label)), (ori_coor[0], ori_coor[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 122, 122))
    cv2.imwrite(os.path.join('output' ,str(uuid.uuid4()) + ".jpg"), img)
    print("predict over")

def predict_images(input_path, predict_func):
    print("predict all images....")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of model')
    parser.add_argument('--input_image', help='path of single image')
    parser.add_argument('--input_path', help='path of images')
    parser.add_argument('--file_path', help='txt file of image include label')
    parser.add_argument('--net_format', choices=['P-Net', 'R-Net', 'Q-Net'], default='P-Net')
    args = parser.parse_args()

    predict_func = get_pred_func(args)
    if args.input_image != None:
        if os.path.isfile('output.jpg'):
            os.remove('output.jpg')
        predict_image(args.input_image, predict_func)
    elif args.input_path != None:
        if os.path.isdir("output"):
            shutil.rmtree("output")
        os.mkdir("output")
        predict_images(args.input_path, predict_func)
    elif args.file_path != None:
        if os.path.isdir('output'):
            shutil.rmtree('output')
        os.mkdir('output')
        predict_on_testimage(args.file_path, predict_func)

