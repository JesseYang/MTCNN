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

def process_resutl(pre_bbox, input_image):
    im = misc.imread(input_image, mode='RGB')
    ori_h, ori_w, _ = im.shape
    
    x1 = pre_bbox[0] * ori_w + ori_w
    y1 = pre_bbox[1] * ori_h + ori_h
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

def predict_image(input_image, predict_func):
    # img = cv2.imread(input_image)
    # cvt_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = misc.imread(input_image, mode='RGB')
    image = cv2.resize(img,(cfg.img_size_12, cfg.img_size_12))
    image = np.expand_dims(image, axis=0)

    # image = tf.cast(image, tf.float32) * (1.0 / 255)
    # image_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    # image_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    # image = (image - image_mean) / image_std
    # pdb.set_trace()
  
    label = np.array([1,1], dtype=np.int32)
    # label = tf.reshape(label, (1,1))
    predcit_result = predict_func([image,label])
    print(predcit_result)
    pre_label = predcit_result[0]
    pre_bbox = predcit_result[1][0]
    print(pre_label[0]) #[[ 0.5  0.5]]
    print(pre_bbox) #[[  3.60741063e-25  -7.15634251e-25   1.89428121e-25  -5.23704886e-25]]

    # result = np.argmax(pre_label)


    bbox = process_resutl(pre_bbox, input_image)
    print(bbox)
    cv2.rectangle(img,(bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,255,0), 4)
    
    cv2.imwrite("out.jpg",img)
    # result  = np.argmax(predict_img, 1)
    # print(result)
    # non_maximum_suppression(predict_img,0.6)
    print("predict over")


def predict_images(input_path, predict_func):
    print("predict all images....")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of model')
    parser.add_argument('--input_image', help='path of single image')
    parser.add_argument('--input_path', help='path of images')
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

