#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
import ntpath
import numpy as np
import math
from scipy import misc
import argparse
import json
import cv2

from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator
from tensorflow.python import debug as tf_debug
import uuid
import tensorflow as tf
from tensorpack import *
from train import Model
from reader import *
from cfgs.config import cfg

def non_maximum_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes).astype("float")
 
    # initialize the list of picked indexes 
    pick = []
 
    # grab the coordinates of the bounding boxes
    conf = boxes[:,0]
    x1 = boxes[:,1]
    y1 = boxes[:,2]
    x2 = boxes[:,3]
    y2 = boxes[:,4]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(conf)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        intersection = w * h
        union = area[idxs[:last]] + area[idxs[last]] - intersection
 
        # compute the ratio of overlap
        # overlap = (w * h) / area[idxs[:last]]
        overlap = intersection / union
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("float")

def process_result(pre_bbox, input_image, im=None):
    if input_image != None:
        im = misc.imread(input_image, mode='RGB')
    ori_h, ori_w, _ = im.shape

    pre_bbox = [float(e) for e in pre_bbox]

    x1 = pre_bbox[0] * ori_w
    y1 = pre_bbox[1] * ori_h
    x2 = pre_bbox[2] * ori_w + ori_w
    y2 = pre_bbox[3] * ori_h + ori_h

    xmin = np.max([x1, 0])
    ymin = np.max([y1, 0])
    xmax = np.min([x2, ori_w])
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
        predict_image(image[0], predict_func, coor,False)
        count += 1
        if count == 300: #after 500 images tested break
            break

def predict_image_multi_scale(input_image, predict_func):
    img = misc.imread(input_image, mode='RGB')
    h, w, _ = img.shape
    minsize = np.min([h, w])
    m = 12 / math.floor(minsize * 0.1)
    minsize = minsize * m
    factor_count = 1
    factor = 0.709 #scale factor
    scales = []
    feature_map = []
    while (minsize >= 12):
        scales.append(m * factor ** factor_count)
        minsize = minsize * factor
        factor_count += 1

    for i in range(len(scales)):
        hs = int(h * scales[i])
        ws = int(w * scales[i])
        # print(str(hs) + " h,w " + str(ws))
        new_img = cv2.resize(img, (hs, ws))
        

        image = cv2.resize(new_img, (cfg.img_size_12, cfg.img_size_12))
        image = np.expand_dims(image, axis=0)

        label = np.array([1], dtype=np.int32)
        predcit_result = predict_func([image, label])
        print(predcit_result)

        pre_label = predcit_result[0][0][0][0]
        pre_label = round(pre_label[1], 2)

        pre_bbox = predcit_result[1][0][0][0]
        if len(pre_bbox) == 0:
            continue
        print(pre_bbox)
        bbox = process_result(pre_bbox, None, new_img)

        x1 = bbox[0]  / scales[i]
        y1 = bbox[1]  / scales[i]
        x2 = bbox[2]  / scales[i]
        y2 = bbox[3]  / scales[i]

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        feature_map.append((pre_label, int(x1), int(y1), int(x2), int(y2)))
        #cv2.imwrite(str(uuid.uuid4()) + '.jpg', img)
    misc.imsave(str(uuid.uuid4()) + '.jpg', img)
    feature_map = np.asarray(feature_map)
    boxs = non_maximum_suppression(feature_map, 0.9)
    print(boxs)
    boxs = [int(e) for e in boxs[0]]
    cv2.rectangle(img, (boxs[1], boxs[2]), (boxs[3], boxs[4]), (255, 121, 154), 1)

    misc.imsave('nms_result.jpg', img)
  
def predict_image(input_image, predict_func, coor=None, flag = True):
    img = misc.imread(input_image, mode='RGB')
    image = cv2.resize(img, (cfg.img_size_12, cfg.img_size_12))
    image = np.expand_dims(image, axis=0)
  
    label = np.array([1], dtype=np.int32)
    predcit_result = predict_func([image, label])
    print(predcit_result)
    pre_label = predcit_result[0][0][0][0]
    pre_label = round(pre_label[1], 2)
    print(pre_label)
    pre_bbox = predcit_result[1][0][0][0]
    print(pre_bbox)
    bbox = process_result(pre_bbox, None, img)
    print(bbox)
    if coor != None:
        ori_coor = process_result(coor, None, img)
        cv2.rectangle(img, (ori_coor[0], ori_coor[1]), (ori_coor[2], ori_coor[3]), (0, 0, 255), 1) #blue
    # cv2.putText(img, str(np.max(pre_label)), (bbox[0], bbox[1] + 6),
    #                 cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 122, 122))
    if pre_label > 0.4:
        file_name = str(pre_label) + "_" + str(bbox[0]) + "_" + str(bbox[1]) + "_" + str(bbox[2]) + "_" + str(bbox[3]) + "_"
    else: file_name = str(pre_label) + "_"
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1) # red

    if flag == True:
        # cv2.imwrite(file_name + '_output.jpg', img)
        misc.imsave(file_name + '_output.jpg', img)
    else:
        # cv2.imwrite(os.path.join('output', file_name + input_image.replace('/','-')), img)
        misc.imsave(os.path.join('output', file_name + input_image.replace('/','-')), img)
    print("predict over")

def predict_images(input_path, predict_func):
    print("predict all images....")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of model')
    parser.add_argument('--net_format', choices=['P-Net', 'R-Net', 'Q-Net'], default='P-Net')

    parser.add_argument('--input_image', help='path of single image')
    parser.add_argument('--multi_scale', help='if need image pyrammid', action='store_true')

    parser.add_argument('--input_path', help='path of images')
    parser.add_argument('--file_path', help='txt file of image include label')
    args = parser.parse_args()

    if os.path.isdir("output"):
        shutil.rmtree("output")
    os.mkdir('output')
    predict_func = get_pred_func(args)
    if args.multi_scale:
        predict_image_multi_scale(args.input_image, predict_func)

    elif args.input_image != None:
        predict_image(args.input_image, predict_func, None)

    elif args.input_path != None:
        predict_images(args.input_path, predict_func)

    elif args.file_path != None:
        predict_on_testimage(args.file_path, predict_func)

