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
from utils import IoU
from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator
from tensorflow.python import debug as tf_debug
import uuid
import tensorflow as tf
from tensorpack import *
from train_p_net import Model
from reader import *
from cfgs.config import cfg
import time

def rerec(bboxA):
    # convert bboxA to square
    h = bboxA[:,3]-bboxA[:,1]
    w = bboxA[:,2]-bboxA[:,0]
    l = np.maximum(w, h)
    bboxA[:,0] = bboxA[:,0]+w*0.5-l*0.5
    bboxA[:,1] = bboxA[:,1]+h*0.5-l*0.5
    bboxA[:,2:4] = bboxA[:,0:2] + np.transpose(np.tile(l,(2,1)))
    return bboxA

def pad(total_boxes, w, h):
    #compute the padding coordinates (pad the bounding boxes to square)
    # pdb.set_trace()
    tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
    tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype = np.int32)
    dy = np.ones((numbox), dtype = np.int32)
    edx = tmpw.copy().astype(np.int32)
    edy = tmph.copy().astype(np.int32)

    x = total_boxes[:, 0].copy().astype(np.int32)
    y = total_boxes[:, 1].copy().astype(np.int32)
    ex = total_boxes[:, 2].copy().astype(np.int32)
    ey = total_boxes[:, 3].copy().astype(np.int32)

    tmp = np.where(ex > w)
    edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
    ex[tmp] = w

    tmp = np.where(ey>h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp]+h+tmph[tmp],1)
    ey[tmp] = h

    tmp = np.where(x<1)
    dx.flat[tmp] = np.expand_dims(2-x[tmp],1)
    x[tmp] = 1

    tmp = np.where(y<1)
    dy.flat[tmp] = np.expand_dims(2-y[tmp],1)
    y[tmp] = 1
    
    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph  ##for each shape (239,)

def generateBoundingBox(imap, reg, scale, t):
    # use heatmap to generate bounding boxes
    # pdb.set_trace()
    # print("generateBoundingBox starting.......")
    # print(imap.shape)
    # print(reg.shape)
    # imap = np.array(imap)
    stride=2
    cellsize=12


    imap = np.transpose(imap)
    dx1 = np.transpose(reg[:,:,0])
    dy1 = np.transpose(reg[:,:,1])
    dx2 = np.transpose(reg[:,:,2])
    dy2 = np.transpose(reg[:,:,3])

    y, x = np.where(imap >= t)
    # a = np.where(imap >= t)
    
    if y.shape[0] == 1:
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)
    score = imap[(y, x)] #(307,)
    reg = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]]))# (4,307) => (307,4)
    if reg.size==0:
        reg = np.empty((0,3))
    bb = np.transpose(np.vstack([y,x]))#(307,2)
    q1 = np.fix((stride*bb+1)/scale)#(307,2)
    q2 = np.fix((stride*bb+cellsize-1+1)/scale)#(307,2)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score,1), reg])##(307,9(x,y,score,rx1,ry1,rx2,ry2))
    return boundingbox, reg

# function pick = nms(boxes,threshold,type)
def nms(boxes, threshold, method):
    if boxes.size==0:
        return np.empty((0,3))
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = (x2-x1+1) * (y2-y1+1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size>0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o<=threshold)]
    pick = pick[0:counter]
    return pick

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

    return [xmin, ymin, xmax, ymax]
def get_pred_func(args):
    sess_init = SaverRestore(args.model_path)
    model = Model()
    predict_config = PredictConfig(session_init = sess_init,
                                    model = model,
                                    input_names = ["input"],
                                    output_names = ["LABELS", "BBOXS"])
    predict_func = OfflinePredictor(predict_config)
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

def detect_face_base12net(input_image, predict_func):
    img = misc.imread(input_image, mode='RGB')
    h, w, _ = img.shape
    minsize = np.min([h, w])
    m = 12 / math.floor(minsize * 0.1)
    minsize = minsize * m
    factor_count = 0
    factor = 0.709 #scale factor
    scales = []
    while (minsize >= 12):
        scales.append(m * np.power(factor, factor_count))
        minsize = minsize * factor
        factor_count += 1

    total_boxes = np.empty((0, 9))
    print("scale num" + str(len(scales)))
    for i in range(len(scales)):
        hs = int(np.ceil(h * scales[i]))
        ws = int(np.ceil(w * scales[i]))
        new_img = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_AREA)
        image = np.expand_dims(new_img, axis=0)
        predict_result = predict_func([image])

        pre_label = predict_result[0]
        # logger.info("classification shape")
        # logger.info(pre_label.shape)
        # # print(pre_label[0,:,:,:])
       
        pre_box = predict_result[1]
        # print(pre_box)
        # logger.info(pre_box[:,:,1])
        # logger.info("regression shape")
        
        # pre_label = round(pre_label[1], 2)
        boxes, _ = generateBoundingBox(predict_result[0][0,:,:,1], predict_result[1][0,:,:,:], scales[i], 0.6)
        # print(len(boxes))
        # logger.info(boxes.shape)
        # print(boxes)
        pick = nms(boxes.copy(), 0.5, 'Union')
        if boxes.size>0 and pick.size>0:
            boxes = boxes[pick,:]
            # print("after nms --------")
            # print(len(boxes))
            # print(boxes[0])
            # cv2.rectangle(img, (int(boxes[0][0]),int(boxes[0][1])), (int(boxes[0][2]),int(boxes[0][3])), (0,255,0))
    # misc.imsave(str(uuid.uuid4()) + '.jpg', img)
            total_boxes = np.append(total_boxes, boxes, axis=0)
    
    numbox = total_boxes.shape[0]#239
    # print("total boxes after scales " +  str(numbox))
    # pdb.set_trace()
    if numbox > 0:
        pick = nms(total_boxes.copy(), 0.7, 'Union') # return (239,)
        total_boxes = total_boxes[pick,:]
        bbw = total_boxes[:, 2] - total_boxes[:, 0]
        bbh = total_boxes[:, 3] - total_boxes[:, 1]
        x1 = total_boxes[:, 0] + total_boxes[:, 5] * bbw
        y1 = total_boxes[:, 1] + total_boxes[:, 6] * bbh
        x2 = total_boxes[:, 2] + total_boxes[:, 7] * bbw
        y2 = total_boxes[:, 3] + total_boxes[:, 8] * bbh 
        total_boxes = np.transpose(np.vstack([x1, y1, x2, y2, total_boxes[:, 4]]))# (239, 5) 4:xmin,ymin,xmax,ymax,prob
        total_boxes = rerec(total_boxes.copy())
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, temw, temh = pad(total_boxes.copy(), w, h)

    numbox = total_boxes.shape[0]
    print("this image detected face: " + str(numbox))
    # for i in range(5):
    #     boxes = total_boxes[i,:][0:5]
        
    #     cv2.rectangle(img, (int(boxes[0]),int(boxes[1])), (int(boxes[2]),int(boxes[3])), (0,255,0))
    # misc.imsave(str(uuid.uuid4()) + '.jpg', img)
    # print("first stage over")
    return total_boxes
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of model')
    args = parser.parse_args()
    predict_func = get_pred_func(args)

    start_time = time.time()

    min_face_size = cfg.img_size_24 ##ignore  min size face from detected face
    label_file = 'train_file/mtcnn12_val.txt'
    neg_save_dir = '24/negval'
    pos_save_dir = '24/posval'
    par_save_dir = '24/parval'
    im_dir = 'dataset/val_images'

    neg_txt = open('24/negval_24.txt', 'w')
    pos_txt = open('24/posval_24.txt', 'w')
    par_txt = open('24/parval_24.txt', 'w')

    ori_file_24  = open('24/ori_neg_val24_.txt', 'w')

    with open(label_file, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    print("total size: " + str(num))
    p_idx = 0
    n_idx = 0
    d_idx = 0


    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_path = os.path.join(im_dir, annotation[0])
        bbox = annotation[2:]
        gts = np.array(bbox, dtype=np.float32).reshape(-1, 4)
        if not os.path.exists(im_path):
            continue
        print("face detecting on 12 net: " + im_path)
        net12_bboxs = detect_face_base12net(im_path, predict_func)
        
        img = misc.imread(im_path, mode = 'RGB')

        for box in net12_bboxs:
            xmin, ymin, xmax, ymax = box[0:4]
            crop_w = xmax -xmin + 1
            crop_h = ymax - ymin + 1
            
            if crop_w < min_face_size or crop_h < min_face_size or xmin < 0 or ymin < 0 or xmin >  xmax or ymin > ymax:
                continue

            Iou = IoU(box, gts)
            # print(Iou)
            # print(box)
            cropped_img = img[int(ymin): int(ymax) +1, int(xmin): int(xmax) + 1] ### +1 why?

            # save neg, pos, par on txt and save image
            if np.max(Iou) < 0.3:
                misc.imsave(os.path.join(neg_save_dir, str(n_idx) + ".jpg"), cropped_img)
                neg_txt.write(os.path.join(neg_save_dir,str(n_idx) + ".jpg 0") + '\n')
                ori_file_24.write(im_path + " " + os.path.join(neg_save_dir,str(n_idx) + ".jpg 0") + '\n')
                n_idx += 1
            else:
                idx = np.argmax(Iou)
                x1, y1, x2, y2 = gts[idx] ##ground true           
                offset_x1 = (x1 - xmin) / float(crop_w)
                offset_y1 = (y1 - ymin) / float(crop_h)
                offset_x2 = (x2 - xmax) / float(crop_w)
                offset_y2 = (y2 - ymax) / float(crop_h)

                if np.max(Iou) >= 0.65:
                    misc.imsave(os.path.join(pos_save_dir, str(p_idx) + ".jpg"), cropped_img)
                    pos_txt.write(os.path.join(pos_save_dir, str(p_idx) + ".jpg") + ' 1 %0.3f %0.3f %0.3f %0.3f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    ori_file_24.write(im_path + " " + os.path.join(pos_save_dir, str(p_idx) + ".jpg 1") + '\n')
                    p_idx += 1
                elif np.max(Iou) >= 0.4:
                    misc.imsave(os.path.join(par_save_dir, str(d_idx) + ".jpg"), cropped_img)
                    par_txt.write(os.path.join(par_save_dir, str(d_idx) + ".jpg") + ' -1 %0.3f %0.3f %0.3f %0.3f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    ori_file_24.write(im_path + " " + os.path.join(par_save_dir, str(d_idx) + ".jpg -1") + '\n')
                    d_idx += 1
            print("images done, positive: %s negative: %s part: %s"%(p_idx,n_idx,d_idx))
    print(float(time.time() - start_time))
    neg_txt.close()
    par_txt.close()
    pos_txt.close()







        # img = cv2.imread(os.path.join(im_dir, im_path))

        # ori_h, ori_w, ori_c = img.shape

        # for box in boxes: ###box (xmin, ymin, xmax, ymax)
        #     x1, y1, x2, y2 = box
        #     w = x2 - x1 + 1
        #     h = y2 - y1 + 1



    # if os.path.isdir("output"):
    #     shutil.rmtree("output")
    # os.mkdir('output')

    # if args.multi_scale:
    #     predict_image_multi_scale(args.input_image, predict_func)

    # elif args.input_image != None:
    #     predict_image(args.input_image, predict_func, None)

    # elif args.input_path != None:
    #     predict_images(args.input_path, predict_func)

    # elif args.file_path != None:
    #     predict_on_testimage(args.file_path, predict_func)

