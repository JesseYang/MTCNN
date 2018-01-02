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
from train_r_net import Model as R_Model
from train_p_net import Model as P_Model
from train_o_net import Model as O_Model
from reader_o_net import *
from cfgs.config import cfg
from face_train_inception_resnet_v1 import Model as Inception_Resnet_V1_Model

import time


def process_result(pre_bbox, input_image=None, im=None):
    if not input_image == None:
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

def process_landmark(pre_bbox, input_image=None, im=None):
    # pdb.set_trace()
    pre_bbox = np.array(pre_bbox, dtype=np.float32).reshape(-1, 2)
    if not input_image == None:
        im = misc.imread(input_image, mode='RGB')
    ori_h, ori_w, _ = im.shape

    new_center_x = ori_w / 2.0
    new_center_y = ori_h / 2.0

    result = []

    for item in pre_bbox:
        newx1 = item[0] * ori_w + new_center_x
        newy1 = item[1] * ori_h + new_center_y
        if newx1 <= 0 or newy1 <=0 or newx1 >= ori_w or newy1 >= ori_h:
            continue
        result.append(newx1)
        result.append(newy1)
    result = [int(e) for e in result]
    return result


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



# function [boundingbox] = bbreg(boundingbox,reg)
def bbreg(boundingbox,reg):
    # calibrate bounding boxes
    if reg.shape[1]==1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:,2]-boundingbox[:,0]+1
    h = boundingbox[:,3]-boundingbox[:,1]+1
    b1 = boundingbox[:,0]+reg[:,0]*w
    b2 = boundingbox[:,1]+reg[:,1]*h
    b3 = boundingbox[:,2]+reg[:,2]*w
    b4 = boundingbox[:,3]+reg[:,3]*h
    boundingbox[:,0:4] = np.transpose(np.vstack([b1, b2, b3, b4 ]))
    return boundingbox

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



def process_landmark_result(pre_bbox, ori_cor):
    # pdb.set_trace()
    pre_bbox = np.array(pre_bbox, dtype=np.float32).reshape(-1, 2)
    # if not input_image == None:
    #     im = misc.imread(input_image, mode='RGB')
    # ori_h, ori_w, _ = im.shape
    ori_h = ori_cor[3] - ori_cor[1]
    ori_w = ori_cor[2] - ori_cor[0]

    new_center_x = ori_w / 2.0
    new_center_y = ori_h / 2.0

    result = []

    for item in pre_bbox:
        newx1 = item[0] * ori_w + new_center_x
        newy1 = item[1] * ori_h + new_center_y
        if newx1 <= 0 or newy1 <=0 or newx1 >= ori_w or newy1 >= ori_h:
            continue
        result.append(newx1 + ori_cor[0])
        result.append(newy1 + ori_cor[1])
    result = [int(e) for e in result]
    return result


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

def get_pred_func(args):
    #classification model
    sess_init = SaverRestore(args.pmodel_path)
    p_model = P_Model()
    p_predict_config = PredictConfig(session_init = sess_init,
                                    model = p_model,
                                    input_names = ["input"],
                                    output_names = ["LABELS", "BBOXS"])
    p_predict_func = OfflinePredictor(p_predict_config)
    #regression model
    sess_init = SaverRestore(args.rmodel_path)
    r_model = R_Model()
    r_predict_config = PredictConfig(session_init = sess_init,
                                    model = r_model,
                                    input_names = ["input"],
                                    output_names = ["LABELS", "BBOXS"])
    r_predict_func = OfflinePredictor(r_predict_config)
    #landmark model
    sess_init = SaverRestore(args.omodel_path)
    o_model = O_Model()
    o_predict_config = PredictConfig(session_init = sess_init,
                                    model = o_model,
                                    input_names = ["input"],
                                    output_names = ["LABELS", "BBOXS", "LANDMARK"])
    o_predict_func = OfflinePredictor(o_predict_config)

    return p_predict_func, r_predict_func, o_predict_func

def predict_on_testimage(file_path, predict_func):

    with open(file_path) as f:
        contents = f.readlines()
    f.close()
    sava_path = '48landmark/output_val'
    if os.path.exists(sava_path):
        shutil.rmtree(sava_path)
    os.mkdir(sava_path)

    for idx, line in enumerate(contents):
        item = line.split(" ")
        if len(item) < 7:
            continue
        img = misc.imread(item[0], mode='RGB')
        image = cv2.resize(img, (cfg.img_size_48, cfg.img_size_48))
        image = np.expand_dims(image, axis=0)
        predict_result = predict_func([image])

        pre_label = predict_result[0]
        pre_label = round(pre_label[0,1], 2)
        pre_bbox = predict_result[1][0]
        # pdb.set_trace()
        pre_landmark = predict_result[2][0]

        #ori four coor
        xmin, ymin, xmax, ymax = process_result(item[2: 6], im=img)
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 1)
        #ori 10 coor
        result = process_landmark(item[6:], input_image=item[0], im=None)
        result = np.array(result, dtype = np.float32).reshape(-1, 2)
        for co in result:
            cv2.circle(img,(co[0], co[1]), 1, (255,255,0), 2) 
        #draw predicted result
        pre_xmin, pre_ymin, pre_xmax, pre_ymax = process_result(pre_bbox, im=img)
        cv2.rectangle(img, (int(pre_xmin), int(pre_ymin)), (int(pre_xmax), int(pre_ymax)), (0, 0, 255), 1)
        
        result = process_landmark(pre_landmark, input_image=item[0], im=None)
        result = np.array(result, dtype = np.float32).reshape(-1, 2)
        for co in result:
            cv2.circle(img,(co[0], co[1]), 1, (0, 0, 255), 2) 
        misc.imsave(os.path.join(sava_path, str(pre_label) + "_" + item[0].replace('/','-')), img)
       

def predict_image_multi_scale(img, p_predict_func, r_predict_func, o_predict_func):
    start_time = time.time()
    # img = misc.imread(args.input_image, mode='RGB')
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
    # print("after multi_scale : " + str(start_time - time.time()))
    total_boxes = np.empty((0, 9))
    # print("scale num" + str(len(scales)))
    for i in range(len(scales)):
        hs = int(np.ceil(h * scales[i]))
        ws = int(np.ceil(w * scales[i]))
        new_img = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_AREA)
        # new_img = cv2.resize(new_img, (cfg.img_size_24, cfg.img_size_24), interpolation=cv2.INTER_AREA)
        image = np.expand_dims(new_img, axis=0)
        predict_result = p_predict_func([image])
        # print("scale : " +  str(i) + " after p-net " + str(start_time - time.time()))
        boxes, _ = generateBoundingBox(predict_result[0][0,:,:,1], predict_result[1][0,:,:,], scales[i], 0.6)
        # print("scale : " +  str(i) + " after boundignbox : " + str(start_time - time.time()))
        # print(len(boxes))
        # logger.info(boxes.shape)
        # print(boxes)
        pick = nms(boxes.copy(), 0.5, 'Union')

        # print("scale : " +  str(i) + " after nms " + str(start_time - time.time()))

        if boxes.size>0 and pick.size>0:
            boxes = boxes[pick,:]
            total_boxes = np.append(total_boxes, boxes, axis=0)
    # pdb.set_trace()
    # print("after all scale and p-net " + str(start_time - time.time()))
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
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)
    # print("after p-net's nms : "  + str(start_time - time.time()))
    numbox = total_boxes.shape[0]
    # print("this image total predicted face: " + str(numbox))
    # for i in range(numbox):
    #     boxes = total_boxes[i,0:-1]
    #     cv2.rectangle(img, (int(boxes[0]),int(boxes[1])), (int(boxes[2]),int(boxes[3])), (0,255,0))
    # misc.imsave(str(uuid.uuid4()) + '.jpg', img)
    # print("first stage over")
    ########## second stage
    if numbox>0:
        # second stage
        tempimg = np.zeros((numbox, 24, 24, 3))
        for k in range(0,numbox):
            tmp = np.zeros((int(tmph[k]),int(tmpw[k]),3))
            tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]
            if tmp.shape[0]>0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1]==0:
                tempimg[k,:,:,:] = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_AREA)
            else:
                return np.empty()
        # tempimg = (tempimg-127.5)*0.0078125
        # tempimg1 = np.transpose(tempimg, (3,1,0,2))
        # pdb.set_trace()
        out = r_predict_func([tempimg])
        # print("after r-net: " + str(start_time - time.time()))
        out0 = np.transpose(out[0])#classification
        out1 = np.transpose(out[1])#regression
        score = out0[1,:]
        ipass = np.where(score>0.7)
        total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])
        mv = out1[:,ipass[0]]
        if total_boxes.shape[0]>0:
            pick = nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick,:]
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:,pick]))
            total_boxes = rerec(total_boxes.copy())

    numbox = total_boxes.shape[0]
    input_box = total_boxes.copy()
    # print(input_box.shape[0])
    # print(numbox)
    # for i in range(numbox):
    #     boxes = total_boxes[i,0:-1]
    #     cv2.rectangle(img, (int(boxes[0]),int(boxes[1])), (int(boxes[2]),int(boxes[3])), (0,255,0))
    # misc.imsave(str(uuid.uuid4()) + '.jpg', img)
###o_net start
    # print("after r-net's nms " + str(start_time - time.time()))
    points = np.empty(0)
    # input_box = np.empty(0)
    if numbox > 0:
        total_boxes = np.fix(total_boxes).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)
        tempimg = np.zeros((numbox, 48, 48, 3))
        for k in range(0,numbox):
            tmp = np.zeros((int(tmph[k]),int(tmpw[k]),3))
            tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]
            if tmp.shape[0]>0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1]==0:
                tempimg[k,:,:,:] = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_AREA)
            else:
                return np.empty()
        # pdb.set_trace()
        out = o_predict_func([tempimg])
        # print("after o-net " + str(start_time - time.time()))
        out0 = np.transpose(out[0])#classification
        out1 = np.transpose(out[1])#regression
        out2 = np.transpose(out[2])#landmark
        score = out0[1,:]
        points = out2
        ipass = np.where(score > 0.7)
        points = points[:, ipass[0]]
        input_box = input_box[ipass[0], :]
        total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])
        # print(out[1].shape)
        mv = out1[:,ipass[0]]

        w = total_boxes[:,2]-total_boxes[:,0]+1
        h = total_boxes[:,3]-total_boxes[:,1]+1
        # print("after o-net " + str(start_time - time.time()))
        # points = np.transpose(points)
        # points[0:5,:] = np.tile(w,(5, 1))*points[0:5,:] + np.tile(total_boxes[:,0],(5, 1))-1
        # points[5:10,:] = np.tile(h,(5, 1))*points[5:10,:] + np.tile(total_boxes[:,1],(5, 1))-1
        if total_boxes.shape[0]>0:
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))
            pick = nms(total_boxes.copy(), 0.7, 'Min')
            total_boxes = total_boxes[pick,:]
            points = points[:,pick]
            input_box = input_box[pick,:]
                
    # return total_boxes, points
    # pdb.set_trace()
    # print("after o-net'nms " + str(start_time - time.time()))

            points = np.transpose(points)
    # total_boxes = np.transpose(total_boxes)
    # face_num = total_boxes.shape[0]
    # print(face_num)

    # for i in range(face_num):
    #     if total_boxes[i,4] < 0.9:
    #         continue
    #     img = cv2.rectangle(img,
    #                        (int(total_boxes[i,0]), int(total_boxes[i,1])),
    #                        (int(total_boxes[i,2]), int(total_boxes[i,3])),
    #                        (255, 0, 0),
    #                        3)
    #     # point = points[i,:]
    #     # pdb.set_trace()
    #     # point = np.reshape(point, (-1 , 2)) 
    #     # for j in range(point.shape[0]):
    #     #     img = cv2.circle(img, (point[j, 0], point[j, 1]), 2, (0, 255, 0), thickness=2, lineType=8, shift=0)
    #     result = process_landmark_result(points[i,:], input_box[i,:])
    #     result = np.array(result, dtype = np.float32).reshape(-1, 2)
    #     for co in result:
    #         cv2.circle(img,(co[0], co[1]), 1, (0,0,255), 2) 
    # # return img
    # misc.imsave('output.jpg', img)
    # print("step over " + str(start_time - time.time()))

    # print("over=====")
    return total_boxes, points, input_box
  
def predict_image(input_image, predict_func, coor=None, flag = True):
    img = misc.imread(input_image, mode='RGB')
    # image = (img-127.5)*0.0078125
    image = cv2.resize(img, (cfg.img_size_24, cfg.img_size_24))
    image = np.expand_dims(image, axis=0)
  
    
    predict_result = predict_func([image])
    # print(predict_result)
    # pdb.set_trace()
    pre_label = predict_result[0]
    pre_label = np.round(pre_label[0, 1], 2)
    # print(pre_label)
    pre_bbox = predict_result[1]

    # print(pre_bbox)
    bbox = process_result(pre_bbox[0, :], None, img)
    bbox = [int(e) for e in bbox]
    # print(bbox)
    if coor != None:
        ori_coor = process_result(coor, None, img)
        ori_coor = [int(e) for e in ori_coor]
        cv2.rectangle(img, (ori_coor[0], ori_coor[1]), (ori_coor[2], ori_coor[3]), (0, 0, 255), 2) 
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

def predict_on_video(video_path, p_predict_func, r_predict_func, o_predict_func, face_model):
 
    cap = cv2.VideoCapture(video_path)
    frame_dix = 0
    if cap.isOpened == False:
        cap.open(video_path)
    # frame_counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))#total frames of this video
    # fps = cap.get(cv2.CAP_PROP_FPS)#frame num per
    # print(frame_counts)
    # print(fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoWrite = cv2.VideoWriter("face_det_result.mp4", cv2.VideoWriter_fourcc(*'mp4v'),30,(width, height))

    with open("face_feature.json", 'r') as f:
        feature = json.loads(f.read())
    face_name = list(feature.keys())
    # pdb.set_trace()
    face_feature = list(feature.values())

    while cap.isOpened:
        ret, img = cap.read()
        if ret == False:
            break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[0:2]
        total_boxes, points, input_box = predict_image_multi_scale(img, p_predict_func, r_predict_func, o_predict_func)
        face_num = total_boxes.shape[0]


        imgs = []
        imgs_idx = []
        for i in range(face_num):
            if total_boxes[i,4] < 0.9:
                continue
            xmin = int(total_boxes[i,0])
            ymin = int(total_boxes[i,1])
            xmax = int(total_boxes[i,2])
            ymax = int(total_boxes[i,3])

            # img = cv2.rectangle(img,
            #                    (int(total_boxes[i,0]), int(total_boxes[i,1])),
            #                    (int(total_boxes[i,2]), int(total_boxes[i,3])),
            #                    (255, 0, 0),
            #                    1)
          
            # result = process_landmark_result(points[i,:], input_box[i,:])
            # result = np.array(result, dtype = np.float32).reshape(-1, 2)
            # for co in result:
            #     cv2.circle(img,(co[0], co[1]), 3, (0,255,255), 3) 
        # return img
            if (xmax - xmin) < 5 or (ymax - ymin) < 5:
                continue
            new_xmin = xmin - (-10)
            new_ymin = ymin - (-10)
            new_xmax = xmax + (-10)
            new_ymax = ymax + (-10)
            if new_xmin < 0:
                new_xmin = 0
            if new_ymin < 0:
                new_ymin = 0
            if new_xmax > w:
                new_xmax = w
            if new_ymax > h:
                new_ymax = h

            detected_face = img[new_ymin:new_ymax, new_xmin:new_xmax]
            resized_face = cv2.resize(detected_face, (160, 160))
            imgs.append(resized_face)
            imgs_idx.append(i)
        if len(imgs_idx) <= 0:
            continue

        predict_result = face_model([imgs])[0]
        # print(face_feature)
        reco_num = predict_result.shape[0]
        assert (reco_num == len(imgs_idx)), "predict length not equal"
        for j in range(reco_num):
            diff_feature  = np.sum(np.square(np.asarray(face_feature) - predict_result[j]), 1)
            max_idx = np.argmax(diff_feature)
            print(diff_feature[max_idx])
            if diff_feature[max_idx] >= 0.268:
                text_area = face_name[max_idx]
            else:
                text_area = "stranger"

            img = cv2.rectangle(img,
                               (int(total_boxes[imgs_idx[j],0]), int(total_boxes[imgs_idx[j],1])),
                               (int(total_boxes[imgs_idx[j],2]), int(total_boxes[imgs_idx[j],3])),
                               (255, 0, 0),
                               1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, text_area, (int(total_boxes[imgs_idx[j],0]), (int(total_boxes[imgs_idx[j],1]))), font, 1, (255,233,134),1)
        # misc.imsave(os.path.join('video_output',str(uuid.uuid4()) + '.jpg'), img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        videoWrite.write(img)
        # print("step over " + str(start_time - time.time()))




    print("video over....")



if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--pmodel_path', default="train_log/train_p_net0920-163732/model-3621124", help='path of pmodel')
    parser.add_argument('--rmodel_path', default="train_log/train_r_net0919-181029/model-3323040", help='path of rmodel')
    # parser.add_argument('--omodel_path', default="train_log/train_o_net0925-104057/model-2603987", help='path of omodel')
    parser.add_argument('--omodel_path', default="train_log/train_o_net0925-104057/model-2611187", help='path of omodel')
    
    parser.add_argument('--face_model_path', default="/home/user/yjf/center_loss/train_log/facenet_mix_per_image_standardizatio/model-219000", help='path of face_recognition')

    parser.add_argument('--input_image', help='path of single image')
    parser.add_argument('--multi_scale', help='if need image pyrammid', action='store_true')

    parser.add_argument('--video_path', help='path of images')
    parser.add_argument('--video_output_path', default='video_output')
    parser.add_argument('--file_path', help='txt file of image include label')
    args = parser.parse_args()

    # if os.path.isdir("output"):
    #     shutil.rmtree("output")
    # os.mkdir('output')
    start_time = time.time()
    p_predict_func, r_predict_func, o_predict_func = get_pred_func(args)
    print("load model: " + str(float(start_time - time.time())))
    if args.multi_scale:
        predict_image_multi_scale(args.input_image, p_predict_func, r_predict_func, o_predict_func)

    elif args.input_image != None:
        predict_image(args.input_image, p_predict_func, r_predict_func, o_predict_func)

    elif args.video_path != None:
        if os.path.exists(args.video_output_path):
            shutil.rmtree(args.video_output_path)
        os.mkdir(args.video_output_path)
        sess_init = SaverRestore(args.face_model_path)
        model = Inception_Resnet_V1_Model(False)

        predict_config = PredictConfig(session_init = sess_init,
                                        model = model,
                                        input_names = ["input"],
                                        output_names = ["FEATURE"])
        face_recognition_func = OfflinePredictor(predict_config)
    
        predict_on_video(args.video_path, p_predict_func, r_predict_func, o_predict_func, face_recognition_func)

    elif args.file_path != None:
        predict_on_testimage(args.file_path, o_predict_func)

