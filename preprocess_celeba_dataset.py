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
from train_r_net import Model as R_Model
from train_p_net import Model as P_Model
from reader import *
from cfgs.config import cfg
import time


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

def detect_face_base_pr_net(input_image, p_predict_func, r_predict_func):
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
    # print("scale num" + str(len(scales)))
    for i in range(len(scales)):
        hs = int(np.ceil(h * scales[i]))
        ws = int(np.ceil(w * scales[i]))
        new_img = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_AREA)
        image = np.expand_dims(new_img, axis=0)
        predict_result = p_predict_func([image])
        pre_label = predict_result[0]
        pre_box = predict_result[1]

        boxes, _ = generateBoundingBox(predict_result[0][0,:,:,1], predict_result[1][0,:,:,], scales[i], 0.6)
       
        pick = nms(boxes.copy(), 0.5, 'Union')
        if boxes.size>0 and pick.size>0:
            boxes = boxes[pick,:]
            total_boxes = np.append(total_boxes, boxes, axis=0)
    
    numbox = total_boxes.shape[0]#239
    
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
    # pdb.set_trace()
    numbox = total_boxes.shape[0]#239
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
        
        out = r_predict_func([tempimg])
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

    # numbox = total_boxes.shape[0]
    # print(numbox)
    # for i in range(numbox):
    #     boxes = total_boxes[i,0:-1]
    #     cv2.rectangle(img, (int(boxes[0]),int(boxes[1])), (int(boxes[2]),int(boxes[3])), (0,255,0))
    # misc.imsave(str(uuid.uuid4()) + '.jpg', img)

    # print("over=====")
    return total_boxes

def get_pred_func(args):
    sess_init = SaverRestore(args.pmodel_path)
    p_model = P_Model()
    p_predict_config = PredictConfig(session_init = sess_init,
                                    model = p_model,
                                    input_names = ["input"],
                                    output_names = ["LABELS", "BBOXS"])
    p_predict_func = OfflinePredictor(p_predict_config)


    sess_init = SaverRestore(args.rmodel_path)
    r_model = R_Model()
    r_predict_config = PredictConfig(session_init = sess_init,
                                    model = r_model,
                                    input_names = ["input"],
                                    output_names = ["LABELS", "BBOXS"])
    r_predict_func = OfflinePredictor(r_predict_config)
    return p_predict_func, r_predict_func 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pmodel_path', default='train_log/train_p_net0920-163732/model-1832376', help='path of pmodel')
    parser.add_argument('--rmodel_path', default='train_log/train_r_net0919-181029/model-3323040', help='path of rmodel')

    # parser.add_argument('--input_image', help='path of single image')
    # parser.add_argument('--multi_scale', help='if need image pyrammid', action='store_true')

    # parser.add_argument('--input_path', help='path of images')
    # parser.add_argument('--file_path', help='txt file of image include label')
    args = parser.parse_args()
    p_predict_func, r_predict_func = get_pred_func(args)

    min_face_size = cfg.img_size_48 ##ignore  min size/2 face from detected face
    p_idx = 0
    n_idx = 0
    d_idx = 0
    neg_save_dir = '48landmark/negtrain'
    pos_save_dir = '48landmark/postrain'
    par_save_dir = '48landmark/partrain'
    
    neg_txt = open('48landmark/neg_landmark.txt', 'w')
    pos_txt = open('48landmark/pos_landmark.txt', 'w')
    par_txt = open('48landmark/par_landmark.txt', 'w')

    ori_file_24  = open('48landmark/ori_.txt', 'w')


    wild_bbox_path = 'dataset/CelebA/Anno/list_bbox_celeba.txt'
    wild_face_path = 'dataset/CelebA/Img/img_celeba'
    wild_face_landmark_path = 'dataset/CelebA/Anno/list_landmarks_celeba.txt'

    with open(wild_bbox_path, 'r') as f:
        wild_faces = f.readlines()
    f.close()

    with open(wild_face_landmark_path, 'r') as f:
        wild_landmarks = f.readlines()
    f.close()

    start_time = time.time()
    error_count = 0
    for idx, face in enumerate(wild_faces):
        # if idx == 1000:
        #     break
        item = face.split()
        # print(item)
        img_path = os.path.join(wild_face_path, item[0])

        if not os.path.exists(img_path):
            continue
        coor = [int(e) for e in item[1: ]]
        coor[2] += coor[0] 
        coor[3] += coor[1]
        # break
        gts = np.array(coor, dtype=np.int32).reshape(-1, 4)
        # gts = np.array(item[1: ], dtype=np.int32).reshape(-1,4)
        landmark = wild_landmarks[idx].split()
        if landmark[0] != item[0]:
            error_count += 1
        coors = landmark[1: ]
        coors = np.asarray(coors, dtype=np.float32)

        print("face detecting on pnet and rnet: " + img_path)
        rectangles = detect_face_base_pr_net(img_path, p_predict_func, r_predict_func)
        
        img = misc.imread(img_path, mode = 'RGB')

        for box in rectangles:
            xmin, ymin, xmax, ymax = box[0:4]
            crop_w = xmax - xmin + 1
            crop_h = ymax - ymin + 1
            
            if crop_w < min_face_size / 2 or crop_h < min_face_size / 2 or xmin < 0 or ymin < 0 or xmin >  xmax or ymin > ymax :#or ymax > img.shape[0] or xmax > img.shape[1]:
                continue

            Iou = IoU(box, gts)
            cropped_img = img[int(ymin): int(ymax), int(xmin): int(xmax)] ### +1 why?

            # save neg, pos, par on txt and save image
            if np.max(Iou) < 0.3:
                misc.imsave(os.path.join(neg_save_dir, str(n_idx) + ".jpg"), cropped_img)
                neg_txt.write(os.path.join(neg_save_dir,str(n_idx) + ".jpg 0") + '\n')
                ori_file_24.write(img_path + " " + os.path.join(neg_save_dir,str(n_idx) + ".jpg 0") + '\n')
                n_idx += 1
            else:
                idx = np.argmax(Iou)
                x1, y1, x2, y2 = gts[idx] ##ground true           
                offset_x1 = (x1 - xmin) / float(crop_w)
                offset_y1 = (y1 - ymin) / float(crop_h)
                offset_x2 = (x2 - xmax) / float(crop_w)
                offset_y2 = (y2 - ymax) / float(crop_h)

                new_center_x = crop_w / 2.0 + xmin
                new_center_y = crop_h / 2.0 + ymin

                left_eye_x = (coors[0] - new_center_x) / float(crop_w)
                left_eye_y = (coors[1] - new_center_y) / float(crop_h)

                right_eye_x = (coors[2] - new_center_x) / float(crop_w)
                right_eye_y = (coors[3] - new_center_y) / float(crop_h)

                nose_x = (coors[4] - new_center_x) / float(crop_w)
                nose_y = (coors[5] - new_center_y) / float(crop_h)

                left_mouse_x = (coors[6] - new_center_x) / float(crop_w)
                left_mouse_y = (coors[7] - new_center_y) / float(crop_h)

                right_mouse_x = (coors[8] - new_center_x) / float(crop_w)
                right_mouse_y = (coors[9] - new_center_y) / float(crop_h)

                if np.max(Iou) >= 0.65:
                    misc.imsave(os.path.join(pos_save_dir, str(p_idx) + ".jpg"), cropped_img)
                    pos_txt.write(os.path.join(pos_save_dir, str(p_idx) + ".jpg") + ' 1 %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f\n'%(offset_x1, offset_y1, offset_x2, offset_y2,
                        left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y, left_mouse_x, left_mouse_y, right_mouse_x, right_mouse_y))
                    ori_file_24.write(img_path + " " + os.path.join(pos_save_dir, str(p_idx) + ".jpg 1") + '\n')
                    p_idx += 1
                elif np.max(Iou) >= 0.4:
                    misc.imsave(os.path.join(par_save_dir, str(d_idx) + ".jpg"), cropped_img)
                    par_txt.write(os.path.join(par_save_dir, str(d_idx) + ".jpg") + ' -1 %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f\n'%(offset_x1, offset_y1, offset_x2, offset_y2,
                        left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y, left_mouse_x, left_mouse_y, right_mouse_x, right_mouse_y))
                    ori_file_24.write(img_path + " " + os.path.join(par_save_dir, str(d_idx) + ".jpg -1") + '\n')
                    d_idx += 1
            print("images done, positive: %s negative: %s part: %s"%(p_idx,n_idx,d_idx))
            print("error sample num: " + str(error_count))
    print(float(time.time() - start_time) / 60.0)
    neg_txt.close()
    par_txt.close()
    pos_txt.close()




    # root = '/home/user/yjf/mtcnn/dataset/CelebA/Anno/list_landmarks_align_celeba.txt'
    # root = '/home/user/yjf/mtcnn/dataset/CelebA/Anno/list_landmarks_celeba.txt'

    # image_path = "/home/user/yjf/mtcnn/dataset/CelebA/Img/img_align_celeba/img_align_celeba" #Cel1
    # image_path = "/home/user/yjf/mtcnn/dataset/CelebA/Img/img_celeba"#Cel
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--pmodel_path', help='path of pmodel')
    # args = parser.parse_args()

    # with open(root, 'r') as f:
    #     content = f.readlines()

    # print(len(content))
    # count = 0
    # if os.path.exists('Cel'):
    #     shutil.rmtree('Cel')
    # os.mkdir('Cel')
    # # content = np.asarray(content)
    # for i in range(50):
    #     if count < 2:
    #         count += 1
    #         continue
    #     # pdb.set_trace()
       
    #     item =  ','.join(content[i].split())
    #     item = item.split(',')
    #     # print(item)
    #     # item = item.strip().split(',')

    #     img_path = os.path.join(image_path, item[0])
    #     print(img_path)
    #     if not os.path.exists(img_path):
    #         continue
    #     # print(item[1: -1])
    #     # pdb.set_trace()
    #     bbox = [int(e) for e in item[1:]]
    #     # print(len(bbox))
    #     # print(bbox)
    #     bbox = np.array(bbox, dtype=np.int32).reshape(-1, 2)
    #     img = misc.imread(img_path, mode='RGB')
    #     # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0))
    #     for box in bbox:
    #         cv2.circle(img,(box[0], box[1]), 1, (0,0,255), 2)  
    #     misc.imsave(os.path.join('Cel', str(item[0])), img)




###bbox 
    # # root = '/home/user/yjf/mtcnn/dataset/CelebA/Anno/list_landmarks_align_celeba.txt'
    # root = '/home/user/yjf/mtcnn/dataset/CelebA/Anno/list_bbox_celeba.txt'

    # # image_path = "/home/user/yjf/mtcnn/dataset/CelebA/Img/img_align_celeba/img_align_celeba" #Cel1
    # image_path = "/home/user/yjf/mtcnn/dataset/CelebA/Img/img_celeba"#Cel

    # # args = parser.parse_args()

    # with open(root, 'r') as f:
    #     content = f.readlines()

    # print(len(content))
    # count = 0
    # if os.path.exists('Cel'):
    #     shutil.rmtree('Cel')
    # os.mkdir('Cel')
    # # content = np.asarray(content)
    # for i in range(50):
    #     if count < 2:
    #         count += 1
    #         continue
    #     # pdb.set_trace()
       
    #     item =  ','.join(content[i].split())
    #     item = item.split(',')
    #     # print(item)
    #     # item = item.strip().split(',')

    #     img_path = os.path.join(image_path, item[0])
    #     print(img_path)
    #     if not os.path.exists(img_path):
    #         continue
    #     # print(item[1: -1])
    #     # pdb.set_trace()
    #     bbox = [int(e) for e in item[1:]]
    #     # print(len(bbox))
    #     # print(bbox)
    #     # bbox = np.array(bbox, dtype=np.int32).reshape(-1, 2)
    #     img = misc.imread(img_path, mode='RGB')
    #     cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2]+bbox[0], bbox[1]+bbox[3]), (255, 255, 0))
       
    #     misc.imsave(os.path.join('Cel', str(item[0])), img)




    # box_path = '/home/user/yjf/mtcnn/dataset/CelebA/Anno/list_bbox_celeba.txt'
    # point_path = '/home/user/yjf/mtcnn/dataset/CelebA/Anno/list_landmarks_celeba.txt'
    # image_path = "/home/user/yjf/mtcnn/dataset/CelebA/Img/img_celeba"

    # mtcnn_file = '48/mtcnn_landmark.txt'
    # save_dir = '48/landmark_image'
    # if os.path.exists(save_dir):
    #     shutil.rmtree(save_dir)
    # os.mkdir(save_dir)
    # with open(box_path, 'r') as f:
    #     boxs = f.readlines()
    # f.close()
    # with open(point_path, 'r') as f:
    #     points = f.readlines()
    # f.close()
    # landmark_file = open(mtcnn_file, 'w')


    # for idx, box in enumerate(boxs):
    #     if idx == 0 or idx == 1:
    #         continue
    #     if idx % 10 == 0:
    #         break
    #     item =  ','.join(box.split())
    #     item = item.split(',')
    #     image_name = item[0]
    #     img_path = os.path.join(image_path, image_name)
    #     if not os.path.exists(img_path):
    #         continue
    #     print(img_path)
    #     img  = misc.imread(img_path, mode='RGB')
    #     height, width, channel = img.shape
    #     item = [int(e) for e in item[1: ]]
    #     xmin = item[0]
    #     ymin = item[1]
    #     xmax = xmin + item[2]
    #     ymax = ymin + item[3]
    #     # pdb.set_trace()
    #     tem = points[idx]
    #     print(tem)
    #     landmarks =  ','.join(tem.split())
    #     landmarks = landmarks.split(',')[1: ]
    #     landmarks = [int(e) for e in landmarks]
    #     # landmarks = np.asarray(landmarks, dtype=np.int32)
       

    #     pos_num = 0
    #     while (pos_num < 5):

    #         size = random.randint(int(min(height, width) * 0.5), np.ceil(1.25 * min(width, height)))
    #         nx1 = random.randint(-xmin * 0.3 + xmin, xmin * 0.3 + xmin)
    #         ny1 = random.randint(-ymin * 0.3 + ymin, ymin * 0.3 + ymin)
    #         nx2 = nx1 + size
    #         ny2 = ny1 + size

    #         if nx2 > width or ny2 > height:
    #             continue

    #         offset_x1 = (xmin - nx1) / size
    #         offset_y1 = (ymin - ny1) / size
    #         offset_x2 = (xmax - nx2) / size
    #         offset_y2 = (ymax - ny2) / size

    #         crop_box = img[ny1: ny2, nx1: nx2]
    #         name = str(uuid.uuid4()) + '.jpg'
    #         misc.imsave(os.path.join(save_dir, name), crop_box)
    #         points = np.asarray(landmarks, dtype=np.int32).reshape(-1, 2)
    #         reg = []
    #         for point in points:
    #             reg.append(point[0] / size - 0.5)
    #             reg.append(point[1] / size - 0.5)
    #         reg = [str(e) for e in reg]
    #         line = ' '.join(reg[0:])
    #         # print(line)
         
            
    #         landmark_file.write(os.path.join(save_dir, name) + " 1 %0.3f %0.3f %0.3f %0.3f "%(offset_x1, offset_y1, offset_x2, offset_y2) + line + '\n')

    #         pos_num += 1
        # break
        

        # for landmark in landmarks:
        #     cv2.circle(img,(landmark[0], landmark[1]), 1, (0,0,255), 2)  

        # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 0), 1)
       
        # misc.imsave(os.path.join('output', image_name), img)

    # landmark_file.close()




        # train_file = '48/mtcnn_landmark.txt'
        # with open(train_file, 'r') as f:
        #     contents = f.readlines()
        # f.close()
        # print(len(contents))
        # for idx, item in enumerate(contents):
        #     line = item.split()

        #     img_path = line[0]
        #     img = misc.imread(img_path, mode='RGB')
        #     point = [float(e) for e in line[2: 6]]
        #     landmarks = [float(e) for e in line[6: ]]
        #     landmarks = np.array(landmarks, dtype=np.float32).reshape(-1, 2)
        #     x1, y1, x2, y2 = process_result(point, img_path, im=None)
        #     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1)
        #     bbox = process_landmark(img_path, landmarks)
        #     for box in bbox:
        #         cv2.circle(img,(box[0], box[1]), 1, (0,0,255), 2) 
                
        #     misc.imsave(os.path.join('48/landmark_image1',str(uuid.uuid4()) + ".jpg"), img)




        # with open('train_file/test.txt', 'r') as f:
        #     contens = f.readlines()

        # for content in contens:
        #     line = content.split(' ')
        #     img = misc.imread(os.path.join('dataset/val_images', line[0]), mode = 'RGB')
        #     cor = line[1: ]
        #     cor = np.asarray(cor, dtype = np.int32).reshape(-1,4)
        #     for line in cor:
        #         cv2.rectangle(img, (line[0], line[1]), (line[2], line[3]), (255, 255, 0), 1)
        #     misc.imsave("draw.jpg", img)


        # input1 = np.random.rand(3,4)
        # input2 = np.random.rand(3,4)
        # k=2
        # # pdb.set_trace()
        # print(input)
        # _, indice = tf.nn.top_k(input, k)
        # # for e in output:
        # #     print(e)
        # # print(cor)
        # # print(output.values)
        # # print(output.indices)
        # print(indice)
        # # print(input[output.indices])