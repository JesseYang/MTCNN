import os
import sys
# os.append('./')
import numpy as np
from utils import IoU
import cv2


mtcnn_train_txt = 'train_file/mtcnn_val12.txt'
images_path = 'dataset/val_images'

neg_dir = '12/negval'
pos_dir = '12/posval'
par_dir = '12/parval'
txt_path = 'train_file'
neg_txt = open(os.path.join(txt_path ,'negval_12.txt'),'w')
pos_txt = open(os.path.join(txt_path ,'posval_12.txt'),'w')
par_txt = open(os.path.join(txt_path ,'parval_12.txt'),'w')

with open(mtcnn_train_txt,'r') as f:
    labels = f.readlines()
num = len(labels)
print("total train num: " + str(num))

p_idx = 0
n_idx = 0
d_idx = 0
id = 0 #global count

for label in labels:
    img_path = os.path.join(images_path, label.split(' ')[0])
    print(img_path)
    # bbox = [int(float(ele)) for ele in label.strip().split(' ')[1:]]
    bbox = [int(ele) for ele in label.strip().split(' ')[1:]]
    bbox = np.array(bbox, dtype = np.float32).reshape(-1,4)
    img = cv2.imread(img_path)
    h, w, c = img.shape
    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    neg_num = 0 
    ##generate negative.txt
    while neg_num < 50:
        size = np.random.randint(40, min(h, w) / 2)
        x_c = np.random.randint(0, w - size)
        y_c = np.random.randint(0, h - size)
        crop_box = np.array([x_c, y_c, x_c + size, y_c + size])
        Iou = IoU(crop_box,bbox)
        # print(Iou)

        corpped_img = img[y_c : y_c + size, x_c : x_c + size, : ]
        # cv2.imwrite('./1.jpg',corpped_img)
        # resized_crop_img = cv2.resize(corpped_img,(12,12), interpolation=cv2.INTER_LINEAR)
        if np.max(Iou) < 0.3:
            neg_txt.write(os.path.join(neg_dir,str(n_idx) + ".jpg 0") + '\n')
            cv2.imwrite(os.path.join(neg_dir,str(n_idx) + ".jpg"),corpped_img)
            n_idx += 1
            neg_num += 1
    for box in bbox:
        #box (xmin, ymin, xmax, ymax)
        x1, y1, x2, y2 = box
        w_ = x2 - x1 + 1
        h_ = y2 - y1 + 1
        ##if img too small ignore it
        if x2 < x1 or y2 < y1:
            continue
        if max(w_, h_) < 40 or x1 <0 or y1 <0:
            continue
    ##generate positive.txt
        for i in range(20):
            # print(i)
            size =np.random.randint(int(min(w_, h_) * 0.8), np.ceil(1.25 * max(w_, h_)))
            #offset to box center
            # print(box)
            # print(w_)
            # print("w,h")
            # print(h_)
            delta_x = np.random.randint(-w_ * 0.2, w_ * 0.2)
            delta_y = np.random.randint(-h_ * 0.2, h_ * 0.2)

            nx1 = max(x1 + w_ / 2 + delta_x - size / 2, 0)
            ny1 = max(y1 + h_ / 2 + delta_y - size / 2, 0)
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > w or ny2 > h:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / size
            offset_y1 = (y1 - ny1) / size
            offset_x2 = (x2 - nx2) / size
            offset_y2 = (y2 - ny2) / size
            corpped_img = img[int(ny1) : int(ny2), int(nx1) : int(nx2), :]
    
            # resized_crop_img = cv2.resize(corpped_img, (12, 12),interpolation=cv2.INTER_LINEAR)
            box_ = box.reshape(1,-1)

            if IoU(crop_box, box_) >= 0.65:
                pos_txt.write(os.path.join(pos_dir, str(p_idx) + '.jpg') + ' 1 %0.3f %0.3f %0.3f %0.3f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(os.path.join(pos_dir, str(p_idx) + '.jpg'), corpped_img)
                p_idx += 1
            elif IoU(crop_box, box_) >= 0.4:
                par_txt.write(os.path.join(par_dir, str(d_idx) + ".jpg") + ' -1 %0.3f %0.3f %0.3f %0.3f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(os.path.join(par_dir, str(d_idx) + ".jpg"), corpped_img)
                d_idx += 1
    print("images done, positive: %s negative: %s part: %s"%(p_idx,n_idx,d_idx))
neg_txt.close()
pos_txt.close()
par_txt.close()

