import os
import sys
# os.append('./')
import numpy as np
from utils import IoU
import cv2


mtcnn_train_txt = '48/mtcnn_train12.txt'
images_path = 'dataset/images'

# ori_file_12  = open('train_file/ori_neg_train12.txt', 'w')


neg_dir = '48/orineg'
txt_path = '48'
neg_txt = open(os.path.join(txt_path ,'orineg.txt'), 'w')

with open(mtcnn_train_txt,'r') as f:
    labels = f.readlines()
num = len(labels)
print("total train num: " + str(num))

p_idx = 0
n_idx = 0
d_idx = 0
id = 0 #global count
tem = 2e6 / num
for label in labels:
    img_path = os.path.join(images_path, label.split(' ')[0])
    print(img_path)
    # if not os.path.exists(img_path):
    #     continue
    # bbox = [int(float(ele)) for ele in label.strip().split(' ')[1:]]
    bbox = [int(ele) for ele in label.strip().split(' ')[1:]]
    bbox = np.array(bbox, dtype = np.float32).reshape(-1,4)
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    h, w, c = img.shape
    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    neg_num = 0 
    ##generate negative.txt
    while neg_num < (tem + 100):
        size = np.random.randint(40, min(h, w) / 2)
        x_c = np.random.randint(0, w - size)
        y_c = np.random.randint(0, h - size)
        crop_box = np.array([x_c, y_c, x_c + size, y_c + size])
        Iou = IoU(crop_box,bbox)
        # print(Iou)

        corpped_img = img[y_c : y_c + size, x_c : x_c + size, : ]
        print(corpped_img.shape)
        # cv2.imwrite('./1.jpg',corpped_img)
        # resized_crop_img = cv2.resize(corpped_img,(12,12), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:
            # if n_idx == 4000:
            #     neg_txt.close()
            #     break
            neg_txt.write(os.path.join(neg_dir,str(n_idx) + ".jpg 0") + '\n')
            cv2.imwrite(os.path.join(neg_dir,str(n_idx) + ".jpg"),corpped_img)
            # ori_file_12.write(img_path + " " + os.path.join(neg_dir,str(n_idx) + ".jpg 0") + '\n')
            print("neg: " + str(n_idx) + " neg_num: " + str(neg_num))  
            n_idx += 1
            neg_num += 1

    print("images done negative: %s"%(d_idx))
neg_txt.close()
# pos_txt.close()
# par_txt.close()

