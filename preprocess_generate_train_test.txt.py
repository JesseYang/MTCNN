import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import uuid

file_path = 'dataset/val_images'
##generate train.txt
# train_file = 'dataset/annotations/wider_face_train_bbx_gt.txt'
####generate test.txt
train_file = 'dataset/annotations/wider_face_val_bbx_gt.txt'

#train.txt result
# mtcnn_train_txt = 'train_file/mtcnn_train12.txt'


#test.txt result
mtcnn_train_txt = 'train_file/mtcnn_val12.txt'


with open(train_file) as f:
    items = f.readlines()
f.close()
size = len(items)
count = 0
k = 0
record = []
while k < size:
    img_path = os.path.join(file_path,items[k].strip())
    print(img_path)
    line = items[k].strip()
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_num = int(items[k+1].strip())
    i = 1
    coors =[]
    while i <= img_num:
        xmin = int(items[k+1+i].strip().split(' ')[0])
        ymin = int(items[k+1+i].strip().split(' ')[1])
        xmax = int(items[k+1+i].strip().split(' ')[2]) + xmin
        ymax = int(items[k+1+i].strip().split(' ')[3]) + ymin
        # result = cv2.rectangle(img,(int(xmin), int(ymin)),
        #                                        (int(xmax), int(ymax)),
        #                                        (255,0,0),
        #                                       3)
        i +=1
        coors.append([xmin, ymin, xmax, ymax])
    coors = [str(ele[0]) + " " + str(ele[1]) + " " +str(ele[2]) + " " + str(ele[3]) for ele in coors]
#     print(coors)
    coors = ' '.join(coors)
    
    re = line + " " + coors + "\n"
    record.append(re)
    # cv2.imwrite(os.path.join('/home/yjf/Downloads/mtcnn/result', str(uuid.uuid4()) +'.jpg'),result)
    
    k = img_num +2 +k
    count += 1
#     if count %10 == 0:
#         break
#         print(count)

print(len(record))

f = open(mtcnn_train_txt, 'w')
for i in record:
    f.write(i)
f.close()