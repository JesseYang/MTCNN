import numpy as np
from numpy import random
size = 'val_12'


with open('train_file/neg%s.txt'%(size), 'r') as f:
	neg = f.readlines()

with open('train_file/pos%s.txt'%(size), 'r') as f:
	pos= f.readlines()

with open('train_file/par%s.txt'%(size), 'r') as f:
	par = f.readlines()

with open('train_file/mtcnn12_val.txt', 'w') as f:
	f.writelines(pos)
	neg_keep = random.choice(len(neg), size = 150000, replace = False)
	par_keep = random.choice(len(par), size = 140000, replace = False)

	for i in neg_keep:
		f.write(neg[i])
	for i in par_keep:
		f.write(par[i])