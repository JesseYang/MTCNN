import numpy as np
from numpy import random
size = 'val_24'


with open('24/neg%s.txt'%(size), 'r') as f:
	neg = f.readlines()
	print(len(neg))

with open('24/pos%s.txt'%(size), 'r') as f:
	pos= f.readlines()
	print(len(pos))

with open('24/par%s.txt'%(size), 'r') as f:
	par = f.readlines()

with open('24/mtcnn24_val.txt', 'w') as f:
	f.writelines(pos)
	neg_keep = random.choice(len(neg), size = 19000, replace = False)
	par_keep = random.choice(len(par), size = 36000, replace = False)

	for i in neg_keep:
		f.write(neg[i])
	for i in par_keep:
		f.write(par[i])