import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.animation as animation
import time
import scipy.io

"""
#### label imageを見たいときはコメントアウト外してください

name = scipy.io.loadmat('label/label_100.mat')
print(name)
print(name['name'].shape)

img = name['name']
np.set_printoptions(threshold=np.inf)
print(img)

plt.imshow(img)
plt.show()
"""
##### image　を見たいときはコメントアウトを外してください

number = input('何番読み込む？')
path = r'C:\Users\josep\Desktop\SNN\DEM\64pix_(0-3deg)_dem(lidar_noisy)_boulder/label/label_'+number
name = scipy.io.loadmat(path)
print(name)

print(type(name))
print(name.keys())
# print(name.values())
# print(name.items())
print(name['label_data'])
print(type(name['label_data']))
print("============")
print(name['label_data'].shape)
print(name['label_data'][:,:])
img = name['label_data'][:,:]
np.set_printoptions(threshold=np.inf)
print(img)
