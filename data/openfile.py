import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.animation as animation
import time
import scipy.io

"""
##label imageを見たいときはコメントアウト外してください

name = scipy.io.loadmat('label/label_100.mat')
print(name)
print(name['name'].shape)

img = name['name']
np.set_printoptions(threshold=np.inf)
print(img)

plt.imshow(img)
plt.show()
"""
#image　を見たいときはコメントアウトを外してください

number = input('何番読み込む？')
path = r'C:\Users\josep\Desktop\SNN\DEM\64pix_(0-3deg)_dem(lidar_noisy)_boulder/image/image_'+number
name = scipy.io.loadmat(path)
print(name)

print(type(name)) 
print(name.keys()) #key(value)
print(name.values()) 
print(name.items())
print(name['time_data'])
print(type(name['time_data']))
print("============")
print(name['time_data'].shape)
print(name['time_data'][:,:,1]) #1層目の64×64ピクセルをすべて表示
img = name['time_data'][:,:,1]
np.set_printoptions(threshold=np.inf) #全表示(省略しない)
print(img)

N = 11
fig, ax = plt.subplots()
def update(i):
    img = name['time_data'][:,:,i]
    
    plt.clf()
    
    plt.imshow(img,cmap='gray')
ani = animation.FuncAnimation(fig, update, np.arange(1,  N), interval=100)  # 代入しないと消される
ani.save(str(path)+'.gif',writer='imagemagick')
plt.show()
