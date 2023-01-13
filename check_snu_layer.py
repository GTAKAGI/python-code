import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
#sys.path.append("/content/drive/My Drive/Colab_Notebooks/Pytorch_test/SNU_PyTorch")
from model import snu_layer

##check_snu_layerと同じ階層にimgファイルを生成。exist_ok=true既に存在しているディレクトリを指定してもエラーにならない##
img_save_dir = "./imgs/"
os.makedirs(img_save_dir, exist_ok=True)
    
""" Build Spiking Neural Unit """
num_time = 100 # simulation time step
V_th = 0.1
tau = 25e-3 # sec
dt = 1e-3 # sec

""" device """
gpu = True
if gpu:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'##if以降の意味##
else:
    device = 'cpu'
#device='cpu'
print(device)

##ここのdt tau
snu_l = snu_layer.SNU(in_channels=1, out_channels=1 ,l_tau=(1-dt/tau),
                      soft=False, initial_bias=-V_th,gpu=gpu)
#snu_l.Wx.W = torch.Tensor(np.array([[1.0]], dtype=np.float32))

""" Generate Poisson Spike Trains """
fr = 100 # Hz
# x = np.where(np.random.rand(1, num_time) < fr*dt, 1, 0)#(1, 100)
x = np.where(np.random.rand(1, num_time) < 0.5, 1, 0)#(1, 100)
x = np.expand_dims(x, 0).astype(np.float32)#(1, 1, 100)
# 入力をnumpy からtensorにしてGPUにのせる
#↑なんで?#

x = torch.from_numpy(x.astype(np.float32)).clone().to(device)
print("x : ",x)
print(x.shape)

s_arr = np.zeros(num_time)
y_arr = np.zeros(num_time) # array to save output

for i in range(num_time):    
    y = snu_l(x[:, :, i]) # (の前にはforwardが隠れてる  
    #print("y : ",y)
    #print("snu_l : ",snu_l.s)

    #s_arr[i] = snu_l.s.array
    s_arr[i] = snu_l.s
    #y_arr[i] = y.array
    y_arr[i] = y

plt.savefig(img_save_dir+"Check_SNU_result.png")

""" Plot results """
plt.figure(figsize=(6,6))

plt.subplot(3,1,1)
plt.title("Spaiking Neural Unit")
plt.plot(x[0,0].to('cpu'))
plt.ylabel("Input")

plt.subplot(3,1,2)
plt.plot(s_arr)
plt.ylabel("Membrane potential")

plt.subplot(3,1,3)
plt.plot(y_arr)
plt.ylabel("Output")
plt.xlabel("Time (ms)")

plt.tight_layout()
plt.show()

plt.savefig(img_save_dir+"Check_SNU_result.png")