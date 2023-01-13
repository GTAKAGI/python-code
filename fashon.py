
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn
import torchvision

print(torch.__version__)
# The coarse network structure is dicated by Fashion MNIST dataset
img = np.zeros((512,512))
plt.imshow(img)
plt.show()

for i in range(512):
    img[i] = 255
    plt.imshow(img)
    plt.show()


"""
nb_inputs = 28*28
nb_hidden = 100
nb_outputs = 10

time_step = 1e-3
nb_steps = 100

batch_size = 256

dtype= torch.float
# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")
# Here we load the Dataset
root = os.path.expanduser("data/datasets/torch/fashion-mnist")
train_dataset = torchvision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root, train=False, transform=None, target_transform=None, download=True)
# Standardize data
# x_train = torch.tensor(train_dataset. train_data, device=device, dtype=dtype)
x_train = np.array(train_dataset.data, dtype=np.float)
x_train = x_train.reshape(x_train.shape[0],-1)/255
# x_test = torch.tensor(test_dataset, test_data, device=device, dtype=dtype)
x_test = np.array(test_dataset.data, dtype=np.float)
x_test = x_test.reshape(x_test.shape[0],-1)/255

# y_train = torch.tensor(train_dataset, train_labels, device=device, dtype=dtype)
# y_test = torch.tensor(test_dataset, test_labels, device=device, dtype=dtype)
y_train = np.array(train_dataset.targets, dtype=np.int)
y_test = np.array(test_dataset.targets, dtype=np.int)

data_id = 5
plt.imshow(x_train[data_id].reshape(28,28), cmap=plt.cm.gray_r)
plt.axis("off")
plt.show()

def current2firing_time(x, tau=20, thr=0.2, tmax=1.0, epsilon=1e-7):
    #Compare first firing time latency for a current input x assuming the charge time of a current based LIF neuron

    #Args:
    #x--The "current" values
    

    idx = x<thr
    x = np.clip(x, thr+epsilon,1e9)
    T = tau*np.log(x/x-thr)
    T[idx]=tmax
    return T

T = current2firing_time(x_train[data_id])
print(T)
plt.imshow(T.reshape(28,28), cmap=plt.cm.gray_r)
plt.axis("off")
plt.show()

x_data = x_train
y_data = y_train
x = x_data
y = y_data

labels_ = np.array(y,dtype=np.int)
number_of_batches = len(x)//batch_size
sample_index = np.arange(len(x))

# compute discrete firing times
tau_eff = 20e-3/time_step
firing_times = np.array(current2firing_time(x,tau=tau_eff,tmax=nb_steps),dtype=np.int)
#unit_numbers = np.arange(nb_units)

print(firing_times.shape)
"""
