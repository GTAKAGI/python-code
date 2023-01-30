##使い方
## python analysis.py --b 1 --d --p →　heatmap

#snu
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import os 
from tqdm import tqdm
from torchsummary import summary
# from rectangle_builder import rectangle,test_img
import sys
sys.path.append("C:/Users/josep/Desktop/SNN/pytorch_test/snu")
from model import snu_layer
from model import network
from tqdm import tqdm
#from mp4_rec import record, rectangle_record
import pandas as pd
import scipy.io
from torchsummary import summary
import argparse
import matplotlib.ticker as ticker
import time
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
start_time = time.time()
######XXXX##############################################################
class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        #self.data_transform = data_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
     
        name = self.df['id'][i]
        image = scipy.io.loadmat(self.df['id'][i])
        label = scipy.io.loadmat(self.df['label'][i])

        image = image['time_data']
        label = label['label_data']
        # print("image : ",image.shape)
        #image = image.reshape(4096,20)
        image = image.reshape(4096,11)##64pix × 64pix, 11step?
      
        #print("image : ",image.shape)
        image = image.astype(np.float32)
        #label = label.astype(np.int64)
        #label = torch.tensor(label,dtype =torch.int64 )
        label = label.reshape(4096)##64pix × 64pix
        label = label.astype(np.float32)
        return image, label, name##
#################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--batch', '-b', type=int, default=1)##バッチサイズを指定。指定しない場合32になる。
parser.add_argument('--epoch', '-e', type=int, default=100)##エポック数を指定。指定しない場合100になる。
parser.add_argument('--time', '-t', type=int, default=11,
                        help='Total simulation time steps.')##ステップ数を指定?
parser.add_argument('--rec', '-r', action='store_true' ,default=False)  # -r付けるとTrue                  

parser.add_argument('--forget', '-f', action='store_true' ,default=False) 
parser.add_argument('--dual', '-d', action='store_true' ,default=False)
parser.add_argument('--power', '-p', action='store_true' ,default=False)
args = parser.parse_args()


print("***************************")##出力
train_dataset = LoadDataset("data/csv_data/semantic_train_loc.csv")##教師用データ
test_dataset = LoadDataset("data/csv_data/semantic_eval_loc.csv")##評価用データ
data_id = 2
#print(train_dataset[data_id][0]) #(784, 100) 
train_iter = DataLoader(train_dataset, batch_size=args.batch, shuffle=False)
test_iter = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

# ネットワーク設計
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 畳み込みオートエンコーダー　リカレントSNN　
#Unet2_SNU
model = network.Unet3_SNU(num_time=args.time,l_tau=0.8,rec=args.rec, forget=args.forget, dual=args.dual, gpu=True, batch_size=args.batch ,power=args.power, heatmap=True) 
#revisedSNU
model_hide = network.revisedSNU2_Network(num_time=args.time,l_tau=0.8,rec=args.rec, forget=args.forget, dual=args.dual, gpu=True, batch_size=args.batch,power = args.power, heatmap = True )
#model = network.Gated_CSNU_Net()

model = model.to(device)
model_path = f'models/gaku2.pth'
model.load_state_dict(torch.load(model_path))


# 秀さんのモデル
model_hide = model_hide.to(device)
model_path = f'models/revised-SNU.pth'
model_hide.load_state_dict(torch.load(model_path))
 
#こっちは指定した番号を見れる
# while 1:
#     n = int(input("何番目のデータ見たい？"))
#     with torch.no_grad():
#         for i, (inputs, labels, name) in enumerate(test_iter):
#             # print(i)
#             if i == n:
#                 print("ok")
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#                 loss, pred, _, iou, cnt, total_spike_count, heat = model(inputs, labels)
#                 pred,_ = torch.max(pred,1)
#                 break
#     labels = labels.to('cpu').detach().numpy().copy()
#     heat = heat.to('cpu').detach().numpy().copy()
#     labels = np.reshape(labels,(64,64))
#     heat = np.reshape(heat,(64,64))
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     sns.heatmap(labels, ax=ax1)
#     sns.heatmap(heat, ax=ax2)
#     plt.tight_layout()
#     plt.show()

##こっちは1から順に結果を見れる
num_test_data = len(test_iter)
ious = {}
ious['my'] = [0]* 12
print(type(ious))
ious['hide'] = [0]* 12
    # n = int(input("何番目のデータ見たい？"))
with torch.no_grad():
    for i, (inputs, labels, name) in enumerate(tqdm(test_iter)):
        # print(i)
        # if i == n:
        #     print("ok")
        inputs = inputs.to(device)
        labels = labels.to(device)
        loss, pred, _, iou, cnt, total_spike_count, heat = model(inputs, labels)
        loss, pred, _, iou_hide, cnt, total_spike_count, heat_hide = model_hide(inputs, labels)
        pred,_ = torch.max(pred,1)

        for idx, score in enumerate(iou):
            #print(score)
            ious['my'][idx] += score
        for idx, score in enumerate(iou_hide):
            #print(score)
            ious['hide'][idx] += score
        labels = labels.to('cpu').detach().numpy().copy()
        heat = heat.to('cpu').detach().numpy().copy()
        labels = np.reshape(labels,(64,64))
        ##スパイクの閾値を設定。今回は4本スパイク発生した場合に危険のピクセルと判定。無くすと
        heat = np.where(heat >= 4,1,0)
        heat = np.reshape(heat,(64,64))
        # print('##########')
        # print((heat))　→　各ピクセルのスパイク列を記述
        # print(type(heat))　→　スパイクの型。ndarray
        # print('##########')
        heat_hide = heat_hide.to('cpu').detach().numpy().copy()
        ##スパイクの閾値を設定。今回は4本スパイク発生した場合に危険のピクセルと判定
        heat_hide = np.where(heat_hide >= 4,1,0)
        heat_hide = np.reshape(heat_hide,(64,64))

##ヒートマップ系##
        # # fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
        # # fig = plt.figure(figsize=(4,4))
        # # ax1 = fig.add_subplot(1,3,1)
        # # ax1.set_title("Input")
        # # ax2 = fig.add_subplot(1,3,2)
        # # ax2.set_title("U-Net heatmap")
        # # ax3 = fig.add_subplot(1,3,3)
        # # ax3.set_title('revised -SNU heatmap')
        # # # ax1.set_aspect('equal')
        # # # ax2.set_aspect('equal')
        # # # ax3.set_aspect('equal')
        # # sns.heatmap(labels, ax=ax1, cmap='CMRmap')
        # # sns.heatmap(heat, ax=ax2, cmap='jet')
        # # sns.heatmap(heat_hide, ax=ax3, cmap='jet')

        # fig,(ax1,ax2,ax3) = plt.subplots(1,3, constrained_layout = True)
        # #fig,(ax1) = plt.subplots(1,1, constrained_layout = True)

        # # ax1 = fig.add_subplot(1,3,1)
        # # ax2 = fig.add_subplot(1,3,2)
        # # ax3 = fig.add_subplot(1,3,3)
        # # ax1.set_ylim(0,65)
        # # ax1.set_xlim(0,65)
        # ax1.set_title("Ground Truth")
        # ax1.set_xticks([])
        # ax1.set_yticks([])

        # # ax2.set_ylim(0,65)
        # # ax2.set_xlim(0,65)
        # # ax2.set_xticks(np.arange(0,64,16))
        # # ax2.set_yticks(np.arange(0,64,16))
        # ax2.set_title("revised-SNU heatmap")
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        # # ax3.set_ylim(0,65)
        # # ax3.set_xlim(0,65)
        # ax3.set_title('U-Net heatmap')
        # ax3.set_xticks([])
        # ax3.set_yticks([])

        # # ax1.set_aspect('equal')
        # # ax2.set_aspect('equal')
        # # ax3.set_aspect('equal')

        # im1 = ax1.imshow(labels,aspect = 'equal',cmap = 'gray')
        # im2 = ax2.imshow(heat_hide,aspect = 'equal')
        # im3 = ax3.imshow(heat,aspect = 'equal')

        # divider1 = make_axes_locatable(ax1)
        # cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        # fig.colorbar(im1, ax=ax1, cax=cax1,ticks = [0,1])
        
        # divider2 = make_axes_locatable(ax2)
        # cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        # fig.colorbar(im2, ax=ax2, cax=cax2, ticks = [0,1])

        # divider3 = make_axes_locatable(ax3)
        # cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        # fig.colorbar(im3, ax=ax3, cax=cax3, ticks = [0,1])

        # filename = "heatmap_fig\heat_"+str(i+1)+".png"
        # #plt.savefig(filename)
        # fig.tight_layout()
        # plt.show()
#############

        #if i == 3:break
        #print('#####')
# print((ious['my']))
# print((ious['hide']))

##IOU比較##
for key in ious.keys():
    ious[key] = list(map(lambda x: (x/num_test_data)*100, ious[key]))
my_max_iou = max(ious['my'])
hide_max_iou = max(ious['hide'])
##object型##
x = [0,1,2,3,4,5,6,7,8,9,10,11]
y1 = ious['my']#my_ious
y2 = ious['hide']
plt.plot(x,y1)
plt.plot(x,y2)
plt.legend(['U-Net'],['Encoder-Decoder'])
plt.show()
#print(type(ious['hide']))#hide_ious,class 'list

# plt.plot(ious['my'])
# plt.plot(ious['hide'])
# plt.xlabel('number of spikes for hazard threshold')
# plt.ylabel('Accuracy [%]')
# plt.xlim(0,11)
# plt.ylim(0,100)
# plt.xticks(np.arange(0,12,1))
# plt.yticks(np.arange(0,101,10))
# plt.vlines(4,ymin=0,ymax=my_max_iou,colors='red')
# plt.hlines(my_max_iou,xmin=0,xmax=4,colors='red')
# plt.hlines(hide_max_iou,xmin=0,xmax=4,colors='red')
# plt.legend()
# print('DONE')
# print(my_max_iou,hide_max_iou)
#plt.show()
###