#snu
import numpy as np
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
import time
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
parser.add_argument('--batch', '-b', type=int, default=32)##バッチサイズを指定。指定しない場合32になる。
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
model = network.Unet3_SNU(num_time=args.time,l_tau=0.8,rec=args.rec, forget=args.forget, dual=args.dual, gpu=True, batch_size=args.batch,power=args.power) 
#revisedSNU
#model = network.revisedSNU2_Network(num_time=args.time,l_tau=0.8,rec=args.rec, forget=args.forget, dual=args.dual, gpu=True, batch_size=args.batch, power = args.power)
#model = network.revisedSNU2_Network(num_time=args.time,l_tau=0.8,rec=args.rec, forget=args.forget, dual=args.dual, gpu=True, batch_size=args.batch, power = args.power)
#model = network.Gated_CSNU_Net()

model = model.to(device)
print("building model")
print(model.state_dict().keys())
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = args.epoch


print("パラメータ")
# params = 0
# for p in model.parameters():
#     print(p)
#     if p.requires_grad:
#         params += p.numel()
# print(params)



loss_hist = []
acc_hist_1 = []
acc_hist_2 = []
acc_hist_3 = []
acc_hist_4 = []
acc_hist_5 = []
acc_eval_hist1 = []
acc_eval_hist2 = []
acc_eval_hist3 = []
acc_eval_hist4 = []
acc_eval_hist5 = []

for epoch in tqdm(range(epochs)):
    running_loss = 0.0
    local_loss = []
    
    acc_1 = []
    acc_2 = []
    acc_3 = []
    acc_4 = []
    acc_5 = []
    eval_acc_1 = []
    eval_acc_2 = []
    eval_acc_3 = []
    eval_acc_4 = []
    eval_acc_5 = []

    print("EPOCH",epoch)
    # モデル保存
    if epoch == 0 :
        torch.save(model.state_dict(), "models/models_state_dict_"+str(epoch)+"epochs.pth")
        print("success model saving")
    with tqdm(total=len(train_dataset),desc=f'Epoch{epoch+1}/{epochs}',unit='img')as pbar:


        for i,(inputs, labels, name) in enumerate(train_iter, 0):##0からスタート
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)
            torch.cuda.memory_summary(device=None, abbreviated=False)
            #print("########")
            if args.power:
                loss, pred, _, iou, cnt, total_spike_count = model(inputs, labels)
            else:
                loss, pred, _, iou, cnt = model(inputs, labels)
            
            #summary(Unet3_SNU,)
            #iou = 各発火閾値ごとに連なり[??(i=1),??(i=2),,,,]
            pred,_ = torch.max(pred,1)
            #print('IoU : ',iou)      
            acc_1.append(iou[0]) #spike 
            acc_2.append(iou[1])
            acc_3.append(iou[2])
            acc_4.append(iou[3])
            acc_5.append(iou[4])
    
            torch.autograd.set_detect_anomaly(True)
            # print(7777777777777777)
            loss.backward(retain_graph=True)
            # print(66666666666)
            running_loss += loss.item()
            local_loss.append(loss.item())
            del loss
            optimizer.step()

            # print statistics
            
            
            if i % 100 == 99:
                print('[{:d}, {:5d}] loss: {:.3f}'
                            .format(epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    
    with torch.no_grad():
        for i,(inputs, labels, name) in enumerate(test_iter, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            loss, pred, _, iou, cnt,total_spike_count = model(inputs, labels)
            pred,_ = torch.max(pred,1)
            eval_acc_1.append(iou[0])
            eval_acc_2.append(iou[1])
            eval_acc_3.append(iou[2])
            eval_acc_4.append(iou[3])
            eval_acc_5.append(iou[4])
            break
    
    mean_acc_1 = np.mean(acc_1)
    mean_acc_2 = np.mean(acc_2)
    mean_acc_3 = np.mean(acc_3)
    mean_acc_4 = np.mean(acc_4)
    mean_acc_5 = np.mean(acc_5)
    
    mean_eval_acc_1 = np.mean(eval_acc_1)
    mean_eval_acc_2 = np.mean(eval_acc_2)
    mean_eval_acc_3 = np.mean(eval_acc_3)
    mean_eval_acc_4 = np.mean(eval_acc_4)
    mean_eval_acc_5 = np.mean(eval_acc_5)
    
    print("mean iou 3:4:5:6:7 ",mean_eval_acc_1,mean_eval_acc_2,mean_eval_acc_3,mean_eval_acc_4,mean_eval_acc_5,sep='--')
    acc_hist_1.append(mean_acc_1)
    acc_hist_2.append(mean_acc_2)
    acc_hist_3.append(mean_acc_3)
    acc_hist_4.append(mean_acc_4)
    acc_hist_5.append(mean_acc_5)
    
    acc_eval_hist1.append(mean_eval_acc_1)
    acc_eval_hist2.append(mean_eval_acc_2)
    acc_eval_hist3.append(mean_eval_acc_3)
    acc_eval_hist4.append(mean_eval_acc_4)
    acc_eval_hist5.append(mean_eval_acc_5)
    
    mean_loss = np.mean(local_loss)
    print("mean loss",mean_loss)
    loss_hist.append(mean_loss)

# ログファイル二セーブ
path_w = 'train_dataset_log.txt'
with open(path_w, mode='w') as f:
# <class '_io.TextIOWrapper'>
    f.write(name[data_id])
#lossの可視化

fig = plt.figure(facecolor='oldlace')
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)
ax1.set_title('loss')
ax1.plot(loss_hist)
ax1.set_xlabel('EPOCH')
ax1.set_ylabel('LOSS')

ax2.set_title('train IOU')
ax2.set_ylabel('Accuracy [%]')
ax2.set_xlabel('EPOCH')
ax2.grid()
ax2.set_ylim(bottom=0, top=100)
#ax2.set_xticks(np.arange(0,epochs,10))
ax2.set_yticks(np.arange(0,101,10))
ax2.plot(acc_hist_1,label='[2]')
ax2.plot(acc_hist_2,label='[3]')
ax2.plot(acc_hist_3,label='[4]')
ax2.plot(acc_hist_4,label='[5]')
ax2.plot(acc_hist_5,label='[6]')
ax2.legend(loc=0)

ax3.set_title('Evaluate IOU')
ax3.grid()
ax3.set_ylim(bottom=0, top=1)
#ax3.set_xticks(0,epochs,10)
ax3.set_yticks(np.arange(0.1,1.1,0.1))
ax3.plot(acc_eval_hist1,label='30[2]')
ax3.plot(acc_eval_hist2,label='40[3]')
ax3.plot(acc_eval_hist3,label='50[4]')
ax3.plot(acc_eval_hist4,label='60[5]')
ax3.plot(acc_eval_hist5,label='70[6]')
fig.tight_layout()

fig.savefig('models/loss--IOU.jpg')
tle = (time.time()- start_time)/3600
plt.show()


torch.save(model.state_dict(), "models/models_state_dict_end.pth")
 # モデル読み込み
print("success model saving")
print()
print(total_spike_count)
print()
print('{:.03f}'.format(tle))

