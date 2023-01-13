# -*- coding: utf-8 -*-
 
from cmath import pi
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from . import snu_layer

import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt


# Network definition]
##########対称型#########(12/4～)
class Unet3_SNU(torch.nn.Module):
    def __init__(self, num_time=20, l_tau=0.8, soft=False, rec=False, forget=False, dual=False, power = False, gpu=True,
                 batch_size=32, heatmap = False):
        super().__init__()

        
        self.num_time = num_time
        self.batch_size = batch_size
        self.rec = rec
        self.forget = forget
        self.dual = dual
        self.power = power
        self.heatmap = heatmap

         # Encoder layers
        self.c1 = 1
        self.c2 = 4
        self.c3 = 16
        self.c4 = 20
        self.c5 = 17
        self.c6 = 8
        ##self.l1～４にはsnu_layer.Conv_SNUのdef y(スパイク出力)が入る
        self.l1 = snu_layer.Conv_SNU(in_channels=self.c1, out_channels=self.c3, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l2 = snu_layer.Conv_SNU(in_channels=self.c3, out_channels=self.c2, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l3 = snu_layer.Conv_SNU(in_channels=self.c2, out_channels=self.c3, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l4 = snu_layer.Conv_SNU(in_channels=self.c4, out_channels=self.c1, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l5 = snu_layer.Conv_SNU(in_channels=self.c5, out_channels=self.c1, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        # self.l6 = snu_layer.Conv_SNU(in_channels=self.c5, out_channels=self.c1, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        #self.l7 = snu_layer.Conv_SNU(in_channels=self.c3, out_channels=self.c4, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        #self.l8 = snu_layer.Conv_SNU(in_channels=self.c5, out_channels=self.c1, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)

        self.up_samp = nn.Upsample(scale_factor=2, mode='nearest')

    def _reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()
        self.l5.reset_state()
        # self.l6.reset_state()
        #self.l7.reset_state()
        #self.l8.reset_state()

    def iou_score(self, outputs, labels):
        smooth = 1e-6
        outputs = outputs.data.cpu().numpy() #outputs.shape: (128, 1, 64, 64)
        labels = labels.data.cpu().numpy() #labels.shape: (128, 1, 64, 64)
        np.set_printoptions(threshold=np.inf)
        outputs = outputs.squeeze(1) # BATCH*1*H*W => BATCH*H*W __outputs.shape : (128, 64, 64)
        labels = labels.squeeze(1) #__labels.shape : (128, 64, 64)
        #print("outputs : ",outputs)
        iou = []
        cnt = []
        ########iouの計算#####
        for i in range(1,6):
            output = np.where(outputs>i,1,0)
            label = np.where(labels>0,1,0)
            intersection = (np.uint64(output) & np.uint64(label)).sum((1,2)) # will be zero if Trueth=0 or Prediction=0
            union = (np.uint64(output) | np.uint64(label)).sum((1,2)) # will be zero if both are 0
        
            iou.append((intersection + smooth) / (union + smooth))
            cnt.append(i)
        
        return iou,cnt
        ######################

    def forward(self, x, y):#cuda 09
        pixel = 64
        loss = None
        correct = 0
        sum_out = None
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        conv_8 = torch.zeros((self.batch_size, 1, pixel, pixel), device=device, dtype=dtype)
        out_rec = [conv_8]##最後の層を引数に指定
        self._reset_state()
        ht_spike_count = 0
        conv_1_spike_count = 0
        conv_2_spike_count = 0
        conv_3_spike_count = 0
        conv_4_spike_count = 0
        conv_5_spike_count = 0
        #conv_6_spike_count = 0
        #conv_7_spike_count = 0
        #conv_8_spike_count = 0

        ####skip connection(2層結合ver)GPU必須#####
        for t in range(self.num_time):
            # print(f'{x.shape=}')#x.shape=torch.Size([32, 4096, 11])
            x_t = x[:,:,t]  ##各ステップ数の64pix × 64pixをx_tに入れる。
            # print(f'{x_t.shape=}')#x_t.shape=torch.Size([32, 4096])
            # x_t = x_t.reshape((len(x_t), 1, 128, 128))
            x_t = x_t.reshape((len(x_t), 1, pixel, pixel))
            conv_1 = self.l1(x_t)  
            pool_1 = F.max_pool2d(conv_1, 2)
            conv_2 = self.l2(pool_1)
            pool_2 = F.max_pool2d(conv_2, 2)
            conv_3 = self.l3(pool_2) 
            unpool_1 = self.up_samp(conv_3) 
            concat_1 = torch.cat([conv_2, unpool_1],dim=1) 
            conv_4 = self.l4(concat_1)
            unpool_2 = self.up_samp(conv_4)           
            concat_2 = torch.cat([conv_1,unpool_2],dim=1)
            conv_5 = self.l5(concat_2)
            out_rec.append(conv_5)

            if self.power:
                ht_spike_count += torch.sum(x_t)
                conv_1_spike_count += torch.sum(conv_1)
                conv_2_spike_count += torch.sum(conv_2)
                conv_3_spike_count += torch.sum(conv_3)
                conv_4_spike_count += torch.sum(conv_4)
                conv_5_spike_count += torch.sum(conv_5)
                # conv_6_spike_count += torch.sum(conv_6)
                #conv_7_spike_count += torch.sum(conv_7)
                #conv_8_spike_count += torch.sum(conv_8)
            
        total_spike_count =[ht_spike_count,conv_1_spike_count,conv_2_spike_count,conv_3_spike_count,conv_4_spike_count,conv_5_spike_count]#conv_7_spike_count,conv_8_spike_count]
        out_rec = torch.stack(out_rec,dim=1)##時間軸を追加dim=1
        #print(f'{out_rec.shape=}') #out_rec.shape=torch.Size([32, 12, 1, 64, 64])12個の画像データをstackでテンソル連結する。stackで結合する理由はtime data方向で結合するため。(時間軸要素を追加する必要がある)
        #print("out_rec.shape",out_rec.shape) #out_rec.shape torch.Size([128, 21, 1, 64, 64]) ([バッチ,時間,分類])
        #m,_=torch.sum(out_rec,1)
        ##発火スパイクの足し合わせ↓##
        m =torch.sum(out_rec,1) #m.shape: torch.Size([256, 10]) for classifiartion
        heat = m
        #m = m/self.num_time
        # m : out_rec(21step)を時間軸で積算したもの
        # 出力mと教師信号yの形式を統一する
        y = y.reshape(len(x_t), 1, pixel, pixel)
        #m = torch.where(m>0,1,0).to(torch.float32)
        #y = torch.where((y>0)&(y<2),self.num_time//2,0).to(torch.float32)
        y = torch.where(y>0,self.num_time,0).to(torch.float32)
        #criterion = nn.CrossEntropyLoss() #MNIST 
        criterion = nn.MSELoss() # semantic segmantation
        loss = criterion(m, y)
        
        #metabolic_cost = self.gamma*torch.sum(m**3)
        #print("MSE_loss : metabplic_cost = ",loss,metabolic_cost)
        #loss += metabolic_cost
        iou,cnt= self.iou_score(m, y)
        if self.heatmap:
            return loss, m, out_rec, iou, cnt, total_spike_count, heat
        if self.power:
            return loss, m, out_rec, iou, cnt, total_spike_count
        else:
            return loss, m, out_rec, iou, cnt
########################(消費電力計算ver)
##U-net_2層結合ver(10/24～)[notification:batchsize 16 only work]
class Unet2a_SNU(torch.nn.Module):
    def __init__(self, num_time=20, l_tau=0.8, soft=False, rec=False, forget=False, dual=False, power = False, gpu=False,
                 batch_size=32):
        super().__init__()

        
        self.num_time = num_time
        self.batch_size = batch_size
        self.rec = rec
        self.forget = forget
        self.dual = dual
        self.power = power

        ##Unet - SNU##(Unet-SNUの結果を見たい場合はここ)
         # Encoder layers
        self.c1 = 1
        self.c2 = 16
        self.c3 = 32
        self.c4 = 64
        self.c5 = 128
        ##self.l1～４にはsnu_layer.Conv_SNUのdef y(スパイク出力)が入る
        self.l1 = snu_layer.Conv_SNU(in_channels=self.c1, out_channels=self.c4, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l2 = snu_layer.Conv_SNU(in_channels=self.c4, out_channels=self.c2, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        
        # Decoder layers
        self.l3 = snu_layer.Conv_SNU(in_channels=self.c2, out_channels=self.c2, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l4 = snu_layer.Conv_SNU(in_channels=self.c3, out_channels=self.c4, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        #2層結合の場合はself.l5をコメント外す
        self.l5 = snu_layer.Conv_SNU(in_channels=self.c5, out_channels=self.c1, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        
        self.up_samp = nn.Upsample(scale_factor=2, mode='nearest')

    def _reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()
        self.l5.reset_state()

    def iou_score(self, outputs, labels):
        smooth = 1e-6
        outputs = outputs.data.cpu().numpy() #outputs.shape: (128, 1, 64, 64)
        labels = labels.data.cpu().numpy() #labels.shape: (128, 1, 64, 64)
        np.set_printoptions(threshold=np.inf)
        outputs = outputs.squeeze(1) # BATCH*1*H*W => BATCH*H*W __outputs.shape : (128, 64, 64)
        labels = labels.squeeze(1) #__labels.shape : (128, 64, 64)
        #print("outputs : ",outputs)
        iou = []
        cnt = []
        ########iouの計算#####
        for i in range(1,6):
            output = np.where(outputs>i,1,0)
            label = np.where(labels>0,1,0)
            intersection = (np.uint64(output) & np.uint64(label)).sum((1,2)) # will be zero if Trueth=0 or Prediction=0
            union = (np.uint64(output) | np.uint64(label)).sum((1,2)) # will be zero if both are 0
        
            iou.append((intersection + smooth) / (union + smooth))
            cnt.append(i)
        
        return iou,cnt
        ######################

    def forward(self, x, y):#cuda 09
        pixel = 64
        loss = None
        correct = 0
        sum_out = None
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        conv_5 = torch.zeros((self.batch_size, 1, pixel, pixel), device=device, dtype=dtype)
        out_rec = [conv_5]##最後の層を引数に指定
        self._reset_state()
        ht_spike_count = 0
        conv_1_spike_count = 0
        conv_2_spike_count = 0
        conv_3_spike_count = 0
        conv_4_spike_count = 0
        conv_5_spike_count = 0

        #####skip connection(2層結合ver)GPU必須#####
        for t in range(self.num_time):
            # print(f'{x.shape=}')#x.shape=torch.Size([32, 4096, 11])
            x_t = x[:,:,t]  ##各ステップ数の64pix × 64pixをx_tに入れる。
            # print(f'{x_t.shape=}')#x_t.shape=torch.Size([32, 4096])
            # x_t = x_t.reshape((len(x_t), 1, 128, 128))
            x_t = x_t.reshape((len(x_t), 1, pixel, pixel))
            conv_1 = self.l1(x_t)  
            pool_1 = F.max_pool2d(conv_1, 2)
            conv_2 = self.l2(pool_1) 
            pool_2 = F.max_pool2d(conv_2, 2) 
            conv_3 = self.l3(pool_2) 
            unpool_1 = self.up_samp(conv_3) 
            concat_1 = torch.cat([conv_2, unpool_1],dim=1) 
            conv_4 = self.l4(concat_1) 
            unpool_2 = self.up_samp(conv_4)           
            concat_2 = torch.cat([conv_1,unpool_2],dim=1)
            conv_5 = self.l5(concat_2)
            out_rec.append(conv_5)
        ####################################
            if self.power:
                ht_spike_count += torch.sum(x_t)
                conv_1_spike_count += torch.sum(conv_1)
                conv_2_spike_count += torch.sum(conv_2)
                conv_3_spike_count += torch.sum(conv_3)
                conv_4_spike_count += torch.sum(conv_4)
                conv_5_spike_count += torch.sum(conv_5)
            
            total_spike_count = [ht_spike_count,conv_1_spike_count,conv_2_spike_count,conv_3_spike_count,conv_4_spike_count,conv_5_spike_count]




        
        out_rec = torch.stack(out_rec,dim=1)##時間軸を追加dim=1
        #print(f'{out_rec.shape=}') #out_rec.shape=torch.Size([32, 12, 1, 64, 64])12個の画像データをstackでテンソル連結する。stackで結合する理由はtime data方向で結合するため。(時間軸要素を追加する必要がある)
        #print("out_rec.shape",out_rec.shape) #out_rec.shape torch.Size([128, 21, 1, 64, 64]) ([バッチ,時間,分類])
        #m,_=torch.sum(out_rec,1)
        ##発火スパイクの足し合わせ↓##
        m =torch.sum(out_rec,1) #m.shape: torch.Size([256, 10]) for classifiartion
        #m = m/self.num_time
        # m : out_rec(21step)を時間軸で積算したもの
        # 出力mと教師信号yの形式を統一する
        y = y.reshape(len(x_t), 1, pixel, pixel)
        #m = torch.where(m>0,1,0).to(torch.float32)
        #y = torch.where((y>0)&(y<2),self.num_time//2,0).to(torch.float32)
        y = torch.where(y>0,self.num_time,0).to(torch.float32)
        #criterion = nn.CrossEntropyLoss() #MNIST 
        criterion = nn.MSELoss() # semantic segmantation
        loss = criterion(m, y)
        
        #metabolic_cost = self.gamma*torch.sum(m**3)
        #print("MSE_loss : metabplic_cost = ",loss,metabolic_cost)
        #loss += metabolic_cost
        iou,cnt= self.iou_score(m, y)
        if self.power:
            return loss, m, out_rec, iou, cnt, total_spike_count
        else:
            return loss, m, out_rec, iou, cnt
######################################################################################################################
########################
##U-net_2層結合ver(10/24～)[notification:batchsize 16 only work]
class Unet2_SNU(torch.nn.Module):
    def __init__(self, num_time=20, l_tau=0.8, soft=False, rec=False, forget=False, dual=False, gpu=True,
                 batch_size=32):
        super().__init__()

        
        self.num_time = num_time
        self.batch_size = batch_size
        self.rec = rec
        self.forget = forget
        self.dual = dual

        ##Unet - SNU##(Unet-SNUの結果を見たい場合はここ)
         # Encoder layers
        self.c1 = 1
        self.c2 = 16
        self.c3 = 32
        self.c4 = 8
        self.c5 = 64
        ##self.l1～４にはsnu_layer.Conv_SNUのdef y(スパイク出力)が入る
        self.l1 = snu_layer.Conv_SNU(in_channels=self.c1, out_channels=self.c2, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l2 = snu_layer.Conv_SNU(in_channels=self.c2, out_channels=self.c3, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        
        # Decoder layers
        self.l3 = snu_layer.Conv_SNU(in_channels=self.c3, out_channels=self.c3, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l4 = snu_layer.Conv_SNU(in_channels=self.c5, out_channels=self.c2, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        #2層結合の場合はself.l5をコメント外す
        self.l5 = snu_layer.Conv_SNU(in_channels=self.c3, out_channels=self.c1, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        
        self.up_samp = nn.Upsample(scale_factor=2, mode='nearest')

    def _reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()
        self.l5.reset_state()

    def iou_score(self, outputs, labels):
        smooth = 1e-6
        outputs = outputs.data.cpu().numpy() #outputs.shape: (128, 1, 64, 64)
        labels = labels.data.cpu().numpy() #labels.shape: (128, 1, 64, 64)
        np.set_printoptions(threshold=np.inf)
        outputs = outputs.squeeze(1) # BATCH*1*H*W => BATCH*H*W __outputs.shape : (128, 64, 64)
        labels = labels.squeeze(1) #__labels.shape : (128, 64, 64)
        #print("outputs : ",outputs)
        iou = []
        cnt = []
        ########iouの計算#####
        for i in range(1,6):
            output = np.where(outputs>i,1,0)
            label = np.where(labels>0,1,0)
            intersection = (np.uint64(output) & np.uint64(label)).sum((1,2)) # will be zero if Trueth=0 or Prediction=0
            union = (np.uint64(output) | np.uint64(label)).sum((1,2)) # will be zero if both are 0
        
            iou.append((intersection + smooth) / (union + smooth))
            cnt.append(i)
        
        return iou,cnt
        ######################

    def forward(self, x, y):#cuda 09
        pixel = 64
        loss = None
        correct = 0
        sum_out = None
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        conv_5 = torch.zeros((self.batch_size, 1, pixel, pixel), device=device, dtype=dtype)
        out_rec = [conv_5]##最後の層を引数に指定
        self._reset_state()

        
        #####skip connection(2層結合ver)GPU必須#####
        for t in range(self.num_time):
            # print(f'{x.shape=}')#x.shape=torch.Size([32, 4096, 11])
            x_t = x[:,:,t]  ##各ステップ数の64pix × 64pixをx_tに入れる。
            # print(f'{x_t.shape=}')#x_t.shape=torch.Size([32, 4096])
            # x_t = x_t.reshape((len(x_t), 1, 128, 128))
            x_t = x_t.reshape((len(x_t), 1, pixel, pixel))
            conv_1 = self.l1(x_t)  
            pool_1 = F.max_pool2d(conv_1, 2)
            conv_2 = self.l2(pool_1) 
            pool_2 = F.max_pool2d(conv_2, 2) 
            conv_3 = self.l3(pool_2) 
            unpool_1 = self.up_samp(conv_3) 
            concat_1 = torch.cat([conv_2, unpool_1],dim=1) 
            conv_4 = self.l4(concat_1) 
            unpool_2 = self.up_samp(conv_4)           
            concat_2 = torch.cat([conv_1,unpool_2],dim=1)
            conv_5 = self.l5(concat_2)
            out_rec.append(conv_5)
        ####################################

        
        out_rec = torch.stack(out_rec,dim=1)##時間軸を追加dim=1
        #print(f'{out_rec.shape=}') #out_rec.shape=torch.Size([32, 12, 1, 64, 64])12個の画像データをstackでテンソル連結する。stackで結合する理由はtime data方向で結合するため。(時間軸要素を追加する必要がある)
        #print("out_rec.shape",out_rec.shape) #out_rec.shape torch.Size([128, 21, 1, 64, 64]) ([バッチ,時間,分類])
        #m,_=torch.sum(out_rec,1)
        ##発火スパイクの足し合わせ↓##
        m =torch.sum(out_rec,1) #m.shape: torch.Size([256, 10]) for classifiartion
        #m = m/self.num_time
        # m : out_rec(21step)を時間軸で積算したもの
        # 出力mと教師信号yの形式を統一する
        y = y.reshape(len(x_t), 1, pixel, pixel)
        #m = torch.where(m>0,1,0).to(torch.float32)
        #y = torch.where((y>0)&(y<2),self.num_time//2,0).to(torch.float32)
        y = torch.where(y>0,self.num_time,0).to(torch.float32)
        #criterion = nn.CrossEntropyLoss() #MNIST 
        criterion = nn.MSELoss() # semantic segmantation
        loss = criterion(m, y)
        
        #metabolic_cost = self.gamma*torch.sum(m**3)
        #print("MSE_loss : metabplic_cost = ",loss,metabolic_cost)
        #loss += metabolic_cost
        iou,cnt= self.iou_score(m, y)
        

        return loss, m, out_rec, iou, cnt
######################################################################################################################
class Unet1_SNU(torch.nn.Module):
    def __init__(self, num_time=20, l_tau=0.8, soft=False, rec=False, forget=False, dual=False, gpu=True,
                 batch_size=32):
        super().__init__()

        
        self.num_time = num_time
        self.batch_size = batch_size
        self.rec = rec
        self.forget = forget
        self.dual = dual

         # Encoder layers
        self.c1 = 1
        self.c2 = 16
        self.c3 = 4
        self.c4 = 8
        self.c5 = 64
        ##self.l1～４にはsnu_layer.Conv_SNUのdef y(スパイク出力)が入る
        self.l1 = snu_layer.Conv_SNU(in_channels=self.c1, out_channels=self.c2, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l2 = snu_layer.Conv_SNU(in_channels=self.c2, out_channels=self.c3, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        
        # Decoder layers
        self.l3 = snu_layer.Conv_SNU(in_channels=self.c3, out_channels=self.c3, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l4 = snu_layer.Conv_SNU(in_channels=self.c4, out_channels=self.c1, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        ###############
        
        self.up_samp = nn.Upsample(scale_factor=2, mode='nearest')

    def _reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()

    def iou_score(self, outputs, labels):
        smooth = 1e-6
        outputs = outputs.data.cpu().numpy() #outputs.shape: (128, 1, 64, 64)
        labels = labels.data.cpu().numpy() #labels.shape: (128, 1, 64, 64)
        np.set_printoptions(threshold=np.inf)
        outputs = outputs.squeeze(1) # BATCH*1*H*W => BATCH*H*W __outputs.shape : (128, 64, 64)
        labels = labels.squeeze(1) #__labels.shape : (128, 64, 64)
        #print("outputs : ",outputs)
        iou = []
        cnt = []
        ########iouの計算#####
        for i in range(1,6):
            output = np.where(outputs>i,1,0)
            label = np.where(labels>0,1,0)
            intersection = (np.uint64(output) & np.uint64(label)).sum((1,2)) # will be zero if Trueth=0 or Prediction=0
            union = (np.uint64(output) | np.uint64(label)).sum((1,2)) # will be zero if both are 0
        
            iou.append((intersection + smooth) / (union + smooth))
            cnt.append(i)
        
        return iou,cnt
        ######################

    def forward(self, x, y):
        pixel = 64
        loss = None
        correct = 0
        sum_out = None
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = torch.zeros((self.batch_size, 1, pixel, pixel), device=device, dtype=dtype)
        out_rec = [out]##最後の層を引数に指定
        self._reset_state()

        

        ####skip connection(1層結合ver)#####
        for t in range(self.num_time):
            # print(f'{x.shape=}')#x.shape=torch.Size([32, 4096, 11])
            x_t = x[:,:,t]  ##各ステップ数の64pix × 64pixをx_tに入れる。
            # print(f'{x_t.shape=}')#x_t.shape=torch.Size([32, 4096])
            # x_t = x_t.reshape((len(x_t), 1, 128, 128))
            x_t = x_t.reshape((len(x_t), 1, pixel, pixel))
            conv_1 = self.l1(x_t)  
            pool_1 = F.max_pool2d(conv_1, 2)
            conv_2 = self.l2(pool_1) 
            pool_2 = F.max_pool2d(conv_2, 2) 
            conv_3 = self.l3(pool_2) 
            unpool_1 = self.up_samp(conv_3) 
            concat_1 = torch.cat([conv_2,unpool_1],dim=1) 
            conv_4 = self.l4(concat_1) 
            unpool_2 = self.up_samp(conv_4)           
            out_rec.append(unpool_2)
        ####################################
    
        out_rec = torch.stack(out_rec,dim=1)##時間軸を追加dim=1
        #print(f'{out_rec.shape=}') #out_rec.shape=torch.Size([32, 12, 1, 64, 64])12個の画像データをstackでテンソル連結する。stackで結合する理由はtime data方向で結合するため。(時間軸要素を追加する必要がある)
        #print("out_rec.shape",out_rec.shape) #out_rec.shape torch.Size([128, 21, 1, 64, 64]) ([バッチ,時間,分類])
        #m,_=torch.sum(out_rec,1)
        ##発火スパイクの足し合わせ↓##
        m =torch.sum(out_rec,1) #m.shape: torch.Size([256, 10]) for classifiartion
        #m = m/self.num_time
        # m : out_rec(21step)を時間軸で積算したもの
        # 出力mと教師信号yの形式を統一する
        y = y.reshape(len(x_t), 1, pixel, pixel)
        #m = torch.where(m>0,1,0).to(torch.float32)
        #y = torch.where((y>0)&(y<2),self.num_time//2,0).to(torch.float32)
        y = torch.where(y>0,self.num_time,0).to(torch.float32)
        #criterion = nn.CrossEntropyLoss() #MNIST 
        criterion = nn.MSELoss() # semantic segmantation
        loss = criterion(m, y)
        
        #metabolic_cost = self.gamma*torch.sum(m**3)
        #print("MSE_loss : metabplic_cost = ",loss,metabolic_cost)
        #loss += metabolic_cost
        iou,cnt= self.iou_score(m, y)
        

        return loss, m, out_rec, iou, cnt
########################################################################################################################################################################################################
class Unet_SNU(torch.nn.Module):##没
    def __init__(self, num_time=20, l_tau=0.8, soft=False, rec=False, forget=False, dual=False, gpu=True,
                 batch_size=32):
        super().__init__()

        
        self.num_time = num_time
        self.batch_size = batch_size
        self.rec = rec
        self.forget = forget
        self.dual = dual

         # Encoder layers
        self.c1 = 1
        self.c2 = 16
        self.c3 = 4
        self.c4 = 8
        self.c5 = 64
        ##self.l1～４にはsnu_layer.Conv_SNUのdef y(スパイク出力)が入る
        self.l1 = snu_layer.Conv_SNU(in_channels=self.c1, out_channels=self.c2, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l2 = snu_layer.Conv_SNU(in_channels=self.c2, out_channels=self.c3, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        
        # Decoder layers
        self.l3 = snu_layer.Conv_SNU(in_channels=self.c3, out_channels=self.c3, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l4 = snu_layer.Conv_SNU(in_channels=self.c3, out_channels=self.c1, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        ###############
        
        self.up_samp = nn.Upsample(scale_factor=2, mode='nearest')

    def _reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()

    def iou_score(self, outputs, labels):
        smooth = 1e-6
        outputs = outputs.data.cpu().numpy() #outputs.shape: (128, 1, 64, 64)
        labels = labels.data.cpu().numpy() #labels.shape: (128, 1, 64, 64)
        np.set_printoptions(threshold=np.inf)
        outputs = outputs.squeeze(1) # BATCH*1*H*W => BATCH*H*W __outputs.shape : (128, 64, 64)
        labels = labels.squeeze(1) #__labels.shape : (128, 64, 64)
        #print("outputs : ",outputs)
        iou = []
        cnt = []
        ########iouの計算#####
        for i in range(1,6):
            output = np.where(outputs>i,1,0)
            label = np.where(labels>0,1,0)
            intersection = (np.uint64(output) & np.uint64(label)).sum((1,2)) # will be zero if Trueth=0 or Prediction=0
            union = (np.uint64(output) | np.uint64(label)).sum((1,2)) # will be zero if both are 0
        
            iou.append((intersection + smooth) / (union + smooth))
            cnt.append(i)
        
        return iou,cnt
        ######################

    def forward(self, x, y):
        pixel = 64
        loss = None
        correct = 0
        sum_out = None
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = torch.zeros((self.batch_size, 1, pixel, pixel), device=device, dtype=dtype)
        out_rec = [out]##最後の層を引数に指定
        self._reset_state()

        ##skip connection使わない場合はコメント外す##
        for t in range(self.num_time):
            # print(f'{x.shape=}')#x.shape=torch.Size([32, 4096, 11])
            x_t = x[:,:,t]  ##各ステップ数の64pix × 64pixをx_tに入れる。
            # print(f'{x_t.shape=}')#x_t.shape=torch.Size([32, 4096])
            # print(x_t.shape)
            # x_t = x_t.reshape((len(x_t), 1, 128, 128))
            x_t = x_t.reshape((len(x_t), 1, pixel, pixel))
            #print("x_t : ",x_t.shape)
            conv_1 = self.l1(x_t) # h1 :  torch.Size([256, 16, 64, 64])  
            pool_1 = F.max_pool2d(conv_1, 2)#h1_ :  torch.Size([256, 16, 32, 32])
            conv_2 = self.l2(pool_1) #h2 :  torch.Size([256, 4, 32, 32])
            pool_2 = F.max_pool2d(conv_2, 2)#h2 :  torch.Size([256, 16, 16, 16])　##max pool2dの引数(領域のサイズ、ストライド)
            conv_3 = self.l3(pool_2)
            unpool_1 = self.up_samp(conv_3)
            out = self.l4(unpool_1) #out.shape=torch.Size([32, 1, 32, 32])
            #print(f'{out.shape=}')#out.shape torch.Size([256, 10]) # [バッチサイズ,output.shape]
            out = self.up_samp(out)##最終層のアップサンプリング後
            #print(f'{out.shape=}')#out.shape=torch.Size([32, 1, 64, 64])

            #print("out.shape",out.shape) #out[0].shape torch.Size([10])
            #print("sum out[0]:",sum(out[0]))  #tensor([1., 0., 1., 0., 1., 0., 1., 1., 0., 1.], device='cuda:0',
            #sum_out = out if sum_out is Nonec else sum_out + out
            out_rec.append(out)##64×64×1channelの画像を配列に入れる。0番目のデータ(line81)+time data 11の合計12個
            # print(out_rec.dtype)
            ######
        
        #####skip connection(2層結合ver)GPU必須#####
        # for t in range(self.num_time):
        #     # print(f'{x.shape=}')#x.shape=torch.Size([32, 4096, 11])
        #     x_t = x[:,:,t]  ##各ステップ数の64pix × 64pixをx_tに入れる。
        #     # print(f'{x_t.shape=}')#x_t.shape=torch.Size([32, 4096])
        #     # x_t = x_t.reshape((len(x_t), 1, 128, 128))
        #     x_t = x_t.reshape((len(x_t), 1, pixel, pixel))
        #     conv_1 = self.l1(x_t)  
        #     pool_1 = F.max_pool2d(conv_1, 2)
        #     conv_2 = self.l2(pool_1) 
        #     pool_2 = F.max_pool2d(conv_2, 2) 
        #     conv_3 = self.l3(pool_2) 
        #     unpool_1 = self.up_samp(conv_3) 
        #     concat_1 = torch.cat([conv_2,unpool_1],dim=1) 
        #     conv_4 = self.l4(concat_1) 
        #     unpool_2 = self.up_samp(conv_4)           
        #     concat_2 = torch.cat([conv_1,unpool_2],dim=1)
        #     conv_5 = self.l5(concat_2)
        #     out_rec.append(conv_5)
        ####################################

        ####skip connection(1層結合ver)#####
        # for t in range(self.num_time):
        #     # print(f'{x.shape=}')#x.shape=torch.Size([32, 4096, 11])
        #     x_t = x[:,:,t]  ##各ステップ数の64pix × 64pixをx_tに入れる。
        #     # print(f'{x_t.shape=}')#x_t.shape=torch.Size([32, 4096])
        #     # x_t = x_t.reshape((len(x_t), 1, 128, 128))
        #     x_t = x_t.reshape((len(x_t), 1, pixel, pixel))
        #     conv_1 = self.l1(x_t)  
        #     pool_1 = F.max_pool2d(conv_1, 2)
        #     conv_2 = self.l2(pool_1) 
        #     pool_2 = F.max_pool2d(conv_2, 2) 
        #     conv_3 = self.l3(pool_2) 
        #     unpool_1 = self.up_samp(conv_3) 
        #     concat_1 = torch.cat([conv_2,unpool_1],dim=1) 
        #     conv_4 = self.l4(concat_1) 
        #     unpool_2 = self.up_samp(conv_4)           
        #     out_rec.append(unpool_2)
        ####################################


            #print("out.shape",out.shape) #out[0].shape torch.Size([10])
            #print("sum out[0]:",sum(out[0]))  #tensor([1., 0., 1., 0., 1., 0., 1., 1., 0., 1.], device='cuda:0',
            #sum_out = out if sum_out is Nonec else sum_out + out
            #out_rec.append(unpool_2)##64×64×1channelの画像を配列に入れる。0番目のデータ(line81)+time data 11の合計12個
            #print(out_rec.dtype)
    
        out_rec = torch.stack(out_rec,dim=1)##時間軸を追加dim=1
        #print(f'{out_rec.shape=}') #out_rec.shape=torch.Size([32, 12, 1, 64, 64])12個の画像データをstackでテンソル連結する。stackで結合する理由はtime data方向で結合するため。(時間軸要素を追加する必要がある)
        #print("out_rec.shape",out_rec.shape) #out_rec.shape torch.Size([128, 21, 1, 64, 64]) ([バッチ,時間,分類])
        #m,_=torch.sum(out_rec,1)
        ##発火スパイクの足し合わせ↓##
        m =torch.sum(out_rec,1) #m.shape: torch.Size([256, 10]) for classifiartion
        #m = m/self.num_time
        # m : out_rec(21step)を時間軸で積算したもの
        # 出力mと教師信号yの形式を統一する
        y = y.reshape(len(x_t), 1, pixel, pixel)
        #m = torch.where(m>0,1,0).to(torch.float32)
        #y = torch.where((y>0)&(y<2),self.num_time//2,0).to(torch.float32)
        y = torch.where(y>0,self.num_time,0).to(torch.float32)
        #criterion = nn.CrossEntropyLoss() #MNIST 
        criterion = nn.MSELoss() # semantic segmantation
        loss = criterion(m, y)
        
        #metabolic_cost = self.gamma*torch.sum(m**3)
        #print("MSE_loss : metabplic_cost = ",loss,metabolic_cost)
        #loss += metabolic_cost
        iou,cnt= self.iou_score(m, y)
        

        return loss, m, out_rec, iou, cnt
####################################################################################################################################################################################################        
#10/19　unetにchannelのparameter合わせたGated - SNU
class revisedSNU2_Network(torch.nn.Module):
    def __init__(self, num_time=20, l_tau=0.8, soft=False, rec=False, forget=False, dual=False, gpu=True,power = False,heatmap = False,
                 batch_size = 32):
        super(revisedSNU2_Network, self).__init__()

        
        self.num_time = num_time
        self.batch_size = batch_size
        self.rec = rec
        self.forget = forget
        self.dual = dual
        self.power = power
        self.heatmap = heatmap
        
        # Encoder layers
        self.l1 = snu_layer.Conv_SNU(in_channels=1, out_channels=16, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l2 = snu_layer.Conv_SNU(in_channels=16, out_channels=4, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        
        # Decoder layers
        self.l3 = snu_layer.Conv_SNU(in_channels=4, out_channels=16, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l4 = snu_layer.Conv_SNU(in_channels=16, out_channels=1, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        
        self.up_samp = nn.Upsample(scale_factor=2, mode='nearest')

    def _reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()

    def iou_score(self, outputs, labels):
        smooth = 1e-6
        outputs = outputs.data.cpu().numpy() #outputs.shape: (128, 1, 64, 64)
        labels = labels.data.cpu().numpy() #labels.shape: (128, 1, 64, 64)
        np.set_printoptions(threshold=np.inf)
        outputs = outputs.squeeze(1) # BATCH*1*H*W => BATCH*H*W __outputs.shape : (128, 64, 64)
        labels = labels.squeeze(1) #__labels.shape : (128, 64, 64)
        #print("outputs : ",outputs)
        iou = []
        cnt = []
        #####iouの計算####
        for i in range(1,6):
            output = np.where(outputs>i,1,0)
            label = np.where(labels>0,1,0)
            intersection = (np.uint64(output) & np.uint64(label)).sum((1,2)) # will be zero if Trueth=0 or Prediction=0
            union = (np.uint64(output) | np.uint64(label)).sum((1,2)) # will be zero if both are 0
        
            iou.append((intersection + smooth) / (union + smooth))
            cnt.append(i)
        
        return iou,cnt
        #################

    def forward(self, x, y):
        pixel = 64
        loss = None
        correct = 0
        sum_out = None
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = torch.zeros((self.batch_size, 1, pixel, pixel), device=device, dtype=dtype)
        out_rec = [out]
        self._reset_state()
        ht_spike_count = 0
        conv_1_spike_count = 0
        conv_2_spike_count = 0
        conv_3_spike_count = 0
        conv_4_spike_count = 0

        for t in range(self.num_time):
            # print(f'{x.shape=}')#x.shape=torch.Size([32, 4096, 11])
            x_t = x[:,:,t]
            # print(f'{x_t.shape=}')#x_t.shape=torch.Size([32, 4096])
            # print(x_t.shape)
            # x_t = x_t.reshape((len(x_t), 1, 128, 128))
            x_t = x_t.reshape((len(x_t), 1, pixel, pixel))
            #print("x_t : ",x_t.shape)
            conv_1 = self.l1(x_t) # h1 :  torch.Size([256, 16, 64, 64])  
            pool_1 = F.max_pool2d(conv_1, 2) #h1_ :  torch.Size([256, 16, 32, 32])
            conv_2 = self.l2(pool_1) #h2 :  torch.Size([256, 4, 32, 32])
            pool_2 = F.max_pool2d(conv_2, 2)#h2 :  torch.Size([256, 16, 16, 16])
            conv_3 = self.l3(pool_2)
            unpool_1 = self.up_samp(conv_3)
            conv_4 = self.l4(unpool_1) #out.shape torch.Size([256, 10]) # [バッチサイズ,output.shape]
            out = self.up_samp(conv_4)
            #print("out.shape",out.shape) #out[0].shape torch.Size([10])
            #print("sum out[0]:",sum(out[0]))  #tensor([1., 0., 1., 0., 1., 0., 1., 1., 0., 1.], device='cuda:0',
            #sum_out = out if sum_out is Nonec else sum_out + out
            out_rec.append(out)

            if self.power:
                ht_spike_count += torch.sum(x_t)
                conv_1_spike_count += torch.sum(conv_1)
                conv_2_spike_count += torch.sum(conv_2)
                conv_3_spike_count += torch.sum(conv_3)
                conv_4_spike_count += torch.sum(conv_4)
                #conv_5_spike_count += torch.sum(conv_5)
        total_spike_count =[ht_spike_count,conv_1_spike_count,conv_2_spike_count,conv_3_spike_count,conv_4_spike_count]
        out_rec = torch.stack(out_rec,dim=1)
        #print("out_rec.shape",out_rec.shape) #out_rec.shape torch.Size([128, 21, 1, 64, 64]) ([バッチ,時間,分類])
        #m,_=torch.sum(out_rec,1)
        ##発火スパイクの足し合わせ↓##
        m =torch.sum(out_rec,1) #m.shape: torch.Size([256, 10]) for classifiartion
        heat = m
        #m = m/self.num_time
        # m : out_rec(21step)を時間軸で積算したもの
        # 出力mと教師信号yの形式を統一する
        y = y.reshape(len(x_t), 1, pixel, pixel)
        #m = torch.where(m>0,1,0).to(torch.float32)
        #y = torch.where((y>0)&(y<2),self.num_time//2,0).to(torch.float32)
        y = torch.where(y>0,self.num_time,0).to(torch.float32)
        #criterion = nn.CrossEntropyLoss() #MNIST 
        criterion = nn.MSELoss() # semantic segmantation
        loss = criterion(m, y)
        
        #metabolic_cost = self.gamma*torch.sum(m**3)
        #print("MSE_loss : metabplic_cost = ",loss,metabolic_cost)
        #loss += metabolic_cost
        iou,cnt= self.iou_score(m, y)
        # if self.heatmap:
        #     return loss, m, out_rec, iou, cnt, total_spike_count
        if self.heatmap:
            return loss, m, out_rec, iou, cnt, total_spike_count, heat
        if self.power:
            return loss, m, out_rec, iou, cnt, total_spike_count
        else:
            return loss, m, out_rec, iou, cnt
################################################################################################################################################################
# 7/24~
class revisedSNU_Network(torch.nn.Module):
    def __init__(self, num_time=20, l_tau=0.8, soft=False, rec=False, forget=False, dual=False, gpu=True,
                 batch_size=32):
        super(revisedSNU_Network, self).__init__()

        
        self.num_time = num_time
        self.batch_size = batch_size
        self.rec = rec
        self.forget = forget
        self.dual = dual
        # Encoder layers
        self.l1 = snu_layer.Conv_SNU(in_channels=1, out_channels=16, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l2 = snu_layer.Conv_SNU(in_channels=16, out_channels=4, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        
        # Decoder layers
        self.l3 = snu_layer.Conv_SNU(in_channels=4, out_channels=16, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l4 = snu_layer.Conv_SNU(in_channels=16, out_channels=1, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        
        self.up_samp = nn.Upsample(scale_factor=2, mode='nearest')

    def _reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()

    def iou_score(self, outputs, labels):
        smooth = 1e-6
        outputs = outputs.data.cpu().numpy() #outputs.shape: (128, 1, 64, 64)
        labels = labels.data.cpu().numpy() #labels.shape: (128, 1, 64, 64)
        np.set_printoptions(threshold=np.inf)
        outputs = outputs.squeeze(1) # BATCH*1*H*W => BATCH*H*W __outputs.shape : (128, 64, 64)
        labels = labels.squeeze(1) #__labels.shape : (128, 64, 64)
        #print("outputs : ",outputs)
        iou = []
        cnt = []
        #####iouの計算####
        for i in range(1,6):
            output = np.where(outputs>i,1,0)
            label = np.where(labels>0,1,0)
            intersection = (np.uint64(output) & np.uint64(label)).sum((1,2)) # will be zero if Trueth=0 or Prediction=0
            union = (np.uint64(output) | np.uint64(label)).sum((1,2)) # will be zero if both are 0
        
            iou.append((intersection + smooth) / (union + smooth))
            cnt.append(i)
        
        return iou,cnt
        #################

    def forward(self, x, y):
        pixel = 64
        loss = None
        correct = 0
        sum_out = None
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = torch.zeros((self.batch_size, 1, pixel, pixel), device=device, dtype=dtype)
        out_rec = [out]
        self._reset_state()

        for t in range(self.num_time):
            # print(f'{x.shape=}')#x.shape=torch.Size([32, 4096, 11])
            x_t = x[:,:,t]
            # print(f'{x_t.shape=}')#x_t.shape=torch.Size([32, 4096])
            # print(x_t.shape)
            # x_t = x_t.reshape((len(x_t), 1, 128, 128))
            x_t = x_t.reshape((len(x_t), 1, pixel, pixel))
            #print("x_t : ",x_t.shape)
            h1 = self.l1(x_t) # h1 :  torch.Size([256, 16, 64, 64])  
            h1 = F.max_pool2d(h1, 2) #h1_ :  torch.Size([256, 16, 32, 32])
            h2 = self.l2(h1) #h2 :  torch.Size([256, 4, 32, 32])
            h2 = F.max_pool2d(h2, 2)#h2 :  torch.Size([256, 16, 16, 16])
            h3 = self.l3(h2)
            h3 = self.up_samp(h3)
            out = self.l4(h3) #out.shape torch.Size([256, 10]) # [バッチサイズ,output.shape]
            out = self.up_samp(out)
            #print("out.shape",out.shape) #out[0].shape torch.Size([10])
            #print("sum out[0]:",sum(out[0]))  #tensor([1., 0., 1., 0., 1., 0., 1., 1., 0., 1.], device='cuda:0',
            #sum_out = out if sum_out is Nonec else sum_out + out
            out_rec.append(out)
    
        out_rec = torch.stack(out_rec,dim=1)
        #print("out_rec.shape",out_rec.shape) #out_rec.shape torch.Size([128, 21, 1, 64, 64]) ([バッチ,時間,分類])
        #m,_=torch.sum(out_rec,1)
        ##発火スパイクの足し合わせ↓##
        m =torch.sum(out_rec,1) #m.shape: torch.Size([256, 10]) for classifiartion
        #m = m/self.num_time
        # m : out_rec(21step)を時間軸で積算したもの
        # 出力mと教師信号yの形式を統一する
        y = y.reshape(len(x_t), 1, pixel, pixel)
        #m = torch.where(m>0,1,0).to(torch.float32)
        #y = torch.where((y>0)&(y<2),self.num_time//2,0).to(torch.float32)
        y = torch.where(y>0,self.num_time,0).to(torch.float32)
        #criterion = nn.CrossEntropyLoss() #MNIST 
        criterion = nn.MSELoss() # semantic segmantation
        loss = criterion(m, y)
        
        #metabolic_cost = self.gamma*torch.sum(m**3)
        #print("MSE_loss : metabplic_cost = ",loss,metabolic_cost)
        #loss += metabolic_cost
        iou,cnt= self.iou_score(m, y)
        

        return loss, m, out_rec, iou, cnt
        

# 7/5 ~
class Fully_Connected_Gated_SNU_Net(torch.nn.Module):
    def __init__(self, n_in=4096, n_mid=4096, n_out=4096,
                 num_time=20, l_tau=0.8, rec=True, gpu=True,batch_size=128):
        super(Fully_Connected_Gated_SNU_Net, self).__init__()

        self.n_out = n_out
        self.num_time = num_time
        self.batch_size = batch_size
        self.rec = rec
        self.l_tau = l_tau
        print("self.rec",self.rec)
 
        self.l1 = revised_snu_layer.Gated_SNU(n_in,  rec=self.rec, l_tau=self.l_tau, gpu=gpu)
        self.l2 = revised_snu_layer.Gated_SNU(n_mid, rec=self.rec, l_tau=self.l_tau, gpu=gpu)
        self.l3 = revised_snu_layer.Gated_SNU(n_mid, rec=self.rec, l_tau=self.l_tau, gpu=gpu)
        self.l4 = revised_snu_layer.Gated_SNU(n_out, rec=self.rec, l_tau=self.l_tau, gpu=gpu)
        

    def _reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()

    def iou_score(self, outputs, labels):
        smooth = 1e-6

        outputs = outputs.data.cpu().numpy().reshape(self.batch_size,64,64)  
        labels = labels.data.cpu().numpy().reshape(self.batch_size,64,64)   
        np.set_printoptions(threshold=np.inf)
        #outputs = outputs.squeeze(1) # BATCH*1*H*W => BATCH*H*W __outputs.shape : (128, 64, 64)
        #labels = labels.squeeze(1) #__labels.shape : (128, 64, 64)
        #print("outputs : ",outputs)
        iou = []
        cnt = []
        for i in range(1,6):
            output = np.where(outputs>i,1,0)
            label = np.where(labels>0,1,0)
           # print("output",output)
            #print("label",label)
            intersection = (np.uint64(output) & np.uint64(label)).sum((1,2))
            #intersection = (np.uint64(output) & np.uint64(label).sum((0,1))) # will be zero if Trueth=0 or Prediction=0
            #union = (np.uint64(output) | np.uint64(label)).sum((0,1)) # will be zero if both are 0
            union = (np.uint64(output) | np.uint64(label)).sum((1,2))
            #print("intersection,union",intersection ,union )
            iou.append((intersection + smooth) / (union + smooth))
            cnt.append(i)
        
        return iou,cnt        
    def forward(self, x, y):
        loss = None

        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = torch.zeros((self.batch_size ,self.n_out), device=device, dtype=dtype)
        out_rec = [out]
        self._reset_state()
        
        for t in range(self.num_time):
            x_t = x[:,:,t]  #torch.Size([256, 784])
            x_t = x_t.reshape((len(x_t),4096))
            h1 = self.l1(x_t) # torch.Size([256, 256])
            #print("sum h1[0]",sum(h1[0]))
            h2 = self.l2(h1) #h2.shape: torch.Size([256, 256])
            #print("sum h2[0]",sum(h2[0]))
            h3 = self.l3(h2)
            #print("sum h3[0]",sum(h3[0]))
            out = self.l4(h3) #out.shape torch.Size([256, 10]) # [バッチサイズ,output.shape]
            #print("out.shape",out.shape) #out[0].shape torch.Size([10])
            #print("out[0]:",out[0])  #tensor([1., 0., 1., 0., 1., 0., 1., 1., 0., 1.], device='cuda:0',            
            #sum_out = out if sum_out is None else sum_out + out
            out_rec.append(out)
    
        out_rec = torch.stack(out_rec,dim=1)
        #print("out_rec.shape",out_rec.shape) #out_rec.shape torch.Size([256, 11, 10]) ([バッチ,時間,分類])
        #m,_=torch.sum(out_rec,1)
        m =torch.sum(out_rec,1) #m.shape: torch.Size([256, 10])
       
        #print("m",m)
        y = torch.tensor(y, dtype=torch.float)
        y = torch.where(y>0,self.num_time,0).to(torch.float32)

        #criterion = nn.CrossEntropyLoss() #MNIST 
        criterion = nn.MSELoss() # semantic segmantation
        loss = criterion(m, y)
        iou,cnt= self.iou_score(m, y)
        out_rec =out_rec.reshape(128,21,64,64)
        return loss, m, out_rec, iou, cnt

# 改(7/6~) NC_KEN 
class SNU_Network(torch.nn.Module):
    def __init__(self, num_time=20, l_tau=0.8, soft=False, rec=False, forget=False, dual=False, power=False, gpu=True,
                 batch_size=128):
        super(SNU_Network, self).__init__()

        
        self.num_time = num_time
        self.batch_size = batch_size
        self.rec = rec
        self.forget = forget
        self.dual = dual
        self.power = power

        # Encoder layers
        self.l1 = snu_layer.Conv_SNU(in_channels=1, out_channels=16, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l2 = snu_layer.Conv_SNU(in_channels=16, out_channels=4, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        
        # Decoder layers
        self.l3 = snu_layer.Conv_SNU(in_channels=4, out_channels=16, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l4 = snu_layer.Conv_SNU(in_channels=16, out_channels=1, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        
        self.up_samp = nn.Upsample(scale_factor=2, mode='nearest')

    def _reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()

    def iou_score(self, outputs, labels):
        smooth = 1e-6
        outputs = outputs.data.cpu().numpy() #outputs.shape: (128, 1, 64, 64)
        labels = labels.data.cpu().numpy() #labels.shape: (128, 1, 64, 64)
        np.set_printoptions(threshold=np.inf)
        outputs = outputs.squeeze(1) # BATCH*1*H*W => BATCH*H*W __outputs.shape : (128, 64, 64)
        labels = labels.squeeze(1) #__labels.shape : (128, 64, 64)
        #print("outputs : ",outputs)
        iou = []
        cnt = []
        for i in range(2,7):
            #i = i*10 # if t-70
            output = np.where(outputs>i,1,0)
            label = np.where(labels>0,1,0)
            intersection = (np.uint64(output) & np.uint64(label)).sum((1,2)) # will be zero if Trueth=0 or Prediction=0
            union = (np.uint64(output) | np.uint64(label)).sum((1,2)) # will be zero if both are 0
        
            iou.append((intersection + smooth) / (union + smooth))
            cnt.append(i)
        
        return iou,cnt
        
    def forward(self, x, y):
        loss = None
        correct = 0
        sum_out = None
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = torch.zeros((self.batch_size, 1, 64, 64), device=device, dtype=dtype)
        out_rec = [out]
        #print('out shape',out.shape)
        self._reset_state()
        ht_spike_count = 0
        h1_spike_count = 0
        h2_spike_count = 0
        h3_spike_count = 0
        out_spike_count = 0

        for t in range(self.num_time):
            x_t = x[:,:,t]  #torch.Size([256, 784])
            x_t = x_t.reshape((len(x_t), 1, 64, 64))
            #print('x_t',x_t.shape)
            #print('x_t.sum',torch.sum(x_t))

            h1 = self.l1(x_t) # h1 :  torch.Size([256, 16, 64, 64])  

            h1_ = F.max_pool2d(h1, 2) #h1_ :  torch.Size([256, 16, 32, 32])
            h2 = self.l2(h1_) #h2 :  torch.Size([256, 4, 32, 32])

            h2_ = F.max_pool2d(h2, 2)#h2 :  torch.Size([256, 16, 16, 16])
            h3 = self.l3(h2_)

            h3_ = self.up_samp(h3)
            out = self.l4(h3_) #out.shape torch.Size([256, 10]) # [バッチサイズ,output.shape]

            out_ = self.up_samp(out)
            # print(out_)
            out_rec.append(out_)

            if self.power:
                ht_spike_count += torch.sum(x_t)
                #print('torch.sum(x_t)',torch.sum(x_t))
                #print('ht_spike_count',ht_spike_count)
                #print('h1:',h1.shape)
                h1_spike_count += torch.sum(h1)
                #print('h2:',h2.shape)
                h2_spike_count += torch.sum(h2)
                #print('h3:',h3.shape)
                h3_spike_count += torch.sum(h3)
                #print('out:',out.shape)
                out_spike_count += torch.sum(out)
        
        total_spike_count = [ht_spike_count,h1_spike_count,h2_spike_count,h3_spike_count,out_spike_count]
                #print('total_spike_count shape : ',total_spike_count.shape)
        # print(out_rec)
        out_rec = torch.stack(out_rec,dim=1)
        # print("out_rec.shape", out_rec.shape)
        #print("out_rec.shape",out_rec.shape) #out_rec.shape torch.Size([128, 21, 1, 64, 64]) ([バッチ,時間,分類])
        #m,_=torch.sum(out_rec,1)
        
        # スパイクの本数を合計
        m =torch.sum(out_rec, dim=1) #m.shape: torch.Size([256, 10]) for classifiartion
        #m = m/self.num_time
        # m : out_rec(21step)を時間軸で積算したもの
        # 出力mと教師信号yの形式を統一する
        y = y.reshape(len(x_t), 1, 64, 64)
        #m = torch.where(m>0,1,0).to(torch.float32)
        #y = torch.where((y>0)&(y<2),self.num_time//2,0).to(torch.float32)
        y = torch.where(y>0,self.num_time,0).to(torch.float32)
        #criterion = nn.CrossEntropyLoss() #MNIST 
        criterion = nn.MSELoss() # semantic segmantation
        loss = criterion(m, y)
        
        #metabolic_cost = self.gamma*torch.sum(m**3)
        #print("MSE_loss : metabplic_cost = ",loss,metabolic_cost)
        #loss += metabolic_cost
        iou,cnt= self.iou_score(m, y)
        if self.power:
            return loss, m, out_rec, iou, cnt, total_spike_count
        else:
            return loss, m, out_rec, iou, cnt
        
class SNU_Network_classification(torch.nn.Module):
    def __init__(self, n_in=784, n_mid=256, n_out=10,
                 num_time=20, l_tau=0.8, soft=False, gpu=False,
                 test_mode=False):
        super(SNU_Network_classification, self).__init__()
        
        self.l1 = snu_layer.SNU(n_in, n_mid, l_tau=l_tau, soft=soft, gpu=gpu)
        self.l2 = snu_layer.SNU(n_mid, n_mid, l_tau=l_tau, soft=soft, gpu=gpu)
        self.l3 = snu_layer.SNU(n_mid, n_mid, l_tau=l_tau, soft=soft, gpu=gpu)
        self.l4 = snu_layer.SNU(n_mid, n_out, l_tau=l_tau, soft=soft, gpu=gpu)
        
        self.n_out = n_out
        self.num_time = num_time
        self.gamma = (1/(num_time*n_out))*1e-3
        self.test_mode = test_mode

    def _reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()
        
    def forward(self, x, y):
        loss = None
        acc = 0
        sum_out = None
        correct = 0
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = torch.zeros((128,self.n_out), device=device, dtype=dtype)
        out_rec = [out]
        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()
        self._reset_state()
        
        if self.test_mode == True:
            h1_list = []
            h2_list = []
            h3_list = []
            out_list = []
        
        for t in range(self.num_time):
            #print("**********************")
            #print("t :",t)
            #print("x[0]  shape:",x[0].shape)
            x_t = x[:,:,t]  #torch.Size([256, 784])
            
            #print("x_t[0] shape",x_t[0].shape)
            #print("sum x_t[0]",sum(x_t[0]))
            
           
            h1 = self.l1(x_t) # torch.Size([256, 256])

            #print("sum h1[0]",sum(h1[0]))
            h2 = self.l2(h1) #h2.shape: torch.Size([256, 256])
            #print("sum h2[0]",sum(h2[0]))
            h3 = self.l3(h2)
            #print("sum h3[0]",sum(h3[0]))
            out = self.l4(h3) #out.shape torch.Size([256, 10]) # [バッチサイズ,output.shape]
            #print("out.shape",out.shape) #out[0].shape torch.Size([10])
            #print("out[0]:",out[0])  #tensor([1., 0., 1., 0., 1., 0., 1., 1., 0., 1.], device='cuda:0',

            
            if self.test_mode == True:
                h1_list.append(h1)
                h2_list.append(h2)
                h3_list.append(h3)
                out_list.append(out)
            
            #sum_out = out if sum_out is None else sum_out + out
            out_rec.append(out)
    
        out_rec = torch.stack(out_rec,dim=1)
        #print("out_rec.shape",out_rec.shape) #out_rec.shape torch.Size([256, 11, 10]) ([バッチ,時間,分類])
        #m,_=torch.sum(out_rec,1)
        m =torch.sum(out_rec,1) #m.shape: torch.Size([256, 10])
        m = m/20
        #print("type m:",m.type())
        #print("m",m)
        
        #print("out_rec.shape",out_rec.shape)
        y = torch.tensor(y, dtype=torch.int64)
        #print("type y",y.type()) #torch.Size([128]))
        #print("y",y)
        print("///////////////////////")

        criterion = nn.CrossEntropyLoss() #MNIST 
        #criterion = nn.MSELoss() # semantic segmantation
        _,m_col =  torch.max(m, 1)
        #_,y_col = torch.max(y,1)
        #acc = torch.sum(m_col == y_col) * 1.0 / len(y)
        #acc = acc.to('cpu').detach().numpy().copy()
        #print("correct : ",acc)
        loss = criterion(m, y)
        print("end of BCE loss :", loss)
        #print("correct : ",acc)
        #acc = torch.sum(m == y) * 1.0 / len(y)
        y = F.one_hot(y,num_classes=2)
        _,y_col = torch.max(y,1)
        acc = torch.sum(m_col == y_col) * 1.0 / len(y)
        acc = acc.to('cpu').detach().numpy().copy()
        print("correct : ",acc)
        #loss += self.gamma*torch.sum(m**2)
        #print("gamma loss",loss)
        
        

        
        if self.test_mode == True:
            return loss, accuracy, h1_list, h2_list, h3_list, out_list
        else:
            return loss, m, out_rec,acc

# 新実装(4/21=)
class Conv_SNU_Network_classification(torch.nn.Module):
    def __init__(self, n_in=784, n_mid=256, n_out=2, filter = 10,
                 num_time=20, l_tau=0.8, soft=False, gpu=False,
                 ):
        super(Conv_SNU_Network_classification, self).__init__()
        
        # 入力チャネル数 出力チャネル数 フィルタサイズ
        self.cn1 = snu_layer.Conv_SNU(in_channels=1, out_channels=6,kernel_size=10, l_tau=l_tau, soft=soft, gpu=gpu)
        self.l2 = snu_layer.SNU(in_channels=55, out_channels=2, l_tau=l_tau, soft=soft, gpu=gpu)
       
        self.n_out = n_out
        self.num_time = num_time
        self.gamma = (1/(num_time*n_out))*1e-3

    def _reset_state(self):
        self.cn1.reset_state()
        self.l2.reset_state()
        
    def forward(self, x, y):
        loss = None
        acc = 0
        sum_out = None
        correct = 0
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = torch.zeros((32,self.n_out), device=device, dtype=dtype)
        out_rec = [out]
        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()
        self._reset_state()

        
        for t in range(self.num_time):

            x_t = x[:,:,t]  #torch.Size([256, 784])
            #print("x_t : ",x_t.shape)
            x_t = x_t.reshape((len(x_t), 1, 32, 32))
            #####
            #fig = plt.figure(figsize=(12,8))
            #ax1 = fig.add_subplot(121)
            x_t_=x_t.to('cpu').detach().numpy().copy()
            #print("x_t_ shape",x_t_.shape)
            ##ax1 = plt.imshow(x_t_[0,0,:,:])
            ax1 = plt.title("input:"+str(t)+"")
            
            # 第一層　畳み込み
            h1 = self.cn1(x_t) 
            #####
            h1_=h1.to('cpu').detach().numpy().copy()
            """
            print("h1_ shape",h1_.shape)
            ax2 = fig.add_subplot(122)
            ax2 = plt.imshow(h1_[0,0,:,:])
            ax2 = plt.title("after conv")
            plt.show()
            """
            # 第二層 最大プーリング
            h2 = F.max_pool2d(h1, 2)
            #print("h2 :",h2.shape)
            h2 = torch.flatten(h2, 1)
            #print("h2_ : ",h2.shape )
            # 第三層　出力
            out = self.l2(h2)
            #print("out_",out.shape)
            
            out_rec.append(out)
    
        out_rec = torch.stack(out_rec,dim=1)
        #print("out_rec.shape",out_rec.shape) #out_rec.shape torch.Size([256, 11, 10]) ([バッチ,時間,分類])
        #m,_=torch.sum(out_rec,1)
        m =torch.sum(out_rec,1) #m.shape: torch.Size([256, 10])
        m = m/self.num_time
        #print("type m:",m.type())
        #print("m",m)
        
        #print("out_rec.shape",out_rec.shape)
        y = torch.tensor(y, dtype=torch.int64)
        #print("type y",y.type()) #torch.Size([128]))
        #print("y",y)


        #print("///////////////////////")
        criterion = nn.CrossEntropyLoss() #MNIST 
        #criterion = nn.MSELoss() # semantic segmantation
        _,m_col =  torch.max(m, 1)
        #_,y_col = torch.max(y,1)
        #acc = torch.sum(m_col == y_col) * 1.0 / len(y)
        #acc = acc.to('cpu').detach().numpy().copy()
        #print("correct : ",acc)
        loss = criterion(m, y)
        #print("end of BCE loss :", loss)
        #print("correct : ",acc)
        #acc = torch.sum(m == y) * 1.0 / len(y)
        y = F.one_hot(y,num_classes=2)
        _,y_col = torch.max(y,1)
        acc = torch.sum(m_col == y_col) * 1.0 / len(y)
        acc = acc.to('cpu').detach().numpy().copy()
        #print("correct : ",acc)
        #loss += self.gamma*torch.sum(m**2)
        #print("gamma loss",loss)
        return loss, m, out_rec,acc