# -*- coding: utf-8 -*-
 
from cmath import pi
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from .. import snu_layer

import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt


# Network definition]

########################################################################################################################################################################################################
class Unet_SNU(torch.nn.Module):
    def __init__(self, num_time=20, l_tau=0.8, soft=False, rec=False, forget=False, dual=False, gpu=True,
                 batch_size=32):
        super().__init__()

        
        self.num_time = num_time
        self.batch_size = batch_size
        self.rec = rec
        self.forget = forget
        self.dual = dual
        # Encoder layers
        self.c1 = 1##入力
        self.c2 = 16##conv_1後channel数
        self.c3 = 8##(変更前32)conv_2後channel数
        self.c4 = 8##conv_3後channel数
        self.c5 = 16##concat後のchannel数
        ##self.l1～４にはsnu_layer.Conv_SNUのdef y(スパイク出力)が入る
        self.l1 = snu_layer.Conv_SNU(in_channels=self.c1, out_channels=self.c2, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l2 = snu_layer.Conv_SNU(in_channels=self.c2, out_channels=self.c3, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        
        # Decoder layers
        self.l3 = snu_layer.Conv_SNU(in_channels=self.c3, out_channels=self.c4, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l4 = snu_layer.Conv_SNU(in_channels=self.c5, out_channels=self.c1, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        
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
        unpool_2 = torch.zeros((self.batch_size, 1, pixel, pixel), device=device, dtype=dtype)
        out_rec = [unpool_2]
        self._reset_state()

        ##skip connection使わない場合はコメント外す##
        # for t in range(self.num_time):
        #     # print(f'{x.shape=}')#x.shape=torch.Size([32, 4096, 11])
        #     x_t = x[:,:,t]  ##各ステップ数の64pix × 64pixをx_tに入れる。
        #     # print(f'{x_t.shape=}')#x_t.shape=torch.Size([32, 4096])
        #     # print(x_t.shape)
        #     # x_t = x_t.reshape((len(x_t), 1, 128, 128))
        #     x_t = x_t.reshape((len(x_t), 1, pixel, pixel))
        #     #print("x_t : ",x_t.shape)
        #     h1 = self.l1(x_t) # h1 :  torch.Size([256, 16, 64, 64])  
        #     h1 = F.max_pool2d(h1, 2)#h1_ :  torch.Size([256, 16, 32, 32])
        #     h2 = self.l2(h1) #h2 :  torch.Size([256, 4, 32, 32])
        #     h2 = F.max_pool2d(h2, 2)#h2 :  torch.Size([256, 16, 16, 16])　##max pool2dの引数(領域のサイズ、ストライド)
        #     h3 = self.l3(h2)
        #     h3 = self.up_samp(h3)
        #     out = self.l4(h3) #out.shape=torch.Size([32, 1, 32, 32])
        #     #print(f'{out.shape=}')#out.shape torch.Size([256, 10]) # [バッチサイズ,output.shape]
        #     out = self.up_samp(out)
        #     #print(f'{out.shape=}')#out.shape=torch.Size([32, 1, 64, 64])

        #     #print("out.shape",out.shape) #out[0].shape torch.Size([10])
        #     #print("sum out[0]:",sum(out[0]))  #tensor([1., 0., 1., 0., 1., 0., 1., 1., 0., 1.], device='cuda:0',
        #     #sum_out = out if sum_out is Nonec else sum_out + out
        #     out_rec.append(out)##64×64×1channelの画像を配列に入れる。0番目のデータ(line81)+time data 11の合計12個
        #     #print(out_rec.dtype)
            ######
        
        ##skip connection##
        for t in range(self.num_time):
            # print(f'{x.shape=}')#x.shape=torch.Size([32, 4096, 11])
            x_t = x[:,:,t]  ##各ステップ数の64pix × 64pixをx_tに入れる。
            # print(f'{x_t.shape=}')#x_t.shape=torch.Size([32, 4096])
            # print(x_t.shape)
            # x_t = x_t.reshape((len(x_t), 1, 128, 128))
            x_t = x_t.reshape((len(x_t), 1, pixel, pixel)) ##torch.Size([32, 1, 64, 64]) ##len(x_t)=32//バッチサイズ?
            # print(f'{x_t.shape}')
            # print('*********************')
            #print("x_t : ",x_t.shape)
            conv_1 = self.l1(x_t) # h1 :  torch.Size([256, 16, 64, 64])　##最初のconv後の特徴マップ##encoder側結合箇所1  
            pool_1 = F.max_pool2d(conv_1, 2)#h1_ :  torch.Size([256, 16, 32, 32])
            conv_2 = self.l2(pool_1) #h2 :  torch.Size([256, 4, 32, 32]) ##二回目のconv後の特徴マップ##encoder側結合箇所2
            #print(f'{conv_2.shape=}')#torch.Size([32, 8, 32, 32]) 
            pool_2 = F.max_pool2d(conv_2, 2)#h2 :  torch.Size([256, 16, 16, 16])　##max pool2dの引数(領域のサイズ、ストライド)
            #print(f'{pool_2.shape=}')#torch.Size([32, 8, 16, 16])
            conv_3 = self.l3(pool_2)
            #print(f'{conv_3.shape=}')#torch.Size([32, 8, 16, 16]) 
            unpool_1 = self.up_samp(conv_3) ##初回アップサンプリング後の特徴マップ##decoder側結合箇所1'
            #print(f'{unpool_1.shape=}')#torch.Size([32, 8, 32, 32]) 
            ##skip connection1##
            concat_1 = torch.cat([conv_2,unpool_1],dim=1)##torch.Size([32, 16, 32, 32]) 
            #print(f'{concat_1.shape=}')
            conv_4 = self.l4(concat_1) #out.shape=torch.Size([32, 1, 32, 32])
            ####################
            # conv_4 = self.l4(unpool_1)
            # #print(f'{out.shape=}')#out.shape torch.Size([256, 10]) # [バッチサイズ,output.shape]
            unpool_2 = self.up_samp(conv_4)##二回目のアップサンプリング後の特徴マップ##decoder側結合箇所2'
            # #print(f'{out.shape=}')#out.shape=torch.Size([32, 1, 64, 64])

            #print("out.shape",out.shape) #out[0].shape torch.Size([10])
            #print("sum out[0]:",sum(out[0]))  #tensor([1., 0., 1., 0., 1., 0., 1., 1., 0., 1.], device='cuda:0',
            #sum_out = out if sum_out is Nonec else sum_out + out
            out_rec.append(unpool_2)##64×64×1channelの画像を配列に入れる。0番目のデータ(line81)+time data 11の合計12個
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
#################################################################################
""""""
#10/19(水):Gated-SNUのネットワークは変更せずに、channel数を変更。(パラメータはonenoteに記載)
#          conv2,unpool1でのconcatによるskip connectionを設計した。           