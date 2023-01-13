# -*- coding: utf-8 -*-

import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
#import torch.nn.parameter as Parameter

import torch.optim as optim
import torchvision
import numpy as np
from torch import cuda

import numpy 
from . import step_func
#import step_func

class SNU(nn.Module):
    def __init__(self, in_channels, out_channels, l_tau=0.8, soft=False, rec=False, nobias=False, initial_bias=-0.5, gpu=True):
        super(SNU,self).__init__()

        self.in_channels = in_channels
        self.out_channels= out_channels
        self.l_tau = l_tau
        self.rec = rec
        self.soft = soft
        self.gpu = gpu
        self.s = None
        self.y = None
        self.initial_bias = initial_bias

        if self.gpu:
            #xp = cuda.cupy
            dtype = torch.float
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dtype = torch.float
            device=torch.device("cpu")
        
        #self.w1 = torch.empty((n_in, n_out),  device=device, dtype=dtype, requires_grad=True)
        #torch.nn.init.normal_(self.w1, mean=0.0)
        
        #self.Wx = torch.einsum("abc,cd->abd", (x_data, w1))
        #self.Wx = nn.Linear(4374, out_channels, bias=False).to(device)
        self.Wx = nn.Linear(in_channels, out_channels, bias=False).to(device)
        #nn.init.uniform_(self.Wx.weight, -0.1, 0.1) #3.0
        torch.nn.init.xavier_uniform_(self.Wx.weight)

    

        if nobias:##ここfalseで飛ばす
            self.b = None
        else:

            #print("initial_bias",initial_bias)
            device = torch.device(device)
            
            self.b = nn.Parameter(torch.Tensor([initial_bias]).to(device))
            #print("self.b",self.b)
                            
    def reset_state(self, s=None, y=None):
        self.s = s
        self.y = y

    def initialize_state(self, shape):
        if self.gpu:
            #xp = cuda.cupy
            dtype = torch.float
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dtype = torch.float
            device=torch.device("cpu")
            
        self.s = torch.zeros((shape[0], self.out_channels),device=device,dtype=dtype)
        self.y = torch.zeros((shape[0], self.out_channels),device=device,dtype=dtype)
              
    def forward(self,x):
        if self.s is None:
            #print("self.s is none")
            self.initialize_state(x.shape)


        if type(self.s) == numpy.ndarray:
            self.s = torch.from_numpy(self.s.astype(np.float32)).clone()
    
        print("x in snu.shape",x.shape) #x in snu.shape torch.Size([256, 784])        
        #print("self.Wx(x).shape",self.Wx(x).shape)
        #print("self.s.shape : ",self.s.shape)
        
        # s = F.elu(abs(self.Wx(x)) + self.l_tau * self.s * (1-self.y))
        ##超重要:ここで膜電位の計算が行われている##
        ## Wx(x):入力、F:ReLu関数、l_tau:1ステップ前の膜電位を引き継ぐ割合(今回は0.8)、s:1ステップ前の膜電位、y:出力(スパイク)
        ##yが1の時に最終項は0に等しくなるので、このとき膜電位が初期値にリセットされることになる##
        s = F.elu(abs(self.Wx(x))*0.1 + self.l_tau * self.s * (1-self.y))
        #print("s : ",s)

        if self.soft:##soft = falseであるのでelseに飛ぶ

            axis = 1
            bias_ = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)]
            #print("bias_:",bias_)
            y = F.sigmoid(bias_)

        else:
            axis = 0
            #print("s.shape:", s.shape)
            #print("self.b.shape:", self.b.shape)
            #print("self.initial_bias.shape:",self.initial_bias.shape)
            #print("self.b.shape !!!!!!!!!!!!!!!! ", self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)].shape)
            bias = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)] #error!! two types
            #print("bias:",bias)
            #print("s in snu:",s)
            bias = s + self.b

            ##ここで非線形変換にステップ関数を用いている##
            y = step_func.spike_fn(bias)
        
        self.s = s
        self.y = y

        return y
########################################################################################################
class Conv_SNU(nn.Module):##エンコーダー部分
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, l_tau=0.8, soft=False, rec=False, forget=False, dual=False,nobias=False, initial_bias=-0.5, gpu=True):
        super(Conv_SNU,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size##カーネル（特徴マップ抽出のための正方行列)
        self.stride = stride##カーネルの移動数
        self.padding = padding##0パディングを外周追加(デコーダの際のカーネルで外周の余白を追加)
        self.l_tau = l_tau##膜電位を引き継ぐ割合
        self.rec = rec##RNNの再帰
        self.forget = forget##忘却ゲートのみ
        self.dual = dual##忘却ゲート+スパイク再帰
        self.soft = soft
        self.gpu = gpu
        self.s = None##膜電位
        self.y = None##出力スパイク
        self.initial_bias = initial_bias
        print("==== self.rec ====",rec)
        print("=== GPU ===",self.gpu)
        print("==== self.forget ====",self.forget)
        print(" ==== dual Gate ====",self.dual)
        if self.gpu:
            
            dtype = torch.float
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dtype = torch.float
            device=torch.device("cpu")

        self.Wx = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
        torch.nn.init.xavier_uniform_(self.Wx.weight)
        #print("self.rec in Conv_SNU",self.rec)
        if rec:
            print("recだよー")
            self.Wy = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Wy.weight)
            self.Wi = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Wi.weight)
            self.Ri = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Ri.weight)
        if forget:
            # 膜電位忘却ゲート
            #print("forgetだよー")
            self.Wf = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Wf.weight,0.1)
            self.Rf = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Rf.weight,0.1)
        if dual:
            # スパイク再突入　＋　膜電位忘却ゲート
            ##重みの初期化?
            self.Wy = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Wy.weight)
            self.Wi = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Wi.weight)
            self.Ri = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Ri.weight)
            self.Wf = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Wf.weight,0.1)
            self.Rf = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Rf.weight,0.1)

        if nobias:
            self.b = None
        else:
            device = torch.device(device)
            self.b = nn.Parameter(torch.Tensor([initial_bias]).to(device))

    def reset_state(self, s=None, y=None):
        self.s = s ##膜電位
        self.y = y ##出力スパイク

    def initialize_state(self, shape): #shape (バッチ,tチャネル,oh,ow)
        if self.gpu:
            dtype = torch.float
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dtype = torch.float
            device=torch.device("cpu")
        #####マックスプーリング処理(ストライド2で実行)#####
        self.oh = int(((shape[2] + 2*self.padding - self.kernel_size)/self.stride) + 1) # OH=H+2*P-FH/s +1 ##マックスプーリングで、ストライド2の場合での出力縦サイズ(確認済み)
        self.ow = int(((shape[3] + 2*self.padding - self.kernel_size)/self.stride) + 1)##↑同様。こっちはおそらく横サイズ(結局正方型で出力するので縦横は多分関係ない)
        ################################################

        ###########\dem_conv_classification.py  #########??????#########
        self.s = torch.zeros((shape[0], self.out_channels, self.oh, self.ow),device=device,dtype=dtype)
        self.y = torch.zeros((shape[0], self.out_channels, self.oh, self.ow),device=device,dtype=dtype)
        ################################################################
        ############dem_autoencoder_segmentation.py
        #self.s = torch.zeros((shape[0], self.out_channels, shape[2], shape[3]),device=device,dtype=dtype)
        #self.y = torch.zeros((shape[0], self.out_channels, shape[2], shape[3]),device=device,dtype=dtype)
        #self.Wrs = nn.Parameter(torch.empty((shape[0], self.out_channels, self.oh, self.ow),device=device,dtype=dtype))
        #self.br = nn.Parameter(torch.empty((shape[0], self.out_channels, self.oh, self.ow),device=device,dtype=dtype))
    
    def forward(self,x):
        if self.s is None:
            self.initialize_state(x.shape)

        if type(self.s) == numpy.ndarray:
            self.s = torch.from_numpy(self.s.astype(np.float32)).clone()
    
        #print('=self.Wy(self.y)',self.Wy(self.y).shape)
        #print('=self.Wx(x)',self.Wx(x).shape)
        if self.rec:
            print("rec yessss")
            #f = torch.sigmoid(self.Wf(x) + self.Rf(self.y))
            # spike 再入力ゲート
            i = torch.sigmoid(self.Wi(x) + self.Ri(self.y))
            s = F.elu(abs(self.Wx(x)) + i*self.Wy(self.y) + self.l_tau * self.s * (1-self.y))
        if self.forget:
            #print("forget yesssss")
            # 膜電位忘却ゲート
            f = torch.sigmoid(self.Wf(x) + self.Rf(self.y))
            s = F.elu(abs(self.Wx(x)) + (self.l_tau-f) * self.s * (1-self.y))
        if self.dual:#Gatedはここ
            #print("dual Gate yesssss")
            i = torch.sigmoid(self.Wi(x) + self.Ri(self.y))##入力
            f = torch.sigmoid(self.Wf(x) + self.Rf(self.y))##忘却
            s = F.elu(abs(self.Wx(x)) + i*self.Wy(self.y) + (self.l_tau-f) * self.s * (1-self.y))##膜電位(ReLU活性化関数)
            #print('i',i.shape)
            #print('f',f.shape)
            #print('s',s.shape)
        else:#ただのSNU
            #print("rec Noooooo")
            s = F.elu(abs(self.Wx(x)) + self.l_tau * self.s * (1-self.y))
        #s = F.elu(abs(self.Wx(x)) + r * self.s * (1-self.y))

        if self.soft:##false判定なのでelseへ飛ぶ

            axis = 1
            bias_ = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)]
            #print("bias_:",bias_)
            y = F.sigmoid(bias_)

        else:
            axis = 0
            bias = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)] #error!! two types
            bias = s + self.b
            y = step_func.spike_fn(bias)
        
        self.s = s
        self.y = y

        return y
##############################################################################################################

class tConv_SNU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=0, l_tau=0.5, soft=False, rec=False, nobias=False, initial_bias=-0.5, gpu=True):
        super(tConv_SNU,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.l_tau = l_tau
        self.rec = rec
        self.soft = soft
        self.gpu = gpu
        self.s = None
        self.y = None
        self.initial_bias = initial_bias

        if self.gpu:
            #xp = cuda.cupy
            dtype = torch.float
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dtype = torch.float
            device=torch.device("cpu")
        

        self.Wx = nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
        torch.nn.init.xavier_uniform_(self.Wx.weight)
        if self.rec==True:

            self.Wy = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=1, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Wy.weight)
            """
            # 膜電位忘却ゲート
            self.Wf = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Wf.weight)
            self.Rf = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Rf.weight)
            """
        if nobias:
            self.b = None
        else:

            #print("initial_bias",initial_bias)
            device = torch.device(device)
            self.b = nn.Parameter(torch.Tensor([initial_bias]).to(device))
            #print("self.b",self.b)

    def reset_state(self, s=None, y=None):
        self.s = s
        self.y = y

    def initialize_state(self, shape):
        if self.gpu:
            #xp = cuda.cupy
            dtype = torch.float
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dtype = torch.float
            device=torch.device("cpu")
            
        self.s = torch.zeros((shape[0], self.out_channels, 2*shape[2], 2*shape[3]),device=device,dtype=dtype)
        self.y = torch.zeros((shape[0], self.out_channels, 2*shape[2], 2*shape[3]),device=device,dtype=dtype)
        #self.Wrs = nn.Parameter(torch.empty((shape[0], self.out_channels, 2*shape[2], 2*shape[3]),device=device,dtype=dtype))
        #self.br = nn.Parameter(torch.empty((shape[0], self.out_channels, 2*shape[2], 2*shape[3]),device=device,dtype=dtype))
    
    def forward(self,x):
        if self.s is None:
            #print("self.s is none")
            self.initialize_state(x.shape)
        #print(self.l_tau)
        #print("rec:",self.rec)
        #"print('self.Wy(self.y)',self.Wy(self.y).shape)
        #print('self.Wx(x)',self.Wx(x).shape)
        if type(self.s) == numpy.ndarray:
            self.s = torch.from_numpy(self.s.astype(np.float32)).clone()
    
        if self.rec==True:
           # f = torch.sigmoid(self.Wf(x) + self.Rf(self.y))
           # spike 再入力ゲート
            i = torch.sigmoid(self.Wx(x) + self.Wy(self.y))
            s = F.elu(abs(self.Wx(x)) + i*self.Wy(self.y) + self.l_tau* self.s * (1-self.y))
        else:
            s = F.elu(abs(self.Wx(x)) + self.l_tau * self.s * (1-self.y))
        #s = F.elu(abs(self.Wx(x)) + r * self.s * (1-self.y))
        

        if self.soft:

            axis = 1
            bias_ = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)]
            #print("bias_:",bias_)
            y = F.sigmoid(bias_)

        else:
            axis = 0

            #print("s.shape:", s.shape)
            #print("self.b.shape:", self.b.shape)
            #print("self.initial_bias.shape:",self.initial_bias.shape)
            #print("self.b.shape !!!!!!!!!!!!!!!! ", self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)].shape)
            bias = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)] #error!! two types
            bias = s + self.b
            y = step_func.spike_fn(bias)
        
        self.s = s
        self.y = y

        return y