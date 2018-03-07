#coding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F



class Econv(nn.Module):  #E-conv1
    def __init__(self, input_channel, output_channel, stride = 1, bias = True):
        super(Econv, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.stride = stride
        self.bias = bias
        self.conv = nn.Conv2d(self.input_channel, self.output_channel, kernel_size=3, stride=self.stride, padding=1, bias=self.bias)
    
    def forward(self, x):
        return self.conv(x)

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, stride = 1, kernel_size=3, kernel_size_hidden=3):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_features = 4
        self.kernel_size = kernel_size
        self.kernel_size_hidden = kernel_size_hidden

        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(self.input_channels, 
                              4 * self.hidden_channels, 
                              kernel_size, 
                              stride = stride, 
                              padding = padding, 
                              bias = True)
        
        padding_hidden = int((kernel_size_hidden - 1) / 2)
        self.conv_hidden = nn.Conv2d(self.hidden_channels, 
                                     4 * self.hidden_channels, 
                                     kernel_size_hidden, 
                                     stride=1, 
                                     padding = padding_hidden, 
                                     bias=False)

    def forward(self, input, h, c):           #LSTM的一步迭代，每层LSTM将h作为下一层LSTM的输入，c,h留存下来用于下一次迭代
        A = self.conv(input) + self.conv_hidden(h)
        (af, ai, ao, ag) = torch.split(A, int(A.size()[1] / self.num_features), dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c

    @staticmethod
    def init_hidden(batch_size, hidden_c, shape):
        return (Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])).cuda())


class Binarizer(nn.Module):    #没写二值化过程呢
    def __init__(self):
        self.Bconv = Econv(512, 32)

    def forward(self, x):
        x = self.Bconv(x)


class Encoder(nn.Module):
    def __init__(self, depth_list, stride_list):     #depth_list是一个列表，里面有四个数，分别代表encoder中四层的output_channel, stride_list中的四个数代表了stride
        super(Encoder, self).__init__()
        self.depth_list = depth_list
        self.stride_list = stride_list
        self.Econv = Econv(3, depth_list[0], stride_list[0])
        self.ERnn1 = ConvLSTMCell(depth_list[0], depth_list[1], stride_list[1])
        self.ERnn2 = ConvLSTMCell(depth_list[1], depth_list[2], stride_list[2])
        self.ERnn3 = ConvLSTMCell(depth_list[2], depth_list[3], stride_list[3])
        self.stored_h = []
        self.stored_c = []      #储存3层LSTM的隐藏层输出
        self.step = 0           #迭代次数，每次forward加一
        
    def forward(self, x):
        x = self.Econv(x)
        if self.step == 0:      #第一次迭代的时候将三层LSTM的h,c初始化
            bsize, _, height, width = x.size()
            for i in range(3):
                (h, c) = ConvLSTMCell.init_hidden(bsize, self.depth_list[i], (int(height/self.stride_list[i]), int(width/self.stride_list[i])))
                self.stored_c.append(c)
                self.stored_h.append(h)
        
        self.step += 1

        h0, c0 = self.ERnn1(x, self.stored_h[0], self.stored_c[0])
        self.stored_c[0] = c0 
        self.stored_h[0] = h0     #更新第一层的h,c

        h1, c1 = self.ERnn2(h0, self.stored_h[1], self.stored_c[1])
        self.stored_c[1] = c1 
        self.stored_h[1] = h1     #更新第二层的h,c

        h2, c2 = self.ERnn3(h1, self.stored_h[2], self.stored_c[2])
        self.stored_c[2] = c2 
        self.stored_h[2] = h2     #更新第三层的h,c

        return h2






