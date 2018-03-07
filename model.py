#coding: utf-8
import torch
from torch import nn
from layers import *
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

class Encoder(nn.Module):
    def __init__(self, repetition, final, type = 'conv'):
        super(Encoder, self).__init__()
        self.preblock = nn.Sequential(nn.Conv2d(3 , 32, kernel_size=5, stride=2,padding=2, bias=True),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU())
        self.model = nn.ModuleList([])
        input_feat = 32
        for n_layer in repetition:
            block = nn.ModuleList([])
            for layer in range(n_layer):
                if layer == 0:
                    if block != 0:
                        downsample = nn.Conv2d(input_feat , input_feat*2, kernel_size=1, stride=2)
                        block.append(BasicBlock(input_feat, input_feat*2, stride = 2, downsample=downsample))
                    else:
                        block.append(BasicBlock(input_feat, input_feat*2, stride = 1))
                    input_feat *= 2
                else:
                    block.append(BasicBlock(input_feat, input_feat, stride = 1))
            self.model.append( nn.Sequential(*block))
        self.type = type
        if self.type == 'conv':
            self.final = nn.Conv2d(512 , final, kernel_size=1, stride=1, bias=True)
        elif self.type == 'fc':
            self.final = nn.Sequential(nn.Conv2d(512 , 32, kernel_size=1, stride=1, bias=True),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Linear(32*6*5, final))
    def forward(self, x):
        x = self.preblock(x)
        for block in self.model:
            x = block(x)
        x = self.final(x)
        return x

class Decoder(nn.Module):
    def __init__(self, repetition, input_dim, temp_dim =0, up_method = 'deconv'):
        super(Decoder, self).__init__()
        self.temp_dim = temp_dim
        if temp_dim>0:
            self.template = nn.Parameter(torch.from_numpy(np.random.randn(temp_dim,6,5).astype('float32')).cuda())
        self.model = nn.ModuleList([])
        input_feat = input_dim + temp_dim
        for n_layer in repetition:
            block = nn.ModuleList([])
            for layer in range(n_layer):
                if layer == 0:
                    if block != 0:
                        if up_method == 'deconv':
                            upsample = nn.ConvTranspose2d(input_feat , input_feat/2, kernel_size=2, stride=2 )
                            block.append(BasicBlockt(input_feat, input_feat/2, stride = 2, upsample=upsample))
                        elif up_method == 'pixshuf':
                            block.append(nn.PixelShuffle(2))
                            downsample = nn.Conv2d(input_feat/4, input_feat/2, kernel_size=1 )
                            block.append(BasicBlock(input_feat/4, input_feat/2, downsample = downsample))
                    else:
                        block.append(BasicBlock(input_feat, input_feat/2))
                    input_feat /= 2
                else:
                    block.append(BasicBlock(input_feat, input_feat))
            self.model.append( nn.Sequential(*block))
            
        self.final = nn.Conv2d(16 , 3, kernel_size=1, stride=1, bias=True)
    def forward(self, x):
        if self.temp_dim>0:
            temp = self.template.expand(x.size()[:1]+self.template.size())
            x = torch.cat((x,temp),1)
        for block in self.model:
            x = block(x)
        x = self.final(x)
        return x
    
class Naive_ae(nn.Module):
    def __init__(self):
        super(Naive_ae, self).__init__()
        self.encoder = nn.Conv2d(3, 16, kernel_size = 8, stride = 8)
        self.decoder = nn.Conv2d(16, 8*8*3, kernel_size = 1)
        self.pixshuf = nn.PixelShuffle(8)
    def forward(self, x):
        feat = self.encoder(x)
        rec = self.pixshuf(self.decoder(feat))
        return rec


class Naive_ae_relu(nn.Module):
    def __init__(self):
        super(Naive_ae_relu, self).__init__()
        self.encoder = nn.Conv2d(3, 16, kernel_size = 8, stride = 8)
        self.decoder = nn.Conv2d(16, 8*8*3, kernel_size = 1)
        self.pixshuf = nn.PixelShuffle(8)
    def forward(self, x):
        feat = F.relu(self.encoder(x))
        rec = self.pixshuf(self.decoder(feat))
        return rec
    

class Naive_ae_deconv(nn.Module):
    def __init__(self):
        super(Naive_ae_deconv, self).__init__()
        self.encoder = nn.Conv2d(3, 16, kernel_size = 8, stride = 8)
        self.decoder = nn.ConvTranspose2d(16, 3, kernel_size = 8, stride = 8)
    def forward(self, x):
        feat = self.encoder(x)
        rec = self.decoder(feat)
        return rec

class Naive_ae_overlap(nn.Module):
    def __init__(self):
        super(Naive_ae_overlap, self).__init__()
        self.encoder = nn.Conv2d(3, 16, kernel_size = 8, stride = 4)
        self.decoder = nn.ConvTranspose2d(16, 3, kernel_size = 8, stride = 4)
        self.pad = nn.ReflectionPad2d(4)
    def forward(self, x):
        padx = self.pad(x)
        feat = self.encoder(padx)
        rec = self.decoder(feat)
        rec = rec[:, :, 4:-4,4:-4]
        return rec

class Naive_ae_1decode(nn.Module):
    def __init__(self):
        super(Naive_ae_1decode, self).__init__()
        self.pad = nn.ReflectionPad2d(4)
        self.encoder = nn.Conv2d(3, 16, kernel_size = 8, stride = 8)
        self.decoder = nn.Conv2d(16, 8*8*3, kernel_size = 1)
        self.pixshuf = nn.PixelShuffle(8)
        self.refine = nn.Conv2d(3, 3, kernel_size = 5, stride = 1, padding = 2)
    def forward(self, x):
        padx = self.pad(x)
        feat = self.encoder(padx)
        rec = self.pixshuf(self.decoder(feat))
        refine = rec + self.refine(rec)
        refine = refine[:, :, 4:-4,4:-4]
        return refine
    
class Naive_ae_2decode(nn.Module):
    def __init__(self):
        super(Naive_ae_2decode, self).__init__()
        self.pad = nn.ReflectionPad2d(4)
        self.encoder = nn.Conv2d(3, 16, kernel_size = 8, stride = 8)
        self.decoder = nn.Conv2d(16, 8*8*3, kernel_size = 1)
        self.pixshuf = nn.PixelShuffle(8)
        self.refine1 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
        self.refine2 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
    def forward(self, x, returnfeat = False):
        padx = self.pad(x)
        feat = self.encoder(padx)
        rec = self.pixshuf(self.decoder(feat))
        refine = rec + self.refine1(self.refine2(rec))
        refine = refine[:, :, 4:-4,4:-4]
        if not returnfeat:
            return refine
        else:
            return refine, feat
    
class Naive_ae_2decode_a(nn.Module):
    def __init__(self):
        super(Naive_ae_2decode_a, self).__init__()
        self.pad = nn.ReflectionPad2d(4)
        self.encoder = nn.Conv2d(3, 16, kernel_size = 8, stride = 8)
        self.decoder = nn.Conv2d(16, 8*8*3, kernel_size = 1)
        self.pixshuf = nn.PixelShuffle(8)
        self.refine1 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
        self.refine2 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
    def forward(self, x):
        padx = self.pad(x)
        feat = self.encoder(padx)
        rec = self.pixshuf(self.decoder(feat))
        refine = rec + self.refine1(F.relu(self.refine2(rec)))
        refine = refine[:, :, 4:-4,4:-4]
        return refine
    
class Naive_ae_2decode_b(nn.Module):
    def __init__(self):
        super(Naive_ae_2decode_b, self).__init__()
        self.pad = nn.ReflectionPad2d(4)
        self.encoder = nn.Conv2d(3, 16, kernel_size = 8, stride = 8)
        self.decoder = nn.Conv2d(16, 8*8*3, kernel_size = 1)
        self.pixshuf = nn.PixelShuffle(8)
        self.refine1 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
        self.refine2 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(3)
    def forward(self, x):
        padx = self.pad(x)
        feat = self.encoder(padx)
        rec = self.pixshuf(self.decoder(feat))
        refine = rec + self.bn2(self.refine1(F.relu(self.bn1(self.refine2(rec)))))
        refine = refine[:, :, 4:-4,4:-4]
        return refine

class Naive_ae_2decode_c(nn.Module):
    def __init__(self):
        super(Naive_ae_2decode_c, self).__init__()
        self.pad = nn.ReflectionPad2d(4)
        self.encoder = nn.Conv2d(3, 16, kernel_size = 8, stride = 8)
        self.decoder = nn.Conv2d(16, 8*8*3, kernel_size = 1)
        self.pixshuf = nn.PixelShuffle(8)
        self.refine1 = nn.Conv2d(3, 3, kernel_size = 5, stride = 1, padding = 2)
        self.refine2 = nn.Conv2d(3, 3, kernel_size = 5, stride = 1, padding = 2)
    def forward(self, x):
        padx = self.pad(x)
        feat = self.encoder(padx)
        rec = self.pixshuf(self.decoder(feat))
        refine = rec + self.refine1(F.relu(self.refine2(rec)))
        refine = smooth_clip(refine[:, :, 4:-4,4:-4], -1, 1)
        return refine

class Naive_ae_2decode_d(nn.Module):
    def __init__(self):
        super(Naive_ae_2decode_d, self).__init__()
        self.pad = nn.ReflectionPad2d(4)
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, kernel_size = 8, stride = 8),
                                    nn.PReLU())
        self.conv1 = nn.Conv2d(16, 96, kernel_size = 3, stride = 1, padding = 1)
        self.refine_mid1 = naive_resunit3(96, 0.1)
        self.refine_mid2 = naive_resunit3(96, 0.1)
        
        self.decoder = nn.Conv2d(96, 8*8*3, kernel_size = 1)
        self.pixshuf = nn.PixelShuffle(8)
        
        
    def forward(self, x, returnfeat = False):
        padx = self.pad(x)
        feat = self.encoder(padx)
        
        l1 = self.conv1(feat)
        res = self.refine_mid2(self.refine_mid1(l1))
        rec = self.pixshuf(self.decoder(l1+res))
        rec = rec[:, :, 4:-4,4:-4]
        if not returnfeat:
            return rec
        else:
            return rec, feat
        
class Naive_ae_2decode_e(nn.Module):
    def __init__(self):
        super(Naive_ae_2decode_e, self).__init__()
        self.pad = nn.ReflectionPad2d(4)
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, kernel_size = 8, stride = 8),
                                    nn.PReLU())
        self.conv1 = nn.Conv2d(16, 96, kernel_size = 3, stride = 1, padding = 1)
        self.refine_mid1 = naive_resunit3(96)
        self.refine_mid2 = naive_resunit3(96)
        
        self.decoder = nn.Conv2d(96, 8*8*3, kernel_size = 1)
        self.pixshuf = nn.PixelShuffle(8)
        
        
    def forward(self, x, returnfeat = False):
        padx = self.pad(x)
        feat = self.encoder(padx)
        
        l1 = self.conv1(feat)
        res = self.refine_mid2(self.refine_mid1(l1))
        rec = self.pixshuf(self.decoder(res))
        rec = rec[:, :, 4:-4,4:-4]
        if not returnfeat:
            return rec
        else:
            return rec, feat

class Naive_ae_2decode_f(nn.Module):
    def __init__(self):
        super(Naive_ae_2decode_f, self).__init__()
        self.pad = nn.ReflectionPad2d(4)
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, kernel_size = 8, stride = 8),
                                    nn.PReLU())
        self.conv1 = nn.Conv2d(16, 96, kernel_size = 3, stride = 1, padding = 1)
        self.refine_mid1 = naive_resunit3(96)
        self.refine_mid2 = naive_resunit3(96)
        
        self.decoder = nn.Conv2d(96, 8*8*3, kernel_size = 1)
        self.pixshuf = nn.PixelShuffle(8)
        
        
    def forward(self, x, returnfeat = False):
        padx = self.pad(x)
        feat = self.encoder(padx)
        
        l1 = self.conv1(feat)
        res = self.refine_mid2(self.refine_mid1(l1+res))
        rec = self.pixshuf(self.decoder(res))
        rec = rec[:, :, 4:-4,4:-4]
        if not returnfeat:
            return rec
        else:
            return rec, feat

class Naive_ae_4decode(nn.Module):
    def __init__(self):
        super(Naive_ae_4decode, self).__init__()
        self.pad = nn.ReflectionPad2d(4)
        self.encoder = nn.Conv2d(3, 16, kernel_size = 8, stride = 8)
        self.decoder = nn.Conv2d(16, 8*8*3, kernel_size = 1)
        self.pixshuf = nn.PixelShuffle(8)
        self.refine1 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
        self.refine2 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
        self.refine3 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
        self.refine4 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
    def forward(self, x, returnfeat = False):
        padx = self.pad(x)
        feat = self.encoder(padx)
        rec = self.pixshuf(self.decoder(feat))
        refine = rec + self.refine1(self.refine2(rec))
        refine = rec + self.refine3(self.refine4(rec))
        refine = smooth_clip(refine[:, :, 4:-4,4:-4], -1, 1)
        if not returnfeat:
            return refine
        else:
            return refine, feat

        
class Naive_ae_4decode_a(nn.Module):
    def __init__(self):
        super(Naive_ae_4decode_a, self).__init__()
        self.pad = nn.ReflectionPad2d(4)
        self.encoder = nn.Conv2d(3, 16, kernel_size = 8, stride = 8)
        self.decoder = nn.Conv2d(16, 8*8*3, kernel_size = 1)
        self.pixshuf = nn.PixelShuffle(8)
        self.refine1 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
        self.refine2 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
        self.refine3 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
        self.refine4 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
    def forward(self, x, returnfeat = False):
        padx = self.pad(x)
        feat = self.encoder(padx)
        rec = self.pixshuf(self.decoder(feat))
        refine = rec + self.refine1(self.refine2(rec))
        refine = rec + self.refine3(F.relu(self.refine4(rec)))
        refine = smooth_clip(refine[:, :, 4:-4,4:-4],-1,1)
        if not returnfeat:
            return refine
        else:
            return refine, feat

class Naive_ae_8decode(nn.Module):
    def __init__(self):
        super(Naive_ae_8decode, self).__init__()
        self.pad = nn.ReflectionPad2d(4)
        self.encoder = nn.Conv2d(3, 16, kernel_size = 8, stride = 8)
        self.decoder = nn.Conv2d(16, 8*8*3, kernel_size = 1)
        self.pixshuf = nn.PixelShuffle(8)
        self.refine1 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
        self.refine2 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
        self.refine3 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
        self.refine4 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
        self.refine5 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
        self.refine6 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
        self.refine7 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
        self.refine8 = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x, returnfeat = False):
        padx = self.pad(x)
        feat = self.encoder(padx)
        rec = self.pixshuf(self.decoder(feat))
        refine = rec + self.refine1(self.refine2(rec))
        refine = rec + self.refine3(self.refine4(rec))
        refine = rec + self.refine5(self.refine6(rec))
        refine = rec + self.refine7(self.refine8(rec))
        refine = smooth_clip(refine[:, :, 4:-4,4:-4], -1, 1)
        if not returnfeat:
            return refine
        else:
            return refine, feat

        
class Naive_ae_3shuf(nn.Module):
    def __init__(self):
        super(Naive_ae_3shuf, self).__init__()
        self.pad = nn.ReflectionPad2d(4)
        self.encoder = nn.Conv2d(3, 16, kernel_size = 8, stride = 8)
        self.decoder1 = nn.Conv2d(16, 16*4, kernel_size = 1)
        self.decoder2 = nn.Conv2d(16, 16*4, kernel_size = 1)
        self.decoder3 = nn.Conv2d(16, 16*4, kernel_size = 1)
        self.pixshuf = nn.PixelShuffle(2)
        self.final_conv = nn.Conv2d(16, 3, kernel_size = 1)
    def forward(self, x, returnfeat = False):
        padx = self.pad(x)
        feat = self.encoder(padx)
        stag1 = self.pixshuf(self.decoder1(feat))
        stag2 = self.pixshuf(self.decoder2(stag1))
        stag3 = self.pixshuf(self.decoder3(stag2))
        rec = self.final_conv(stag3)
        refine = smooth_clip(rec[:, :, 4:-4,4:-4], -1, 1)
        if not returnfeat:
            return refine
        else:
            return refine, feat
        
class Naive_ae_3shuf_a(nn.Module):
    def __init__(self):
        super(Naive_ae_3shuf_a, self).__init__()
        self.pad = nn.ReflectionPad2d(4)
        self.encoder = nn.Conv2d(3, 16, kernel_size = 8, stride = 8)
        self.decoder1 = nn.Conv2d(16, 64*4, kernel_size = 1)
        self.refine1 = naive_resunit(64)
        self.decoder2 = nn.Conv2d(64, 64*4, kernel_size = 1)
        self.refine2 = naive_resunit(64)
        self.decoder3 = nn.Conv2d(64, 32*4, kernel_size = 1)
        self.refine3 = naive_resunit(32)
        self.pixshuf = nn.PixelShuffle(2)
        self.final_conv = nn.Conv2d(32, 3, kernel_size = 1)
    def forward(self, x, returnfeat = False):
        padx = self.pad(x)
        feat = self.encoder(padx)
        stag1 = self.refine1(self.pixshuf(self.decoder1(feat)))
        stag2 = self.refine2(self.pixshuf(self.decoder2(stag1)))
        stag3 = self.refine3(self.pixshuf(self.decoder3(stag2)))
        rec = self.final_conv(stag3)
        refine = smooth_clip(rec[:, :, 4:-4,4:-4], -1, 1)
        if not returnfeat:
            return refine
        else:
            return refine, feat

class Naive_ae_3shuf_b(nn.Module):
    def __init__(self):
        super(Naive_ae_3shuf_b, self).__init__()
        self.pad = nn.ReflectionPad2d(4)
        self.encoder = nn.Conv2d(3, 16, kernel_size = 8, stride = 8)
        
        self.decoder1 = nn.Conv2d(16, 64*4, kernel_size = 1)
        self.refine1 = naive_resunit(64)
        self.decoder2 = nn.Conv2d(64, 64*4, kernel_size = 1)
        self.refine2 = naive_resunit(64)
        self.decoder3 = nn.Conv2d(64, 32*4, kernel_size = 1)
        self.refine3 = naive_resunit(32)
        self.pixshuf = nn.PixelShuffle(2)
        self.final_conv = nn.Conv2d(32, 3, kernel_size = 1)
        
        self.decoder_skip = nn.Conv2d(16, 8*8*3, kernel_size = 1)
        self.refine1_skip = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)
        self.refine2_skip = nn.Conv2d(3, 3, kernel_size = 3, stride = 1, padding = 1)

        self.pixshuf2 = nn.PixelShuffle(8)
        
        
    def forward(self, x, returnfeat = False):
        padx = self.pad(x)
        feat = self.encoder(padx)
        
        stag1 = self.refine1(self.pixshuf(self.decoder1(feat)))
        stag2 = self.refine2(self.pixshuf(self.decoder2(stag1)))
        stag3 = self.refine3(self.pixshuf(self.decoder3(stag2)))
        rec1 = self.final_conv(stag3)
        
        rec2 = self.pixshuf2(self.decoder_skip(feat))
        rec2 = rec2 + self.refine1_skip(self.refine2_skip(rec2))
        
        rec = rec1+rec2
        refine = smooth_clip(rec[:, :, 4:-4,4:-4], -1, 1)
        if not returnfeat:
            return refine
        else:
            return refine, feat

class Naive_ae_3shuf_c(nn.Module):
    def __init__(self):
        super(Naive_ae_3shuf_c, self).__init__()
        self.pad = nn.ReflectionPad2d(4)
        self.encoder = nn.Conv2d(3, 16, kernel_size = 8, stride = 8)
        self.decoder1 = nn.Conv2d(16, 64*4, kernel_size = 1)
        self.refine1 = naive_resunit2(64)
        self.decoder2 = nn.Conv2d(64, 64*4, kernel_size = 1)
        self.refine2 = naive_resunit2(64)
        self.decoder3 = nn.Conv2d(64, 32*4, kernel_size = 1)
        self.refine3 = naive_resunit2(32)
        self.pixshuf = nn.PixelShuffle(2)
        self.final_conv = nn.Conv2d(32, 3, kernel_size = 1)
    def forward(self, x, returnfeat = False):
        padx = self.pad(x)
        feat = self.encoder(padx)
        stag1 = self.refine1(self.pixshuf(self.decoder1(feat)))
        stag2 = self.refine2(self.pixshuf(self.decoder2(stag1)))
        stag3 = self.refine3(self.pixshuf(self.decoder3(stag2)))
        rec = self.final_conv(stag3)
        refine = smooth_clip(rec[:, :, 4:-4,4:-4], -1, 1)
        if not returnfeat:
            return refine
        else:
            return refine, feat


class Naive_ae_1res_decode(nn.Module):
    def __init__(self):
        super(Naive_ae_1res_decode, self).__init__()
        self.pad = nn.ReflectionPad2d(4)
        self.encoder = nn.Conv2d(3, 16, kernel_size = 8, stride = 8)
        self.decoder = nn.Conv2d(16, 8*8*3, kernel_size = 1)
        self.pixshuf = nn.PixelShuffle(8)
        self.refine = BasicBlock(3, 3, stride = 1)
    def forward(self, x):
        padx = self.pad(x)
        feat = self.encoder(padx)
        rec = self.pixshuf(self.decoder(feat))
        refine = rec + self.refine(rec)
        refine = refine[:, :, 4:-4,4:-4]
        return refine

class Naive_ae_1encode(nn.Module):
    def __init__(self):
        super(Naive_ae_1encode, self).__init__()
        self.pad = nn.ReflectionPad2d(4)
        self.encoder = nn.Conv2d(3, 16, kernel_size = 8, stride = 8)
        self.decoder = nn.Conv2d(16, 8*8*3, kernel_size = 1)
        self.pixshuf = nn.PixelShuffle(8)
        self.refine = nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1)
    def forward(self, x):
        padx = self.pad(x)
        feat = self.encoder(padx)
        feat = feat + self.refine(feat)

        rec = self.pixshuf(self.decoder(feat))
        rec = rec[:, :, 4:-4,4:-4]
        return rec

class Model_ae_temp(nn.Module):
    def __init__(self, rep_enc, rep_dec):
        super(Model_ae_temp, self).__init__()
        self.encoder = Encoder(rep_enc, final =32)
        self.decoder = Decoder(rep_dec, input_dim = 32, temp_dim = 480)
        
    
    def forward(self, x):
        feat = self.encoder(x)
        rec = self.decoder(feat)
        return rec


class Model_ae(nn.Module):
    def __init__(self, rep_enc, rep_dec):
        super(Model_ae, self).__init__()
        self.encoder = Encoder(rep_enc, final =512)
        self.decoder = Decoder(rep_dec, input_dim = 512)
        
    def forward(self, x, returnfeat = False):
        feat = self.encoder(x)
        rec = self.decoder(feat)
        if not returnfeat:
            return rec
        else:
            return rec, feat
    
class Model_ae_temp_shuf(nn.Module):
    def __init__(self, rep_enc, rep_dec):
        super(Model_ae_temp_shuf, self).__init__()
        self.encoder = Encoder(rep_enc, final =32)
        self.decoder = Decoder(rep_dec, input_dim = 32, temp_dim = 480, up_method = 'pixshuf')
        
    
    def forward(self, x, returnfeat = False):
        feat = self.encoder(x)
        rec = self.decoder(feat)
        if not returnfeat:
            return rec
        else:
            return rec, feat

class Model_ae_shuf(nn.Module):
    def __init__(self, rep_enc, rep_dec):
        super(Model_ae_shuf, self).__init__()
        self.encoder = Encoder(rep_enc, final =512, up_method = 'pixshuf')
        self.decoder = Decoder(rep_dec, input_dim = 512)
        
    def forward(self, x, returnfeat = False):
        feat = self.encoder(x)
        rec = self.decoder(feat)
        if not returnfeat:
            return rec
        else:
            return rec, feat

class Model_cae(nn.Module):
    def __init__(self):
        super(Model_cae, self).__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(self._npad(3)),
            nn.Conv2d(3, 64, kernel_size=5, stride=2),
            nn.Conv2d(64,128,kernel_size=5, stride=2),
            naive_resunit(128),
            naive_resunit(128),
            nn.Conv2d(128,96, kernel_size=5, stride=2))
        
        self.decode = nn.Sequential(
            nn.Conv2d(96,512,kernel_size=3,stride=1, padding=1),
            nn.PixelShuffle(2),
            naive_resunit(128),
            naive_resunit(128),
            nn.Conv2d(128,256,kernel_size=3,stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64,12,kernel_size=3,stride=1, padding=1),
            nn.PixelShuffle(2))
    def forward(self,x, returnfeat = False):
        feat = smooth_round(self.encoder(x))
        rec = smooth_clip(self.decode(feat),-1,1)
       # recon = self.decode(code)
        if not returnfeat:
            return rec
        else:
            return rec, feat
    
    def _npad(self, n_downsample):
        in_shape = 2**n_downsample*10
        new = in_shape/(2**n_downsample)
        for i in range(n_downsample):
            new = new*2+4
        return int((new-in_shape)/2)

class Gain_estimator(nn.Module):   #5层卷积来获得缩放系数
    def __init__(self):
        super(Gain_estimator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride = 2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, 3, stride = 2, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, 32, 3, stride = 2, padding=1, bias=True)
        self.conv4 = nn.Conv2d(32, 32, 3, stride = 2, padding=1, bias=True)
        self.conv5 = nn.Conv2d(32, 1, 3, stride = 2, padding=1, bias=True) #最后一层写的有问题，原论文中是2x2
    
    def forward(self, x):         #通过x来求新的缩放系数
        out = F.elu(self.conv1(x))
        out = F.elu(self.conv2(out))
        out = F.elu(self.conv3(out))
        out = F.elu(self.conv4(out))
        out = F.elu(self.conv5(out)) + 2

        return out

class Model_Full(nn.Module):
    def __init__(self):
        super(Model_Full, self).__init__()
        self.rnnType = "ConvGRU"
        self.reconstructType = "One-shot"
        self.encoder = Full_encoder(self.rnnType)
        self.binarizer = New_Binarizer()
        self.decoder = Full_decoder(self.rnnType)
        self.Gain_estimator = Gain_estimator()
        
    def forward(self, input):
        for iter in range(1,15):
            if iter == 1:
                batch, channel, height, width = input.size()
                self.ground = input     
                self.reconstruct = Variable(torch.zeros(batch, channel, height, width)).cuda()  
            output = self.decoder(self.binarizer(self.encoder(input, iter)), iter)

            if self.reconstructType == "One-shot":
            	self.reconstruct = output
            elif self.reconstructType == "Additive":
            	self.reconstruct = output + self.reconstruct
            #self.gain = self.Gain_estimator(self.restruct)[0]
            input = self.ground - self.reconstruct
        return self.reconstruct

def get_model(model_name):
    if model_name == 'ae_temp1':
        return Model_ae_temp([2,2,2,2], [2,2,2,2,2])
    elif model_name == 'ae1':
        return Model_ae([2,2,2,2], [2,2,2,2,2])
    elif model_name == 'ae_temp_shuf1':
        return Model_ae_temp_shuf([2,2,2,2], [2,2,2,2,2])
    elif model_name == 'ae_shuf1':
        return Model_ae_temp_shuf([2,2,2,2], [2,2,2,2,2])
    elif model_name == 'naive_patch':
        return Naive_ae()
    elif model_name == 'naive_patch_relu':
        return Naive_ae_relu()
    elif model_name == 'naive_overlap':
        return Naive_ae_overlap()
    elif model_name == 'naive_deconv':
        return Naive_ae_deconv()
    elif model_name == 'naive_1decode':
        return Naive_ae_1decode()
    elif model_name == 'naive_1encode':
        return Naive_ae_1encode()
    elif model_name == 'naive_1res_decode':
        return Naive_ae_1res_decode()
    elif model_name == 'naive_2decode':
        return Naive_ae_2decode()
    elif model_name == 'naive_2decode_a':
        return Naive_ae_2decode_a()
    elif model_name == 'naive_2decode_b':
        return Naive_ae_2decode_b()
    elif model_name == 'cae':
        return Model_cae()
    elif model_name == 'full':
    	return Model_Full()
    else:
        model = eval(model_name+'()')
        return model
