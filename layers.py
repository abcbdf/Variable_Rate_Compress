import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv2x2t(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlockt(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(BasicBlockt, self).__init__()
        if stride == 2:
            self.conv1 = conv2x2t(inplanes, planes, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def preblock(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def _blank(self, x):
        return x
    
    def forward(self, x, feat_select = 1):
        features = []        
        feat_layers = [self._blank, self.preblock, self.layer1, self.layer2, self.layer3, self.layer4]
        if feat_select == 1:
            feat_select = [1]*len(feat_layers)
        highest_layer = [i+1 for i,v in enumerate(feat_select) if v] [-1]
        
        for i in range(highest_layer):
            x = feat_layers[i](x)
            if feat_select:
                features.append(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

        return features

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

class ConvGRU(nn.Module):
    def __init__(self, input_channels, hidden_channels, stride = 1, kernel_size = 3, kernel_size_hidden = 3):
        super(ConvGRU, self).__init__()
        self.stride = stride
        self.hidden_channels = hidden_channels
        padding = int((kernel_size - 1) / 2)
        padding_hidden = int((kernel_size_hidden - 1) / 2)
        self.conv = nn.ModuleList([nn.Conv2d(input_channels, hidden_channels, kernel_size, stride = stride, padding = padding, bias = True) for _ in range(4)])
        self.conv_hidden = nn.ModuleList([nn.Conv2d(hidden_channels, hidden_channels, kernel_size_hidden, stride = 1, padding = padding_hidden, bias = False) for _ in range(4)])
        self.ax = 0.1
        self.ah = 0.1

    def forward(self, input, iter):
        if iter == 1:
            bsize, _, height, width = input.size()
            self.h = Variable(torch.zeros(bsize, self.hidden_channels, int(height / self.stride), int(width / self.stride))).cuda()
        z = torch.sigmoid(self.conv[0](input) + self.conv_hidden[0](self.h))
        r = torch.sigmoid(self.conv[1](input) + self.conv_hidden[1](self.h))
        self.h = (1 - z) * self.h + z * torch.tanh(self.conv[2](input) + self.conv_hidden[2](r * self.h)) + self.ah * self.conv_hidden[3](self.h)
        output = self.h + self.ax * self.conv[3](input)
        return output


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, stride = 1, kernel_size=3, kernel_size_hidden=3):
        super(ConvLSTM, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_features = 4
        self.kernel_size = kernel_size
        self.kernel_size_hidden = kernel_size_hidden
        self.stride = stride

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

    def forward(self, input, iter):
        if iter == 1:
            bsize, _, height, width = input.size()
            self.h = Variable(torch.zeros(bsize, self.hidden_channels, int(height / self.stride), int(width / self.stride))).cuda()
            self.c = Variable(torch.zeros(bsize, self.hidden_channels, int(height / self.stride), int(width / self.stride))).cuda()
        A = self.conv(input) + self.conv_hidden(self.h)
        (ai, af, ao, ag) = torch.split(A, int(A.size()[1] / self.num_features), dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        new_c = f * self.c + i * g
        new_h = o * torch.tanh(new_c)
        self.h = new_h
        self.c = new_c
        return new_h

class Full_encoder(nn.Module):
    def __init__(self, phase):
        super(Full_encoder, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 1)
        self.rnn = eval(phase)
        self.rnn1 = self.rnn(64, 256, stride = 2)
        self.rnn2 = self.rnn(256, 512, stride = 2)
        self.rnn3 = self.rnn(512, 512, stride = 2)

    def forward(self, input, iter):
        output = self.conv(input)
        output = self.rnn1(output, iter)
        output = self.rnn2(output, iter)
        output = self.rnn3(output, iter)
        return output

class Full_decoder(nn.Module):
    def __init__(self, phase):
        super(Full_decoder, self).__init__()
        self.conv1 = nn.Conv2d(32, 512, kernel_size = 1)
        self.rnn = eval(phase)
        self.rnn1 = self.rnn(512, 512)
        self.rnn2 = self.rnn(128, 512)
        self.rnn3 = self.rnn(128, 256)
        self.rnn4 = self.rnn(64, 128)
        self.conv2 = nn.Conv2d(32, 3, kernel_size = 1)

    def forward(self, input, iter):
        output = self.conv1(input)
        output = self.rnn1(output, iter)
        output = F.pixel_shuffle(output, 2)
        output = self.rnn2(output, iter)
        output = F.pixel_shuffle(output, 2)
        output = self.rnn3(output, iter)
        output = F.pixel_shuffle(output, 2)
        output = self.rnn4(output, iter)
        output = F.pixel_shuffle(output, 2)
        output = self.conv2(output)
        return output

class Binarizer(nn.Module):
    def __init__(self):
        super(Binarizer, self).__init__()
        self.conv = nn.Conv2d(512, 32, kernel_size = 1)
    def forward(self, input):
        output = self.conv(input)
        output = torch.tanh(output)
        output = smooth_binary(output)
        return output
    
def sign(input):
    func = Sign()
    return func(input)


class Sign(Function):
    """
    Variable Rate Image Compression with Recurrent Neural Networks
    https://arxiv.org/abs/1511.06085
    """

    def __init__(self):
        super(Sign, self).__init__()

    def forward(self, input):
        prob = input.new(input.size()).uniform_()
        x = input.clone()
        x[(1 - input) / 2 <= prob] = 1
        x[(1 - input) / 2 > prob] = -1
        return x

    def backward(self, grad_output):
        return grad_output, None
    
class New_Binarizer(nn.Module):
    def __init__(self):
        super(New_Binarizer, self).__init__()
        self.conv = nn.Conv2d(512, 32, kernel_size=1, bias=False)

    def forward(self, input):
        feat = self.conv(input)
        x = F.tanh(feat)
      #  return x
        return sign(x)
    

class Clip_loss(nn.Module):
    def __init__(self,low, high, dotloss, R_sample = 0, N_hard = 0):
        super(Clip_loss, self).__init__()
        self.low = low
        self.high = high
        self.dotloss = dotloss
        self.R_sample = R_sample
        self.N_hard = N_hard
        
    def forward(self, x, y):
        mask1 = (x.data>=self.high) & (y.data == self.high)
        mask2 = (x.data<=self.low)  & (y.data == self.low)
        maska = ~ (mask1 + mask2)
        if self.R_sample > 0:
            maskb = torch.rand(maska.size())<self.R_sample
            mask = maska & maskb
        else:
            mask = maska
        diff = self.dotloss(x, y)*Variable(mask.float())
        diff = diff.view([diff.size()[0],-1])
        if self.R_sample>0:
            diff, _ = torch.topk(diff, self.N_hard)
            
        loss = torch.mean(diff)
        return loss
    
def l1_loss(x, y):
    return torch.abs(x-y)
def l2_loss(x, y):
    return (x-y)*(x-y)

    
def smooth_round(x):
    residual = Variable((torch.round(x) -x).data)
    return x + residual
        
def smooth_clip(x, min, max):
    residual = Variable((torch.clamp(x,min,max)-x).data)
    return x + residual

def smooth_binary(x):
    residual = Variable((torch.sign(x) - x).data)
    return x + residual


def crelu(x):
    pos = F.relu(x)
    neg = F.relu(-x)
    return torch.cat([pos,neg],1)

class naive_resunit(nn.Module):
    def __init__(self, n_feat):
        super(naive_resunit, self).__init__()
        self.conv1 = nn.Conv2d(n_feat,n_feat,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(n_feat,n_feat,kernel_size=3,stride=1,padding=1)
    def forward(self,x):
        return x + self.conv1(self.relu(self.conv2(x)))
    
class naive_resunit2(nn.Module):
    def __init__(self, n_feat):
        super(naive_resunit2, self).__init__()
        self.conv1 = nn.Conv2d(n_feat,n_feat,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(n_feat,int(n_feat/2),kernel_size=3,stride=1,padding=1)
    def forward(self,x):
        return x + self.conv1(crelu(self.conv2(x)))

class naive_resunit3(nn.Module):
    def __init__(self, n_feat, scale = 1):
        super(naive_resunit3, self).__init__()
        self.conv1 = nn.Conv2d(n_feat,n_feat,kernel_size=3,stride=1,padding=1)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(n_feat,n_feat,kernel_size=3,stride=1,padding=1)
        self.scale = scale
    def forward(self,x):
        return x + self.conv1(self.relu(self.conv2(x)))*self.scale

class Feat_loss(nn.Module):
    def __init__(self, features, feat_select, weights = 1, layer_loss = torch.nn.L1Loss()):
        super(Feat_loss, self).__init__()
        self.features = features
        self.feat_select = feat_select
        if len(weights)==1:
            self.weights = weights*len(feat_select)
        else:
            self.weights = weights
        if len(layer_loss) == 1:
            self.layer_loss = layer_loss*len(feat_select)
        else:
            self.layer_loss = layer_loss
            
    def forward(self, x, y):
        feat1s = self.features(x, self.feat_select)
        feat2s = self.features(y, self.feat_select)
        ls = []
        totalloss = 0
        for i, (f1, f2) in enumerate(zip(feat1s, feat2s)):
            ls.append(self.layer_loss[i](f1, f2))
            totalloss += ls[-1]*self.weights[i]
        return ls, totalloss
    
def get_loss(name, feature):
    if name == 'l1_hard_im':
        layer_loss1 = Clip_loss(-1, 1, l1_loss, R_sample=0.1, N_hard = 30 )
        layer_select = [1]
        layer_loss = [layer_loss1]
        weights = [1] 
        loss = Feat_loss(feature, layer_select, weights, layer_loss)
        return loss
    elif name == 'l1_hard_feat':
        layer_loss1 = Clip_loss(-1, 1, l1_loss, R_sample=0.1, N_hard = 300 )
        naivel1 = Clip_loss(0, 1000, l1_loss, R_sample=0.1, N_hard = 300 )
        layer_select = [1,1,1]
        layer_loss = [layer_loss1, naivel1, naivel1]
        weights = [1, .2, .1] 
        loss = Feat_loss(feature, layer_select, weights, layer_loss)
        return loss
    elif name == 'l2_im':
        layer_loss1 = Clip_loss(-1, 1, nn.MSELoss())
        layer_select = [1]
        layer_loss = [layer_loss1]
        weights = [1] 
        loss = Feat_loss(feature, layer_select, weights, layer_loss)
        return loss
    elif name == 'l2_im_naive':
        layer_loss1 = nn.MSELoss()
        layer_select = [1]
        layer_loss = [layer_loss1]
        weights = [1] 
        loss = Feat_loss(feature, layer_select, weights, layer_loss)
        return loss
