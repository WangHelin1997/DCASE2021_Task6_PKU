'''https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py'''

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from augmentation import *
from hparams import hparams as hp
def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class Cnn10(nn.Module):
    def __init__(self):

        super(Cnn10, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)

    def forward(self, input):

        x = input.unsqueeze(1)   # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)     #(batch_size, 512, T/16, mel_bins/16)
 
        return x

def _resnet_conv3x3(in_planes, out_planes):
    #3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)


def _resnet_conv1x1(in_planes, out_planes):
    #1x1 convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class _ResnetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('_ResnetBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in _ResnetBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.stride = stride

        self.conv1 = _resnet_conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _resnet_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            out = F.avg_pool2d(x, kernel_size=(2, 2))
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResnetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.stride = stride
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _resnet_conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _resnet_conv3x3(width, width)
        self.bn2 = norm_layer(width)
        self.conv3 = _resnet_conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        init_layer(self.conv3)
        init_bn(self.bn3)
        nn.init.constant_(self.bn3.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            x = F.avg_pool2d(x, kernel_size=(2, 2))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(_ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[0])
                init_bn(downsample[1])
            elif stride == 2:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2), 
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[1])
                init_bn(downsample[2])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x1 = x
        x = self.layer3(x)
        x2 = x
        x = self.layer4(x)

        return x, x1, x2

class ResNet38(nn.Module):
    def __init__(self):
        super(ResNet38, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[3, 4, 6, 3], zero_init_residual=True)
        self.conv_block_after1 = ConvBlock(in_channels=512, out_channels=2048)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)

    def forward(self, input):
        """
        Input: (batch_size, data_length)"""

        x = input.unsqueeze(1)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x, x1, x2 = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)

        return x, x1, x2

## Wavegram 
class ConvPreWavBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvPreWavBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=3, stride=1,
                              padding=1, bias=False)
                              
        self.conv2 = nn.Conv1d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=3, stride=1, dilation=2, 
                              padding=2, bias=False)
                              
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=pool_size)
        
        return x

class Wavegram_Logmel_Cnn14(nn.Module):
    def __init__(self):
        
        super(Wavegram_Logmel_Cnn14, self).__init__()

        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 128)
        self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)


        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_bn(self.bn0)
        init_layer(self.fc1)
 
    def forward(self, input, wave, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # Wavegram
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(wave[:, None, :])))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)
        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
        a1 = self.pre_block4(a1, pool_size=(2, 1))

        print("the size of a1 is ",a1.shape)
        x = input.unsqueeze(1)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        print("the size of x is", x.shape)
        # Concatenate Wavegram and Log mel spectrogram along the channel dimension
        x = torch.cat((x, a1), dim=1)

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        return x


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V * 2, dim_V * 2)

    def forward(self, Q, K):
        mu0 = F.avg_pool2d(K, (K.shape[1], 1))
        sig0 = F.avg_pool2d(K ** 2, (K.shape[1], 1))
        sig0 = torch.sqrt(torch.clamp(sig0 - mu0 ** 2, 1e-6))
        K, V = K, K
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        A1 = torch.cat(A.split(Q.size(0), 0), 2)
        M = A.bmm(V_)
        div = torch.sqrt(torch.clamp(A.bmm(V_ ** 2) - M ** 2, 1e-6))
        M = torch.cat(M.split(Q.size(0), 0), 2)
        div = torch.cat(div.split(Q.size(0), 0), 2)
        O = torch.cat((M, div), 2)

        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O1 = F.relu(self.fc_o(O))
        O1 = F.dropout(O1, p=0.2, training=self.training)
        O = O + O1
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        O = O.squeeze(dim=1)
        return O, A1


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class Tag(nn.Module):
    def __init__(self, class_num, model_type, pretrain_model_path=None, freeze_cnn=False, GMAP=False, specMix=False):
        super(Tag, self).__init__()
        if model_type == 'Cnn10':
            self.feature = Cnn10()
            channel = 512
        elif model_type == 'resnet38':
            self.feature = ResNet38()
            channel = 2048
        elif model_type == 'wavegram':
            self.feature = Wavegram_Logmel_Cnn14()
            channel = 2048
        else :
            print('Error!!!')
        print("model type is:", model_type)
        if pretrain_model_path:
            pretrain_cnn = torch.load(pretrain_model_path,map_location="cpu")
            dict_trained = pretrain_cnn['model']
            dict_new = self.feature.state_dict().copy()
            new_list = list(self.feature.state_dict().keys())
            trained_list = list(dict_trained.keys())
            for name in new_list:
                if name in trained_list:
                    print(name)
                    dict_new[name] = dict_trained[name]
            self.feature.load_state_dict(dict_new)

        if freeze_cnn:
            self.freeze_cnn()
            print('freeze_cnn')
        if GMAP:
            self.pma = PMA(dim = channel, num_heads = 4, num_seeds = 1, ln=False)
            self.fc_prob_att = nn.Linear(channel * 2, channel * 2)
            self.fc1 = nn.Linear(channel*2,channel,bias=True)

            self.pma_f1 = PMA(dim = 128, num_heads = 4, num_seeds = 1, ln=False)
            self.fc_prob_att_f1 = nn.Linear(256, 256)
            self.fc1_f1 = nn.Linear(256,channel,bias=True)

            self.pma_f2 = PMA(dim = 256, num_heads = 4, num_seeds = 1, ln=False)
            self.fc_prob_att_f2 = nn.Linear(512, 512)
            self.fc1_f2 = nn.Linear(512,channel,bias=True)

            self.fc = nn.Linear(channel*3,class_num,bias=True)
        else:
            self.fc1 = nn.Linear(channel, channel, bias=True)
            self.fc1_f1 = nn.Linear(128, channel, bias=True)
            self.fc1_f2 = nn.Linear(256, channel, bias=True)
            self.fc = nn.Linear(channel*3,class_num,bias=True)

        self.GMAP = GMAP
        self.init_weights()
        self.specMix = specMix
        if specMix:
            self.spec_augmenter = SpecMixAugmentation(time_mix_width=64, time_stripes_num=3, freq_mix_width=5, freq_stripes_num=2)

    def init_weights(self):
        init_layer(self.fc1)
        init_layer(self.fc)
        init_layer(self.fc1_f1)
        init_layer(self.fc1_f2)
        if self.GMAP:
            init_layer(self.fc_prob_att_f1)
            init_layer(self.fc_prob_att_f2)

    def freeze_cnn(self):
        for p in self.feature.parameters():
            p.requires_grad = False
            
    def forward(self,input):
        '''
        :param input: (batch_size,time_steps, mel_bins)
        :return: ()
        '''
        if self.specMix and self.training:
            input = self.spec_augmenter(input[:,None,:,:])
            input = input[:,0]

        feature, f1, f2 = self.feature(input)     #(batch_size, 512/2048, T/16, mel_bins/16)
        x = torch.mean(feature,dim=3)     #(batch_size, 512/2048, T/16) resnet 2048
        f1 = torch.mean(f1,dim=3)
        f2 = torch.mean(f2, dim=3)
        out1,out2,out3 = x,f1,f2

        if self.GMAP:
            x = x.transpose(1, 2)
            x, A = self.pma(x)
            x = (F.sigmoid(self.fc_prob_att(x))) * x
            f1 = f1.transpose(1, 2)
            f1, A_f1 = self.pma_f1(f1)
            f1 = (F.sigmoid(self.fc_prob_att_f1(f1))) * f1
            f2 = f2.transpose(1, 2)
            f2, A_f2 = self.pma_f2(f2)
            f2 = (F.sigmoid(self.fc_prob_att_f2(f2))) * f2
        else:
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            (a1, _) = torch.max(f1, dim=2)
            a2 = torch.mean(f1, dim=2)
            f1 = a1 + a2
            (b1, _) = torch.max(f2, dim=2)
            b2 = torch.mean(f2, dim=2)
            f2 = b1 + b2

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu_(self.fc1(x))
        f1 = F.dropout(f1, p=0.2, training=self.training)
        f1 = F.relu_(self.fc1_f1(f1))
        f2 = F.dropout(f2, p=0.2, training=self.training)
        f2 = F.relu_(self.fc1_f2(f2))
        x = torch.cat((x, f1, f2), dim=1)
        output = torch.sigmoid(self.fc(x))
        return output,out1,out2,out3
if __name__ == '__main__':
    encoder = Tag(300,'wavegram',"pretrainmodel/Wavegram_Logmel_Cnn14_mAP=0.439.pth",True)
