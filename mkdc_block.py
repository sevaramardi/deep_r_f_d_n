import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels,
               out_channels,
               kernel_size=3,
               bias=True):
    
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)

# Main block multikernalDepthwiseConv()
class MultiKernalDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernels, stride=1, dilation=1, groups=1):
        super(MultiKernalDepthwiseConv, self).__init__() 

        padding_dict = {1: 0, 3: 1, 5: 2, 7: 3}
        self.seps = nn.ModuleList()
        for kernel in kernels:
            sep = nn.Conv2d(in_channels, in_channels, kernel_size=kernel, stride=1, padding=padding_dict[kernel],
                            dilation=1,
                            groups=in_channels, bias=False)
            self.seps.append(sep)
        self.bn1 = nn.BatchNorm2d(in_channels * len(kernels))
        self.pointwise = nn.Conv2d(in_channels * len(kernels), out_channels, 1, stride, 0, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.SiLU(inplace=True)

    def forward(self, input):
        seps = []
        for sep in self.seps:
            seps.append(sep(input))
        out_seq = torch.cat(seps, dim=1)
        out = self.pointwise(out_seq)


        out = self.relu(out)



        return out


def conv_layer_sep(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
   
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))


    depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size,stride, padding=padding, bias=True, dilation=dilation,
                     groups=in_channels)




    rel = nn.SiLU(inplace=True)

    point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)


    bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01, affine=True)
    return nn.Sequential(depth_conv, point_conv, bn, rel)



def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - stride) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)

def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer




def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

class ESA(nn.Module):
    def __init__(self,esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m

class DRFDB(nn.Module):
    def __init__(self, in_channels, mid_channel, out_channels=None, esa_channels=16):
        super(DRFDB, self).__init__()
        dis_channel = [36, 24, 12]

        if(mid_channel==None):
            mid_channel =  in_channels
        if(out_channels==None):
            out_channels = in_channels

        self.c1_r = MultiKernalDepthwiseConv(in_channels, mid_channel //2, [5])
        self.c2_r = MultiKernalDepthwiseConv(mid_channel //2, mid_channel //2, [1,3])
        self.c3_r = MultiKernalDepthwiseConv(mid_channel // 2, mid_channel, [1,3])
        self.c4_r = MultiKernalDepthwiseConv(mid_channel, in_channels, [5])
        self.c5 = conv_layer(mid_channel, out_channels, 1)
        self.esa = ESA(esa_channels, out_channels, nn.Conv2d)

        self.act = activation('lrelu', neg_slope=0.05)




