import torch
import torch.nn as nn
import torchvision.models as models
from ResNet import ResNet50
from torch.nn import functional as F

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class TransBasicBlock(nn.Module):    #用来改变输入通道数，以及判断是否使用残差网络   PTM中的TransB
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)     #44
        self.bn1 = nn.BatchNorm2d(inplanes)          #44
        self.relu = nn.ReLU(inplace=True)            #44
        if upsample is not None and stride != 1:     #buzhixing
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)   #zhixing  44
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x) #44
        out = self.bn1(out) #44
        out = self.relu(out)

        out = self.conv2(out) #44
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out
class ChannelAttention(nn.Module):     #DEM中的通道注意力机制，步骤主要为 1 全局平均池化 2 转变通道数，进行特征提取
    def __init__(self, in_planes, ratio):
        super(ChannelAttention, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)  #全局最大平均池化，将H*W转化为1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = max_out + avg_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):    #空间注意力机制，先将通道数转化为1
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)#寻求每一行的最大值，将通道数转化为1
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicConv2d(nn.Module):    #很多模块的使用卷积层都是以其为基础，论文中的BConvN
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

#Global Contextual module
class GCM(nn.Module):  # 输入通道首先经过四个卷积层的特征提取，并采用torch.cat()进行连接，最后和输入通道的残差进行相加
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 9), padding=(0, 4)),
            BasicConv2d(out_channel, out_channel, kernel_size=(9, 1), padding=(4, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=9, dilation=9)
        )
        self.conv_cat = BasicConv2d(6*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=True)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3, x4, x5), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

#aggregation of the high-level(teacher) features
class aggregation_init(nn.Module):           #F_CD1

    def __init__(self, channel):
        super(aggregation_init, self).__init__()
        self.relu = nn.ReLU(True)
        # first_down = F.interpolate(first_mask, second_mask.size()[2:], mode='bilinear')
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  #上采样为原来的两倍
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)#5个BConv3
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(channel*2, channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample7 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample8 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample9 = BasicConv2d(channel*2, channel, 3, padding=1)
        self.conv_upsample10 = BasicConv2d(channel*2, channel*2, 3, padding=1)
        self.conv_upsample11 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample12 = BasicConv2d(channel*2, channel, 3, padding=1)
        self.conv_upsample13 = BasicConv2d(channel*3, channel*3, 3, padding=1)
        self.conv_upsample14 = BasicConv2d(channel * 2, channel, 3, padding=1)
        self.conv5 = nn.Conv2d(channel*1, 1, 1)
        self.conv6 = nn.Conv2d(channel*1, 1, 1)
        self.conv7 = nn.Conv2d(channel*2, 1, 1)
        self.conv8 = nn.Conv2d(channel*3, 1, 1)
        self.conv9 = nn.Conv2d(channel, 1, 1)
        # self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1)
        # self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        # self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        #
        # self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        # self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)


    def forward(self, x1, x2, x3, x4 ,x5):  #x1，x2，x3三个GCM的输入
        # x1_1 = x1
        # if x3.size()[2:] != x2.size()[2:]:
        #     x3_1 = F.interpolate(x3,x2.size()[2:],mode='bilinear')
        # if x3.size()[2:] != x1.size()[2:]:
        #     x3_2 = F.interpolate(x3, x1.size()[2:], mode='bilinear')
        # if x2.size()[2:] != x1.size()[2:]:
        #     x2_1 = F.interpolate(x2, x1.size()[2:], mode='bilinear')
        x1_1 = self.conv_upsample1(self.upsample(x1))    #22*22*32
        x2_1 = self.conv_upsample2(x1_1*x2) #22*22*32
        x3_1 = self.conv_upsample3(self.upsample(x2_1))  # 44*44*32
        x3_2 = self.conv_upsample4(x3 * x3_1)            #44*44*32
        x3_3 = self.conv_upsample5(torch.cat((x3,x3_2),dim=1))   #44*44*32
        x3_4 = self.conv_upsample6(x3_3*x3_1) #44*44*32
        x4_1 = self.conv_upsample7(self.upsample(x3_4))   #88*88*32
        x4_2 = self.conv_upsample8(x4 * x4_1)               #88*88*32
        x4_3 = self.conv_upsample9(torch.cat((x4,x4_2),dim=1)) #88*88*32
        x4_4 = self.conv_upsample10(torch.cat((x4_3,x4_1),dim=1)) #88*88*64
        x5_1 = self.conv_upsample11(x5 * self.conv_upsample14(x4_4) )   #88*88*32
        x5_2 = self.conv_upsample12(torch.cat((x5,x5_1),dim=1))         #88*88*32
        x5_3 = self.conv_upsample13(torch.cat((x5_2,x4_4),dim=1))       #88*88*96
        loss0 = self.conv9(x1)
        loss1 = self.conv5(x2_1)
        loss2 = self.conv6(x3_4)
        loss3 = self.conv7(x4_4)
        loss4 = self.conv8(x5_3)
        # x_global = self.conv_upsample1(x3_2)*self.conv_upsample2(x2_1)*x1
        # x_middle = self.conv_upsample3(x3_1)*x2
        # if x_middle.size()[2:] != x1.size()[2:]:
        #     x_middle1 = F.interpolate(x_middle,x1.size()[2:],mode='bilinear')
        # x_middle2 = self.conv_upsample4(x_middle1)*x1
        # x_final = self.conv_upsample5(torch.cat((x_global,x_middle2),dim=1))
        # x= self.upsample(self.upsample(x_final))
        # x = self.conv5(x)
        return loss0,loss1,loss2,loss3,x5_3,loss4
        # x2_1 = self.conv_upsample1(self.upsample(x1)) * x2     #第一个点乘
        # print(x2_1.size())
        # x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
        #        * self.conv_upsample3(self.upsample(x2)) * x3   #第二个点乘
        #
        # x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        # x2_2 = self.conv_concat2(x2_2)                                      #第一个c，然后改变特征尺寸
        #
        # x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        # x3_2 = self.conv_concat3(x3_2)                                      #第二个c 通道数为3*channel
        #
        # x = self.conv4(x3_2)
        # x = self.conv5(x)

        # return x

# #aggregation of the low-level(student) features
# class aggregation_final(nn.Module):      #F_CD2
#
#     def __init__(self, channel):
#         super(aggregation_final, self).__init__()
#         self.relu = nn.ReLU(True)
#
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample4 = BasicConv2d(channel*2, channel*2, 3, padding=1)
#         self.conv_upsample5 = BasicConv2d(2*channel, 3*channel, 3, padding=1)
#
#     def forward(self, x1, x2, x3):
#         print(x1.size())
#         print(x2.size())
#         print(x3.size())
#         x1_1 = x1
#         if x3.size()[2:] != x1.size()[2:]:
#             x3_1 = F.interpolate(x3,x1.size()[2:],mode='bilinear')
#         if x2.size()[2:] != x1.size()[2:]:
#             x2_1 = F.interpolate(x2,x1.size()[2:],mode='bilinear')
#         x_global = self.conv_upsample1(x3_1)*self.conv_upsample2(x2_1)*x1
#         x_middle = torch.mul(x3,x2)
#         if x_middle.size()[2:] != x1.size()[2:]:
#             x_middle1 = F.interpolate(x_middle,x1.size()[2:],mode='bilinear')
#         x_middle2 = self.conv_upsample3(x_middle1)* x1
#         x_final = self.conv_upsample4(torch.cat((x_global,x_middle2),dim=1))
#         x_final=self.conv_upsample5(x_final)
#         x3_2 =self.upsample(x_final)
#         return x3_2

# #Refinement flow
# class Refine(nn.Module):
#     def __init__(self):
#         super(Refine,self).__init__()
#         self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv1 = BasicConv2d(128, 64,3,padding=1)
#         self.conv2 = BasicConv2d(512, 256, 3, padding=1)
#         self.conv3 = BasicConv2d(1024, 512, 3, padding=1)
#
#     def forward(self, attention,x1,x2,x3):
#         #Note that there is an error in the manuscript. In the paper, the refinement strategy is depicted as ""f'=f*S1"", it should be ""f'=f+f*S1"".
#         x1_1 = torch.mul(x1, self.upsample2(attention))
#         x1 = self.conv1(torch.cat((x1,x1_1),dim=1))
#         x2_1 = torch.mul(x2,self.upsample2(attention))
#         x2 = self.conv2(torch.cat((x2, x2_1), dim=1))
#         x3_1 = torch.mul(x3,attention)
#         x3 = self.conv3(torch.cat((x3,x3_1),dim=1))
#         return x1,x2,x3
#
#BBSNet
class BBSNet(nn.Module):    #网络框架
    def __init__(self, channel=32):
        super(BBSNet, self).__init__()
        
        #Backbone model
        self.resnet = ResNet50('rgb')
        self.resnet_depth=ResNet50('rgbd')

        #Decoder 1  高层次GCM
        self.rfb2_1 = GCM(512, channel)
        self.rfb3_1 = GCM(1024, channel)
        self.rfb4_1 = GCM(2048, channel)
        #Decoder 2  低层次GCM
        self.rfb0_2 = GCM(64, channel)
        self.rfb1_2 = GCM(256, channel)
        self.agg1 = aggregation_init(channel)
        # self.agg2 = aggregation_final(channel)

        #upsample function
        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv1 = BasicConv2d(32, 96, 3, padding=1)
        # self.conv2 = BasicConv2d(32, 1, 3, padding=1)
        # self.conv3 = BasicConv2d(32, 1, 3, padding=1)
        # self.conv4 = BasicConv2d(32, 1, 3, padding=1)
        #Refinement flow
        # self.HA = Refine()

        #Components of DEM module
        self.atten_depth_channel_0=ChannelAttention(64,16)
        self.atten_depth_channel_1=ChannelAttention(256,64)
        self.atten_depth_channel_2=ChannelAttention(512,128)
        self.atten_depth_channel_3_1=ChannelAttention(1024,256)
        self.atten_depth_channel_4_1=ChannelAttention(2048,512)

        self.atten_depth_spatial_0=SpatialAttention()
        self.atten_depth_spatial_1=SpatialAttention()
        self.atten_depth_spatial_2=SpatialAttention()
        self.atten_depth_spatial_3_1=SpatialAttention()
        self.atten_depth_spatial_4_1=SpatialAttention()

        #Components of PTM module   PTM组合
        self.inplanes =32*2
        self.deconv1 = self._make_transpose(TransBasicBlock, 32*2, 3, stride=2)
        self.inplanes =32
        self.deconv2 = self._make_transpose(TransBasicBlock, 32, 3, stride=2)
        self.agant1 = self._make_agant_layer(32*3, 32*2)
        self.agant2 = self._make_agant_layer(32*2, 32)
        self.out0_conv = nn.Conv2d(32*3, 1, kernel_size=1, stride=1, bias=True)
        self.out1_conv = nn.Conv2d(32*2, 1, kernel_size=1, stride=1, bias=True)
        self.out2_conv = nn.Conv2d(32*1, 1, kernel_size=1, stride=1, bias=True)

        self.sigmoid = nn.Sigmoid()
        if self.training:
            self.initialize_weights()

    def forward(self, x, x_depth):
        x = self.resnet.conv1(x)     #骨干网络提取第一层特征
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_depth = self.resnet_depth.conv1(x_depth)
        x_depth = self.resnet_depth.bn1(x_depth)
        x_depth = self.resnet_depth.relu(x_depth)
        x_depth = self.resnet_depth.maxpool(x_depth)
        # x_origan1 = x_depth
        #layer0 merge                                       #第一层DEM
        temp = x_depth.mul(self.atten_depth_channel_0(x_depth)) #通道注意点乘部分
        temp = temp.mul(self.atten_depth_spatial_0(temp)) #空间注意点乘部分

        x=x+temp #RGB和深度信相加
        #layer0 merge end
                                                            #骨干网络提取第二层特征
        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x1_depth=self.resnet_depth.layer1(x_depth)

        #layer1 merge                                             #第二层DEM
        temp = x1_depth.mul(self.atten_depth_channel_1(x1_depth))
        temp = temp.mul(self.atten_depth_spatial_1(temp))
        x1=x1+temp
        #layer1 merge end

        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32         #骨干网络提取第三层特征
        x2_depth=self.resnet_depth.layer2(x1_depth)

        #layer2 merge                                        #第三层DEM
        temp = x2_depth.mul(self.atten_depth_channel_2(x2_depth))
        temp = temp.mul(self.atten_depth_spatial_2(temp))
        x2 = x2+temp
        #layer2 merge end

        x2_1 = x2
                                                             # 骨干网络提取第四层特征
        x3_1 = self.resnet.layer3_1(x2_1)  # 1024 x 16 x 16
        x3_1_depth=self.resnet_depth.layer3_1(x2_depth)

        #layer3_1 merge                                       #第四层DEM
        temp = x3_1_depth.mul(self.atten_depth_channel_3_1(x3_1_depth))
        temp = temp.mul(self.atten_depth_spatial_3_1(temp))
        x3_1= x3_1+temp
        #layer3_1 merge end

        x4_1 = self.resnet.layer4_1(x3_1)  # 2048 x 8 x 8
        x4_1_depth=self.resnet_depth.layer4_1(x3_1_depth)

        #layer4_1 merge                                          #骨干网络提取第五层特征
        temp = x4_1_depth.mul(self.atten_depth_channel_4_1(x4_1_depth))
        temp = temp.mul(self.atten_depth_spatial_4_1(temp))
        x4_1= x4_1+temp                                      #第五层DEM
        #layer4_1 merge end
        
        #produce initial saliency map by decoder1
        x4_1 = self.rfb4_1(x4_1)
        x3_1 = self.rfb3_1(x3_1)
        x2_1 = self.rfb2_1(x2_1)                                 #高层次网络经历CGM
        x1_1 = self.rfb1_2(x1)                                    #低层次网络经历CGM
        x1_2 = self.rfb0_2(x)
        loss0,loss1,loss2,loss3,x5_3,loss4 = self.agg1(x4_1, x3_1, x2_1,x1_1,x1_2)              #F_CD1
        #Refine low-layer features by initial map
        # x,x1,x5 = self.HA(attention_map.sigmoid(), x,x1,x2)

        #produce final saliency map by decoder2

        # y = self.agg2(x5_2, x1_2, x0_2) #*4                      #F_CD2

        #PTM module
        # #PTM
        y =x5_3
        y = self.agant1(y)
        y = self.deconv1(y)
        y = self.agant2(y)
        y = self.deconv2(y)
        y = self.out2_conv(y)
        L1 = self.upsample32(loss0)
        L2 = self.upsample16(loss1)
        L3 = self.upsample8(loss2)
        L4 = self.upsample4(loss3)
        L5 = self.upsample4(loss4)

        return L1,L2,L3,L4,L5,y,self.sigmoid(L1),self.sigmoid(L2),self.sigmoid(L3),self.sigmoid(L4),self.sigmoid(L5),self.sigmoid(y)
    
    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,           #88-2/2 +1  =44
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):   #liangceng
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)
    
    #initialize the weights
    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)

        all_params = {}
        for k, v in self.resnet_depth.state_dict().items():
            if k=='conv1.weight':
                all_params[k]=torch.nn.init.normal_(v, mean=0, std=1)
            elif k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet_depth.state_dict().keys())
        self.resnet_depth.load_state_dict(all_params)

