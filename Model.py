import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import torch.nn.init as init

# for other networks
def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(inplace=True)
        )

        initialize_weights(self, scale=0.1)

    def forward(self, x):
        return self.conv(x)

def adjust(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    return x1

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, ks, slope=0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ks, stride=1, padding=(ks - 1) // 2)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.act  = nn.LeakyReLU(slope, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ZPool(nn.Module):
    def forward(self, x):
        x_mean = x.mean(dim=1, keepdim=True)
        x_max = x.max(dim=1, keepdim=True)[0]
        return torch.cat([x_mean, x_max], dim=1)

class AttentionGate(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.compress   = ZPool()
        self.conv       = BasicConv2d(2, 1, kernel_size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        y = self.compress(x)
        y = self.conv(y)
        y = self.activation(y)
        return x * y

class TripletAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.ch = AttentionGate(kernel_size)
        self.cw = AttentionGate(kernel_size)
        self.hw = AttentionGate(kernel_size)

    def forward(self, x):
        b, c, h, w = x.shape
        x_ch = self.ch(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # c and h
        x_cw = self.cw(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x_hw = self.hw(x)
        return 1 / 3 * (x_ch + x_cw + x_hw)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[32, 64, 128]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of U-Net
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of U-Net
        for feature in reversed(features):
            self.ups.append(nn.Conv2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        initialize_weights(self, scale=0.1)

    def forward(self, x):
        skip_connections = []

        # Downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upsampling
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Adjust x size to match skip connection
            if x.shape != skip_connection.shape:
                x = adjust(x, skip_connection)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width).to(x)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        return self.gamma*(out_H + out_W) + x


class ARB(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.3, kernel_size=3):
        super(ARB, self).__init__()

        self.bn1   = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2   = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.silu(self.bn1(x))))
        out = self.conv2(F.silu(self.bn2(out)))
        short = self.shortcut(x)
        out += short

        return out

class ADLB(nn.Module):
    def __init__(self, in_channels, out_channels, n_res_blocks = 16):
        super().__init__()
        self.init_res = ARB(in_channels, out_channels)
        self.blocks = nn.ModuleList()
        for _ in range(1, n_res_blocks):
            self.blocks.append(
                ARB(out_channels, out_channels)
            )

    def forward(self,x):
        new_inputs = self.init_res(x)
        for block in self.blocks:
            new_inputs = torch.add(block(new_inputs), x)
        return new_inputs

class UpScaleBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, scaling_factor=2):
        super(UpScaleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.ups  = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.ups(x)
        x = self.silu(x)
        return x

class UpScaleBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, scale_factor=2):
        super(UpScaleBlock, self).__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size // 2)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size // 2)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size // 2)
        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)
        self.act_f = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.up_nn(self.act_f(self.norm1(x)))
        s = self.conv3(x)
        x = self.act_f(self.norm2(self.conv1(x)))
        x = self.conv2(x)
        return x + s

class ChannelGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialGate, self).__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class BAM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel_attn = ChannelGate(channel)
        self.spatial_attn = SpatialGate()

    def forward(self, x):
        attn = x + self.channel_attn(x)
        attn = x + self.spatial_attn(attn)
        return attn

class ADUNet(nn.Module):
    def __init__(self, channels, num_blocks=3):
        super().__init__()
        self.ds = ADLB(channels, channels, n_res_blocks = num_blocks)
        self.un = UNet(channels, channels)

    def forward(self, x, r, scale_factor=1):
        x = self.ds(x)
        x = self.un(x)
        y = F.interpolate(r, scale_factor=scale_factor, mode='bicubic', align_corners=False) if scale_factor!=1 else r
        x = adjust(x, y)
        x = x + y
        return x

class CoCABlock(nn.Module):
    def __init__(self, channels, num_blocks=3):
        super().__init__()
        self.cca = CrissCrossAttention(channels)
        self.ds  = ADUNet(channels, num_blocks) 

    def forward(self, x, r, scale_factor=1):
        x = self.cca(x)
        x = self.ds(x, r, scale_factor)
        return x

class UpAB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ups  = UpScaleBlock(channels)
        self.attn = TripletAttention() 

    def forward(self, x):
        x = self.ups(x)
        x = self.attn(x)
        return x

class PreNet(nn.Module):
    def __init__(self, in_channels, dim, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class RefinementLayer(nn.Module):
    def __init__(self, base_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.Conv2d(base_channels, base_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class CoCAG(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, kernel_size=9, patch_size=16):
        super(CoCAG, self).__init__()

        self.feats = PreNet(in_channels, base_channels, kernel_size) 

        self.patch_embed = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=patch_size, stride=patch_size),
            nn.SiLU(inplace=True)
        )

        self.b1 = CoCABlock(base_channels, 3)
        self.u1 = UpAB(base_channels)

        self.b2 = CoCABlock(base_channels, 3)
        self.u2 = UpAB(base_channels)

        self.b3 = CoCABlock(base_channels, 3)
        self.u3 = UpAB(base_channels)

        self.b4 = CoCABlock(base_channels, 3)
        self.u4 = UpAB(base_channels)

        self.b5 = ADUNet(base_channels, 2)
        self.u5 = UpAB(base_channels)

        self.b6 = ADUNet(base_channels, 2)
        self.u6 = UpAB(base_channels)

        self.final = RefinementLayer(base_channels, out_channels, kernel_size)

        initialize_weights(self, scale=0.1)


    def forward(self, x):
        o = x
        x = self.feats(x)
        r = x
        # ViT-like processing
        x = self.patch_embed(x)

        # working the latent space
        x = self.b1(x, r, 0.0625)
        # upscale by 2: h/16 x w/16 - h/8 x w/8
        x = self.u1(x)

        x = self.b2(x, r, 0.125)
        # upscale by 2: h/8 x w/8 - h/4 x w/4
        x = self.u2(x)

        x = self.b3(x, r, 0.25)
        # upscale by 2: h/4 x w/4 - h/2 x w/2
        x = self.u3(x)

        x = self.b4(x, r, 0.5)
        # upscale by 2: h/2 x w/2 - h/1 x w/1
        x = self.u4(x)
        #x = adjust(x, r)

        x = self.b5(x, r, 1)
        # upscale by 2: h x w - h*2 x w*2
        x = self.u5(x)

        x = self.b6(x, r, 2)
        # upscale by 2: h*2 x w*2 - h*4 x w*4
        x = self.u6(x)

        x = self.final(x) + F.interpolate(o, scale_factor=4, mode='bicubic', align_corners=False)

        return x.clamp(0, 1)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, 1, normalize=False),
            *discriminator_block(64, 64, 2, normalize=True),
            *discriminator_block(64, 128, 1, normalize=True),
            *discriminator_block(128, 128, 2, normalize=True),
            *discriminator_block(128, 256, 1, normalize=True),
            *discriminator_block(256, 256, 2, normalize=True),
            *discriminator_block(256, 512, 1, normalize=True),
            *discriminator_block(512, 512, 2, normalize=True),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(in_features=512*8*8, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)