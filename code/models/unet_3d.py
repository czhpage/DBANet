"""
3D UNet implementation based on EVIL paper's 2D UNet architecture
Adapted from the original EVIL UNet implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
import numbers
import math


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


class ConvBlock3D(nn.Module):
    """3D convolution block with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock3D, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout3d(dropout_p),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock3D(nn.Module):
    """3D Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock3D, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock3D(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock3D(nn.Module):
    """3D Upsampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, bilinear=True):
        super(UpBlock3D, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv3d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder3D(nn.Module):
    def __init__(self, params):
        super(Encoder3D, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        
        self.in_conv = ConvBlock3D(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock3D(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock3D(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock3D(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock3D(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder3D(nn.Module):
    def __init__(self, params):
        super(Decoder3D, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock3D(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock3D(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock3D(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock3D(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv3d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output


class UNet3D(nn.Module):
    """3D UNet implementation for EVIL method"""
    
    def __init__(self, in_chns, class_num, feature_chns=None, dropout=None, bilinear=False):
        super(UNet3D, self).__init__()

        # Default parameters matching EVIL paper
        if feature_chns is None:
            feature_chns = [16, 32, 64, 128, 256]
        if dropout is None:
            dropout = [0.05, 0.1, 0.2, 0.3, 0.5]

        params = {
            'in_chns': in_chns,
            'feature_chns': feature_chns,
            'dropout': dropout,
            'class_num': class_num,
            'bilinear': bilinear,
            'acti_func': 'relu'
        }

        self.encoder = Encoder3D(params)
        self.decoder = Decoder3D(params)

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output


# Feature noise and dropout functions for uncertainty estimation
def Dropout3D(x, p=0.3):
    """3D dropout function"""
    x = torch.nn.functional.dropout3d(x, p)
    return x


def FeatureDropout3D(x):
    """3D feature dropout based on attention"""
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * torch.rand(1).item() * 0.2 + 0.7  # Random threshold between 0.7-0.9
    threshold = threshold.view(x.size(0), 1, 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class FeatureNoise3D(nn.Module):
    """3D feature noise for uncertainty estimation"""
    
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise3D, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x


# Specialized UNet variants for different SSL methods
class UNet3D_CCT(nn.Module):
    """3D UNet with Cross-Consistency Training support"""
    
    def __init__(self, in_chns, class_num, feature_chns=None, dropout=None):
        super(UNet3D_CCT, self).__init__()

        if feature_chns is None:
            feature_chns = [16, 32, 64, 128, 256]
        if dropout is None:
            dropout = [0.05, 0.1, 0.2, 0.3, 0.5]

        params = {
            'in_chns': in_chns,
            'feature_chns': feature_chns,
            'dropout': dropout,
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu'
        }
        
        self.encoder = Encoder3D(params)
        self.main_decoder = Decoder3D(params)
        self.aux_decoder1 = Decoder3D(params)
        self.aux_decoder2 = Decoder3D(params)
        self.aux_decoder3 = Decoder3D(params)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [FeatureNoise3D()(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        aux2_feature = [Dropout3D(i) for i in feature]
        aux_seg2 = self.aux_decoder2(aux2_feature)
        aux3_feature = [FeatureDropout3D(i) for i in feature]
        aux_seg3 = self.aux_decoder3(aux3_feature)
        return main_seg, aux_seg1, aux_seg2, aux_seg3


class UNet3D_DeepSupervision(nn.Module):
    """3D UNet with deep supervision"""
    
    def __init__(self, in_chns, class_num, feature_chns=None, dropout=None):
        super(UNet3D_DeepSupervision, self).__init__()

        if feature_chns is None:
            feature_chns = [16, 32, 64, 128, 256]
        if dropout is None:
            dropout = [0.05, 0.1, 0.2, 0.3, 0.5]

        params = {
            'in_chns': in_chns,
            'feature_chns': feature_chns,
            'dropout': dropout,
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu'
        }
        
        self.encoder = Encoder3D(params)
        self.decoder = Decoder3D_DS(params)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg = self.decoder(feature, shape)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class Decoder3D_DS(nn.Module):
    """3D Decoder with deep supervision"""
    
    def __init__(self, params):
        super(Decoder3D_DS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock3D(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock3D(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock3D(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock3D(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv3d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv3d(self.ft_chns[4], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv3d(self.ft_chns[3], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv3d(self.ft_chns[2], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv3d(self.ft_chns[1], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        
        x = self.up1(x4, x3)
        dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape, mode='trilinear', align_corners=True)

        x = self.up2(x, x2)
        dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape, mode='trilinear', align_corners=True)

        x = self.up3(x, x1)
        dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape, mode='trilinear', align_corners=True)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg
