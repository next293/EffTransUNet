# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
import timm

# =====================================
#           Attention Modules
# =====================================

class CoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0,1,3,2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0,1,3,2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        out = identity * a_h * a_w
        return out


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction, in_channels, 1, bias=False)
        )

        self.compress = lambda x: torch.cat(
            (torch.mean(x, dim=1, keepdim=True),
             torch.max(x, dim=1, keepdim=True)[0]), dim=1)
        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                 padding=kernel_size//2, bias=False)

    def forward(self, x):
        # Channel attention
        avg = self.mlp(self.avg_pool(x))
        max = self.mlp(self.max_pool(x))
        ca = torch.sigmoid(avg + max)
        x = x * ca
        # Spatial attention
        sa = torch.sigmoid(self.spatial(self.compress(x)))
        out = x * sa
        return out

# =====================================
#          Backbone: EfficientNet
# =====================================

class EfficientNetBackbone(nn.Module):
    def __init__(self, backbone_name='efficientnet_b3'):
        super(EfficientNetBackbone, self).__init__()
        # backbone_name  'efficientnet_b3' / 'efficientnet_b4'  'efficientnet_b5'
        self.backbone_name = backbone_name
        print(backbone_name)
        self.net = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=(1,2,3,4),
        )

        self.feature_channels = self.net.feature_info.channels()

    def forward(self, x):
        features = self.net(x)
        return features


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        heads = config.transformer.num_heads
        self.head_size = config.hidden_size // heads
        self.all_head_size = heads * self.head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key   = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out    = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer.attention_dropout_rate)
        self.proj_dropout = Dropout(config.transformer.attention_dropout_rate)
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + ( -1, self.head_size )
        x = x.view(*new_shape)
        return x.permute(0,2,1,3)

    def forward(self, hidden_states):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        scores = torch.matmul(q, k.transpose(-1,-2))
        scores = scores / math.sqrt(self.head_size)
        probs = self.softmax(scores)
        weights = probs if self.vis else None
        probs = self.attn_dropout(probs)

        context = torch.matmul(probs, v)
        context = context.permute(0,2,1,3).contiguous()
        new_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_shape)
        out, _ = self.out(context), weights
        out = self.proj_dropout(out)
        return out, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer.mlp_dim)
        self.fc2 = Linear(config.transformer.mlp_dim, config.hidden_size)
        self.act = nn.GELU()
        self.drop= Dropout(config.transformer.dropout_rate)
        # init
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class Embeddings(nn.Module):
    def __init__(self, config, img_size,backbone=None):
        super(Embeddings, self).__init__()
        self.config = config
        self.backbone = backbone or EfficientNetBackbone()

        feat_size = img_size // 32
        num_patches = feat_size * feat_size
        last_ch = self.backbone.feature_channels[-1]
        self.patch_embeddings = Conv2d(
            in_channels=last_ch,
            out_channels=config.hidden_size,
            kernel_size=1,
            bias=False
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches, config.hidden_size)
        )
        self.dropout = Dropout(config.transformer.dropout_rate)



    def forward(self, x):
        features = self.backbone(x)
        last = features[-1]
        x = self.patch_embeddings(last)
        B, C, H, W = x.size()
        x = x.flatten(2).transpose(1,2)
        x = x + self.position_embeddings
        x = self.dropout(x)

        return x, features

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.norm1 = LayerNorm(config.hidden_size, eps=1e-6)
        self.attn  = Attention(config, vis)
        self.norm2 = LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp   = Mlp(config)

    def forward(self, x):
        h = x
        x, _ = self.attn(self.norm1(x))
        x = x + h
        h = x
        x = self.mlp(self.norm2(x))
        x = x + h
        return x, None


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layers = nn.ModuleList([
            copy.deepcopy(Block(config, vis))
            for _ in range(config.transformer.num_layers)
        ])
        self.norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        x = self.norm(x)
        return x, None

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis=False, backbone=None):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size, backbone)
        self.encoder    = Encoder(config, vis)

    def forward(self, x):
        x, features = self.embeddings(x)
        x, _ = self.encoder(x)
        return x, features

# =====================================
#          Decoder
# =====================================

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, padding, stride=1, use_bn=True):
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=not use_bn)]
        if use_bn: layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        super(Conv2dReLU, self).__init__(*layers)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super(DecoderBlock, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, 3, padding=1)
        self.conv2 = Conv2dReLU(out_channels, out_channels, 3, padding=1)
        self.cbam  = CBAM(out_channels)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cbam(x)
        return x

class DecoderCup(nn.Module):
    def __init__(self, config, img_size):
        super(DecoderCup, self).__init__()
        self.config = config
        self.coordatt = CoordAttention(config.hidden_size)
        head_channels = 512
        self.conv_more = Conv2dReLU(config.hidden_size, head_channels, 3, padding=1)
        dec_chs = list(config.decoder_channels)
        in_chs = [head_channels] + dec_chs[:-1]
        skip_chs = list(config.skip_channels)
        for i in range(4 - config.n_skip):
            skip_chs[-1-i] = 0

        self.blocks = nn.ModuleList([
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_chs, dec_chs, skip_chs)
        ])

    def forward(self, x, features):
        B, N, hidden = x.size()
        h = w = int(math.sqrt(N))
        x = x.permute(0,2,1).contiguous().view(B, hidden, h, w)

        # Coord-Attention
        x = self.coordatt(x)
        x = self.conv_more(x)
        skips = features[:-1]
        skips = skips[::-1]
        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < self.config.n_skip else None
            x = block(x, skip)
        return x

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=2):
        conv = nn.Conv2d(in_channels, out_channels,
                         kernel_size=kernel_size,
                         padding=kernel_size//2)
        up = nn.UpsamplingBilinear2d(scale_factor=upsampling)
        super(SegmentationHead, self).__init__(conv, up)

# =====================================
#      EffTransUNet
# =====================================
class EffTransUNet(nn.Module):
    def __init__(self, config_name='EfficientNet-B3', backbone_name='efficientnet_b3',img_size=224, num_classes=2, vis=False):
        super(EffTransUNet, self).__init__()

        config = CONFIGS[config_name]

        print(backbone_name)
        fb = EfficientNetBackbone(backbone_name)
        ch = fb.feature_channels
        config.skip_channels = [ch[2], ch[1], ch[0], 0]

        self.transformer = Transformer(config, img_size, vis, backbone=fb)
        self.decoder     = DecoderCup(config, img_size)
        self.seg_head    = SegmentationHead(
            config.decoder_channels[-1],
            config.n_classes,
            kernel_size=3,
            upsampling=2
        )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, features = self.transformer(x)
        x = self.decoder(x, features)
        x = self.seg_head(x)
        return x


CONFIGS = {
    'EfficientNet-B3': configs.get_b16_config(),
    'EfficientNet-B4': configs.get_b16_config(),
    'EfficientNet-B5': configs.get_b16_config(),
}