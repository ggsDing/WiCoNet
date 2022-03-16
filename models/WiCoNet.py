import math
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from utils.misc import initialize_weights

args = {'n': 4,
        'L': 4,
        'D': 512,
        'mlp_dim': 1024,
        'input_size': 1024,
        'dropout_rate': 0.}

def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class FCN_res50(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(FCN_res50, self).__init__()
        resnet = models.resnet50(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if in_channels > 3:
            newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels - 3, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)


class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(args['D'], args['mlp_dim'])
        self.fc2 = nn.Linear(args['mlp_dim'], args['D'])
        self.act_fn = torch.nn.functional.gelu  # torch.nn.functional.relu
        self.dropout = nn.Dropout(args['dropout_rate'])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.attention_norm = nn.LayerNorm(args['D'], eps=1e-6)
        self.ffn_norm = nn.LayerNorm(args['D'], eps=1e-6)
        self.ffn = Mlp()
        self.norm = nn.LayerNorm(args['D'], eps=1e-6)
        self.attn = Attention()

    def forward(self, x1, x2):
        identity = x1
        x1 = self.attention_norm(x1)
        x2 = self.norm(x2)
        x1 = self.attn(x1, x2)
        x1 = x1 + identity

        identity = x1
        x1 = self.ffn_norm(x1)
        x1 = self.ffn(x1)
        x1 = x1 + identity

        return x1


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(args['D'], eps=1e-6)
        for _ in range(args['L']):
            layer = Block()
            self.layer.append(layer)

    def forward(self, x1, x2):
        for layer_block in self.layer:
            x1 = layer_block(x1, x2)
        encoded = self.encoder_norm(x1)
        return encoded


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.num_attention_heads = args['n']
        self.attention_head_size = int(args['D'] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args['D'], self.all_head_size)
        self.key = nn.Linear(args['D'], self.all_head_size)
        self.value = nn.Linear(args['D'], self.all_head_size)

        self.out = nn.Linear(args['D'], args['D'])
        self.attn_dropout = nn.Dropout(args['dropout_rate'])
        self.proj_dropout = nn.Dropout(args['dropout_rate'])

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x1, x2):
        mixed_query_layer = self.query(x1)
        mixed_key_layer = self.key(x2)
        mixed_value_layer = self.value(x2)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output

#Patch Embedding
class Embeddings(nn.Module):
    def __init__(self, in_channels=3, hidden_size=args['D'], img_size=args['input_size']):
        super(Embeddings, self).__init__()
        n_patches = img_size * img_size
        k_size = 1
        self.patch_embeddings = nn.Conv2d(in_channels, out_channels=hidden_size, kernel_size=k_size, stride=k_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        self.dropout = nn.Dropout(args['dropout_rate'])

    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2).transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(_EncoderBlock, self).__init__()
        self.downsample = downsample
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layers = [
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        if self.downsample:
            x = self.maxpool(x)
        x = self.encode(x)
        return x

class Transformer(nn.Module):
    def __init__(self, in_channels1, in_channels2, feat_size1, feat_size2, hidden_size=args['D']):
        super(Transformer, self).__init__()
        self.embed1 = Embeddings(in_channels1, hidden_size, feat_size1)
        self.embed2 = Embeddings(in_channels2, hidden_size, feat_size2)
        self.encoder = Encoder()

    def forward(self, x_l, x_g):
        embed1 = self.embed1(x_l)
        embed2 = self.embed2(x_g)  # (B, n_patch, hidden//2)
        encoded = self.encoder(embed1, embed2)
        B, n_patch, hidden = encoded.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        encoded = encoded.permute(0, 2, 1)
        encoded = encoded.contiguous().view(B, hidden, h, w)
        return encoded


class ContextEncoder(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(ContextEncoder, self).__init__()

        self.Enc0 = _EncoderBlock(in_channels, 64, downsample=False)
        self.Enc1 = _EncoderBlock(64, 128)
        self.Enc2 = _EncoderBlock(128, 256)
        self.Enc3 = _EncoderBlock(256, 512)

    def forward(self, x):
        x_size = x.size()

        enc0 = self.Enc0(x)
        enc1 = self.Enc1(enc0)
        enc2 = self.Enc2(enc1)
        enc3 = self.Enc3(enc2)

        return enc3

class WiCoNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, size_context=1024, size_local=512, scale=4):
        super(WiCoNet, self).__init__()
        feat_size1 = size_local // 8
        feat_size2 = size_context // scale // 8
        self.context_branch = ContextEncoder(in_channels, num_classes)
        self.local_branch = FCN_res50(in_channels, num_classes, pretrained=True)
        self.head = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(512, momentum=0.95), nn.ReLU())
        self.transformer = Transformer(512, 512, feat_size1, feat_size2)

        self.classifier_aux = nn.Sequential(conv1x1(512, 64), nn.BatchNorm2d(64), nn.ReLU(), conv1x1(64, num_classes))
        self.classifier = nn.Sequential(conv1x1(512, 64), nn.BatchNorm2d(64), nn.ReLU(), conv1x1(64, num_classes))
        initialize_weights(self.classifier, self.classifier_aux, self.head)

    # Two inputs are needed, x_s is the scaled input from the context window.
    def forward(self, x_s, x):
        x_size = x.size()
        xs_size = x_s.size()
        enc3 = self.context_branch(x_s)  # 1/16
        aux = self.classifier_aux(enc3)

        x = self.local_branch.layer0(x)  # 1/2, 64
        x = self.local_branch.maxpool(x)  # 1/4, 64
        x = self.local_branch.layer1(x)  # 1/4, 256
        x = self.local_branch.layer2(x)  # 1/8, 512
        x = self.local_branch.layer3(x)  # 1/8, 1024
        x = self.local_branch.layer4(x)  # 1/8, 2048
        x = self.head(x)  # 1/8, 512

        x = self.transformer(x, enc3)
        out = self.classifier(x)
        return F.upsample(out, x_size[2:], mode='bilinear'), F.upsample(aux, xs_size[2:], mode='bilinear')