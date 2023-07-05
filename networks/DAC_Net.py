import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.segformer import *
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from torchvision.ops import DeformConv2d
# from torchvision.ops import deform_conv2d
from networks.segformer import *
from networks.softpool import SoftPool2D

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Dualtrans(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, dim, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        nfx = self.norm(x)
        return nfx, H, W
class EfficientAttention(nn.Module):
    """
    input  -> x:[B, D, H, W]
    output ->   [B, D, H, W]

    in_channels:    int -> Embedding Dimension
    key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
    value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2)
    head_count:     int -> It divides the embedding dimension by the head_count and process each part individually

    Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
    """

    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()

        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)

            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)

            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]

            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)  # n*dv

            # context =key.transpose(1, 2) @ value   # dk*dv
            # attended_value = (query @ context).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention


class ChannelAttention(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class DualTransformerBlock(nn.Module):
    """
    Input  -> x (Size: (b, (H*W), d)), H, W
    Output -> (b, (H*W), d)
    """

    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientAttention(in_channels=in_dim, key_channels=key_dim, value_channels=value_dim, head_count=1)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.channel_attn = ChannelAttention(in_dim)
        self.norm4 = nn.LayerNorm(in_dim)
        if token_mlp == "mix":
            self.mlp1 = MixFFN(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN(in_dim, int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp1 = MixFFN_skip(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN_skip(in_dim, int(in_dim * 4))
        else:
            self.mlp1 = MLP_FFN(in_dim, int(in_dim * 4))
            self.mlp2 = MLP_FFN(in_dim, int(in_dim * 4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        # # dual attention structure, efficient attention first then transpose attention
        # norm1 = self.norm1(x)
        # norm1 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(norm1)
        #
        # attn = self.attn(norm1)
        # attn = Rearrange("b d h w -> b (h w) d")(attn)
        #
        # add1 = x + attn
        # norm2 = self.norm2(add1)
        # mlp1 = self.mlp1(norm2, H, W)
        #
        # add2 = add1 + mlp1
        # norm3 = self.norm3(add2)
        # channel_attn = self.channel_attn(norm3)
        #
        # add3 = add2 + channel_attn
        # norm4 = self.norm4(add3)
        # mlp2 = self.mlp2(norm4, H, W)

        norm1 = self.norm1(x)
        channel_attn = self.channel_attn(norm1)

        add1=x+channel_attn
        norm2 = self.norm2(add1)
        mlp1 = self.mlp1(norm2, H, W)

        add2=add1+mlp1
        norm3 = self.norm3(add2)
        norm3 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(norm3)
        attn = self.attn(norm3)
        attn = Rearrange("b d h w -> b (h w) d")(attn)

        add3 = add2 + attn
        norm4 = self.norm4(add3)
        mlp2 = self.mlp2(norm4, H, W)
        # channel_attn = self.channel_attn(norm1)
        # channel_attn = Rearrange("b d h w -> b (h w) d")(channel_attn)
        #
        # add1 = x + channel_attn
        # norm2 = self.norm2(add1)
        # mlp1 = self.mlp1(norm2, H, W)
        #
        # add2 = add1 + mlp1
        # norm3 = self.norm3(add2)
        # attn = self.attn(norm3)
        #
        # add3 = add2 + attn
        # norm4 = self.norm4(add3)
        # mlp2 = self.mlp2(norm4, H, W)

        mx = add3 + mlp2
        return mx
class DConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False):
        super(DConv, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)
        self.conv2 = DeformConv2d(inplanes, planes, kernel_size=kernel_size,stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(x, out)
        return out
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DConv(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DConv(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DConv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            SoftPool2D(kernel_size=2,stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = DConv(in_channels, out_channels, kernel_size=1,padding=0)

    def forward(self, x):
        return self.conv(x)
class DACNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.dual1 = DualTransformerBlock(64, 64, 64)
        self.dualtrans1=Dualtrans(64)
        self.down1 = (Down(64, 128))
        self.dual2=DualTransformerBlock(128,128,128)
        self.dualtrans2 = Dualtrans(128)
        self.down2 = (Down(128, 256))
        self.dual3 = DualTransformerBlock(256, 256, 256)
        self.dualtrans3 = Dualtrans(256)
        self.down3 = (Down(256, 512))
        self.dual4 = DualTransformerBlock(512, 512, 512)
        self.dualtrans4 = Dualtrans(512)
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x1,H,W=self.dualtrans1(x1)
        x1=self.dual1(x1,H,W)
        x1 = x1.reshape(x1.size(0), H, W, -1).permute(0, 3, 1, 2).contiguous()
        x2 = self.down1(x1)
        x2,H,W=self.dualtrans2(x2)
        x2=self.dual2(x2,H,W)
        x2 = x2.reshape(x2.size(0), H, W, -1).permute(0, 3, 1, 2).contiguous()
        x3 = self.down2(x2)
        x3,H,W=self.dualtrans3(x3)
        x3=self.dual3(x3,H,W)
        x3 = x3.reshape(x3.size(0), H, W, -1).permute(0, 3, 1, 2).contiguous()
        x4 = self.down3(x3)
        x4, H, W = self.dualtrans4(x4)
        x4 = self.dual4(x4, H, W)
        x4 = x4.reshape(x4.size(0), H, W, -1).permute(0, 3, 1, 2).contiguous()
        x5 = self.down4(x4)
        x5, H, W = self.dualtrans4(x5)
        x5 = self.dual4(x5, H, W)
        x5 = x5.reshape(x5.size(0), H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.up1(x5, x4)
        x, H, W = self.dualtrans3(x)
        x = self.dual3(x, H, W)
        x = x.reshape(x.size(0), H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.up2(x, x3)
        x, H, W = self.dualtrans2(x)
        x = self.dual2(x, H, W)
        x = x.reshape(x.size(0), H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.up3(x, x2)
        x, H, W = self.dualtrans1(x)
        x = self.dual1(x, H, W)
        x = x.reshape(x.size(0), H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.up4(x, x1)
        x, H, W = self.dualtrans1(x)
        x = self.dual1(x, H, W)
        x = x.reshape(x.size(0), H, W, -1).permute(0, 3, 1, 2).contiguous()
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
