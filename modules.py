__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

"""
This code is based on @dome272 implementation of DDPM's
https://github.com/dome272/Diffusion-Models-pytorch
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(
            current_model.parameters(), ema_model.parameters()
        ):
            old_weight, up_weight = ema_params.data, current_params.data
            ema_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        if type(model) in (
            nn.parallel.DataParallel,
            nn.parallel.DistributedDataParallel,
        ):  # checks multiprocessing/multi-GPU's
            ema_model.load_state_dict(model.module.state_dict())
        else:
            ema_model.load_state_dict(model.state_dict())


class LRWarmupCosineDecay(LambdaLR):
    """Linear warmup and then cosine decay.
      Linearly increases the factor the learning rate is multiplied with from start_lr
      to target_lr over the specified number of steps
      Decreases this factor from target_lr to start_lr over remaining steps.
      Set lr in optimizer to 1 to ensure that this factor equals the lr.

    Parameters
    ----------
    LambdaLR : _type_
        PyTorch Class

    Returns
    -------
    _type_
        lr
    """
    
    

    def __init__(
        self, optimizer, warmup_steps, steps_total, start_lr, target_lr, last_epoch=-1
    ):
        self.warmup_steps = warmup_steps
        self.steps_total = steps_total
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.increase = (target_lr - start_lr) / warmup_steps
        super(LRWarmupCosineDecay, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return self.start_lr + (step * self.increase)
        return self.start_lr + (self.target_lr - self.start_lr) * (
            (
                1
                + math.cos(
                    math.pi
                    * (step - self.warmup_steps)
                    / float(self.steps_total - self.warmup_steps)
                )
            )
            * 0.5
        )


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(
            channels, 4, batch_first=True
        )  # second argument is number of head -> increase
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2).contiguous()
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return (
            attention_value.swapaxes(2, 1)
            .view(-1, self.channels, self.size, self.size)
            .contiguous()
        )


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            #nn.GELU(),
            #nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            #nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class UNet(nn.Module):
    def __init__(self, c_in=4, c_out=4, img_size=128, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 32)  # c_in, c_out
        self.inc2 = DoubleConv(32, 32, residual=True)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)  # c_in, c_out
        self.sa1 = SelfAttention(128, int(img_size / 4))  # c_in, image_size
        self.down3 = Down(128, 256)
        self.sa2 = SelfAttention(256, int(img_size / 8))
        self.down4 = Down(256, 256)
        self.sa3 = SelfAttention(256, int(img_size / 16))

        self.bot1 = DoubleConv(256, 512)
        #self.bot = DoubleConv(256, 256)
        self.bot3 = DoubleConv(512, 256)

        self.sa4 = SelfAttention(256,int(img_size / 16))
        self.up1 = Up(512, 128)
        self.sa5 = SelfAttention(128, int(img_size / 8))
        self.up2 = Up(256, 64)
        self.sa6 = SelfAttention(64, int(img_size / 4))
        self.up3 = Up(128, 32)
        self.up4 = Up(64,32)
        self.outc = nn.Conv2d(32, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forward(self, x, t):
        x1 = self.inc(x)
        x1 = self.inc2(x1)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x3 = self.sa1(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa2(x4)
        x5 = self.down4(x4, t)
        x5 = self.sa3(x5)

        x5 = self.bot1(x5)
        #x5 = self.bot(x5)
        x5 = self.bot3(x5)

        x = self.sa4(x5)
        x = self.up1(x, x4, t)
        x = self.sa5(x)
        x = self.up2(x, x3, t)
        x = self.sa6(x)
        x = self.up3(x, x2, t)
        x = self.up4(x, x1, t)
        output = self.outc(x)
        return output

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(x.dtype)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forward(x, t)


class UNet_conditional(UNet):
    def __init__(
        self,
        c_in=4,
        c_out=4,
        img_size=128,
        time_dim=256,
        num_classes=None,
        device="cuda",
    ):
        super().__init__(c_in, c_out, img_size, time_dim, device)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, y):
        # t = t.unsqueeze(-1).type(torch.float)
        t = t.unsqueeze(-1).type(x.dtype)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        return self.unet_forward(x, t)
