# -*- coding: utf-8 -*-
"""BiPredNoiseNetV2: STFDiff noise predictor with structural innovations.

Innovations (independently switchable, from work/ARCH_INNOVATIONS.md):
  I   delta_enc  : explicit temporal-difference encoder in the input stage
  II  time_gate  : time-conditioned adaptive gating of the clean condition
  III reinject   : multi-scale condition re-injection in the U-Net decoder

The default (all off) is equivalent to the original BiPredNoiseNet.
"""
from __future__ import annotations

import torch
from torch import nn

from bi_pred_resnet import (default, ResBlock, Downsample, Upsample,
                            SinusoidalPosEmb, LearnedSinusoidalPosEmb)


class BiPredNoiseNetV2(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4),
        channels=3,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        learned_sinusoidal_dim=16,
        delta_enc=False,
        time_gate=False,
        reinject=False,
    ):
        super().__init__()
        self.delta_enc = delta_enc
        self.time_gate = time_gate
        self.reinject = reinject

        time_dim = dim * 4
        self.learned_sinusoidal_cond = learned_sinusoidal_cond
        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)
        init_dim = default(init_dim, dim)

        self.init_conv = nn.Conv2d(input_channels, init_dim, 3, 1, 1)
        self.fine_init_conv = nn.Conv2d(input_channels * 2, init_dim, 3, 1, 1)
        self.coarse_init_conv = nn.Conv2d(input_channels * 3, init_dim, 3, 1, 1)
        self.noisy_init_conv = nn.Conv2d(input_channels, init_dim, 3, 1, 1)

        # Innovation I: explicit temporal-difference encoder
        if self.delta_enc:
            self.delta_init = nn.Conv2d(channels * 3, init_dim, 3, 1, 1)
            self.clean_init_conv = nn.Conv2d(init_dim * 3, init_dim, 1)
        else:
            self.clean_init_conv = nn.Conv2d(init_dim * 2, init_dim, 1)

        # Innovation II: time-conditioned clean gate
        if self.time_gate:
            self.clean_gate = nn.Sequential(
                nn.Linear(time_dim, init_dim),
                nn.Sigmoid(),
            )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.clean_downs = nn.ModuleList([])
        self.noisy_downs = nn.ModuleList([])
        self.noise_ups = nn.ModuleList([])

        # Innovation III: per-scale condition re-injection
        if self.reinject:
            # x_clean channel at each scale is the running dim: init_dim -> d_out sequence
            clean_chans = [init_dim] + [d_out for _, d_out in in_out[:-1]]
            self.cond_proj = nn.ModuleList([
                nn.Conv2d(c, d_in, 1) for c, (d_in, _) in zip(clean_chans, in_out)
            ])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.clean_downs.append(nn.ModuleList([
                ResBlock(dim_in, dim_in, groups=resnet_block_groups),
                Downsample(dim_in, dim_out) if not is_last
                else nn.Conv2d(dim_in, dim_out, 3, padding=1),
            ]))
            noisy_in = dim_in * 2 if self.reinject else dim_in
            self.noisy_downs.append(nn.ModuleList([
                ResBlock(noisy_in, dim_in, time_emb_dim=time_dim,
                         groups=resnet_block_groups),
                Downsample(dim_in, dim_out) if not is_last
                else nn.Conv2d(dim_in, dim_out, 3, padding=1),
            ]))

        mid_dim = dims[-1]
        self.clean_mid_block = ResBlock(mid_dim, mid_dim, groups=resnet_block_groups)
        self.noisy_mid_block = ResBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=8)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.noise_ups.append(nn.ModuleList([
                ResBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim,
                         groups=resnet_block_groups),
                Upsample(dim_out, dim_in) if not is_last
                else nn.Conv2d(dim_out, dim_in, 3, padding=1),
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = ResBlock(dim * 2, dim, time_emb_dim=time_dim,
                                        groups=resnet_block_groups)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, coarse_img_01, coarse_img_02, coarse_img_03,
                fine_img_01, noisy_fine_img_02, fine_img_03,
                time, x_self_cond=None):
        x_fine = self.fine_init_conv(torch.cat((fine_img_01, fine_img_03), dim=1))
        x_coarse = self.coarse_init_conv(
            torch.cat((coarse_img_01, coarse_img_02, coarse_img_03), dim=1))
        x_noisy = self.noisy_init_conv(noisy_fine_img_02)

        if self.delta_enc:
            dC1 = coarse_img_02 - coarse_img_01
            dC2 = coarse_img_03 - coarse_img_02
            dF = fine_img_03 - fine_img_01
            z_delta = self.delta_init(torch.cat((dC1, dC2, dF), dim=1))
            x_clean = self.clean_init_conv(torch.cat((x_fine, x_coarse, z_delta), dim=1))
        else:
            x_clean = self.clean_init_conv(torch.cat((x_fine, x_coarse), dim=1))

        t = self.time_mlp(time)

        if self.time_gate:
            g = self.clean_gate(t).view(-1, x_clean.shape[1], 1, 1)
            # sigmoid(gate(t)): learned per-channel trust in the reference
            x_clean_eff = g * x_clean
        else:
            x_clean_eff = x_clean

        noise = x_noisy - x_clean_eff
        r = noise.clone()
        h = []

        for i, ((clean_res, clean_ds), (noisy_res, noisy_ds)) in enumerate(
                zip(self.clean_downs, self.noisy_downs)):
            x_clean = clean_res(x_clean)
            if self.reinject:
                cond = self.cond_proj[i](x_clean)
                noisy_in = torch.cat((x_noisy, cond), dim=1)
            else:
                noisy_in = x_noisy
            x_noisy = noisy_res(noisy_in, t)
            noise = x_noisy - x_clean
            h.append(noise)
            x_clean = clean_ds(x_clean)
            x_noisy = noisy_ds(x_noisy)

        x_clean = self.clean_mid_block(x_clean)
        x_noisy = self.noisy_mid_block(x_noisy, t)
        noise = x_noisy - x_clean

        for noise_res, noise_up in self.noise_ups:
            noise = torch.cat((noise, h.pop()), dim=1)
            noise = noise_res(noise, t)
            noise = noise_up(noise)

        noise = torch.cat((noise, r), dim=1)
        noise = self.final_res_block(noise, t)
        return self.final_conv(noise)
