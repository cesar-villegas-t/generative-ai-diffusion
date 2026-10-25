"""
Module: score_model.py
Defines the ScoreNet architecture used for score-based generative modeling.
"""

import torch
import torch.nn as nn


class GaussianRandomFourierFeatures(nn.Module):
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.rff_weights = nn.Parameter(
            torch.randn(embed_dim // 2) * scale,
            requires_grad=False,
        )

    def forward(self, t):
        x_proj = t[:, None] * self.rff_weights[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)[:, :, None, None]


class ScoreNet(nn.Module):
    def __init__(self, marginal_prob_std, embed_dim=256, num_classes=None):
        super().__init__()

        self.num_classes = num_classes
        self.class_embedding = nn.Embedding(num_classes, embed_dim) if num_classes is not None else None

        self.nf = 128

        self.embed = nn.Sequential(
            GaussianRandomFourierFeatures(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        self.conv1 = nn.Conv2d(3, self.nf, 3, stride=1, padding=1, bias=False)
        self.dense1 = Dense(embed_dim, self.nf)
        self.gnorm1 = nn.GroupNorm(8, self.nf)

        self.conv2 = nn.Conv2d(self.nf, self.nf, 3, stride=2, padding=1, bias=False)
        self.dense2 = Dense(embed_dim, self.nf)
        self.gnorm2 = nn.GroupNorm(8, self.nf)

        self.conv3 = nn.Conv2d(self.nf, self.nf, 3, stride=2, padding=1, bias=False)
        self.dense3 = Dense(embed_dim, self.nf)
        self.gnorm3 = nn.GroupNorm(8, self.nf)

        self.conv4 = nn.Conv2d(self.nf, self.nf, 3, stride=2, padding=1, bias=False)
        self.dense4 = Dense(embed_dim, self.nf)
        self.gnorm4 = nn.GroupNorm(8, self.nf)

        self.tconv4 = nn.ConvTranspose2d(self.nf, self.nf, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense5 = Dense(embed_dim, self.nf)
        self.tgnorm4 = nn.GroupNorm(8, self.nf)

        self.tconv3 = nn.ConvTranspose2d(self.nf + self.nf, self.nf, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense6 = Dense(embed_dim, self.nf)
        self.tgnorm3 = nn.GroupNorm(8, self.nf)

        self.tconv2 = nn.ConvTranspose2d(self.nf + self.nf, self.nf, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense7 = Dense(embed_dim, self.nf)
        self.tgnorm2 = nn.GroupNorm(8, self.nf)

        self.tconv1 = nn.ConvTranspose2d(self.nf + self.nf, 3, 3, stride=1, padding=1)

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t, class_labels=None):
        embed = self.act(self.embed(t))
        if self.class_embedding is not None and class_labels is not None:
            embed += self.class_embedding(class_labels)

        h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))
        h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))

        h = self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))

        if h.shape[-2:] != h3.shape[-2:]:
            min_h = min(h.shape[-2], h3.shape[-2])
            min_w = min(h.shape[-1], h3.shape[-1])
            h = h[:, :, :min_h, :min_w]
            h3 = h3[:, :, :min_h, :min_w]
        h = self.act(self.tgnorm3(self.tconv3(torch.cat([h, h3], dim=1)) + self.dense6(embed)))

        if h.shape[-2:] != h2.shape[-2:]:
            min_h = min(h.shape[-2], h2.shape[-2])
            min_w = min(h.shape[-1], h2.shape[-1])
            h = h[:, :, :min_h, :min_w]
            h2 = h2[:, :, :min_h, :min_w]
        h = self.act(self.tgnorm2(self.tconv2(torch.cat([h, h2], dim=1)) + self.dense7(embed)))

        if h.shape[-2:] != h1.shape[-2:]:
            min_h = min(h.shape[-2], h1.shape[-2])
            min_w = min(h.shape[-1], h1.shape[-1])
            h = h[:, :, :min_h, :min_w]
            h1 = h1[:, :, :min_h, :min_w]
        h = self.tconv1(torch.cat([h, h1], dim=1))

        return h / self.marginal_prob_std(t)[:, None, None, None]
