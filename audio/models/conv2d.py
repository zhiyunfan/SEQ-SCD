#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Juan Manuel Coria

# The TDNN class defined here was taken from https://github.com/jonasvdd/TDNN

# Please give proper credit to the authors if you are using TDNN based or X-Vector based
# models by citing their papers:

# Waibel, Alexander H., Toshiyuki Hanazawa, Geoffrey E. Hinton, Kiyohiro Shikano and Kevin J. Lang.
# "Phoneme recognition using time-delay neural networks."
# IEEE Trans. Acoustics, Speech, and Signal Processing 37 (1989): 328-339.
# https://pdfs.semanticscholar.org/cd62/c9976534a6a2096a38244f6cbb03635a127e.pdf?_ga=2.86820248.1800960571.1579515113-23298545.1575886658

# Peddinti, Vijayaditya, Daniel Povey and Sanjeev Khudanpur.
# "A time delay neural network architecture for efficient modeling of long temporal contexts."
# INTERSPEECH (2015).
# https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

# Snyder, David, Daniel Garcia-Romero, Gregory Sell, Daniel Povey and Sanjeev Khudanpur.
# "X-Vectors: Robust DNN Embeddings for Speaker Recognition."
# 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2018): 5329-5333.
# https://www.danielpovey.com/files/2018_icassp_xvectors.pdf

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from .pooling import StatsPool
import pdb
def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
    
class CONV2D_front(nn.Module):
    def __init__(
        self,
        channels: list,
        n_features: int,
        kernel_size: int = 3,
        padding: int = 1,
        norm_type: str = 'ln',
    ):
        super(CONV2D_front, self).__init__()
        self.convs = nn.ModuleList([])
        self.norms = nn.ModuleList([])
        self.num_layers = len(channels)
        self.n_features = n_features
        self.channels = channels
        self.norm_type = norm_type
        for i in range(len(channels)):
            if i == 0:
                conv = nn.Conv2d(1, channels[i], kernel_size=kernel_size, stride=2, padding=padding)
            else:
                conv = nn.Conv2d(channels[i-1], channels[i], kernel_size=kernel_size, stride=2, padding=padding)
            if norm_type == 'ln':
                norm = LayerNorm(channels[i])
            elif norm_type == 'bn':
                norm = nn.BatchNorm2d(channels[i])
            else:
                raise ValueError(f'wrong norm type for 2d conv')

            self.convs.append(conv)
            self.norms.append(norm)

    def forward(self, inputs):
        """
        :param x: is one batch of data, x.size(): [batch_size, sequence_length, input_channels]
            sequence length is the dimension of the arbitrary length data
        :return: [batch_size, len(valid_steps), output_dim]
        """
        outputs = inputs.unsqueeze(1)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            outputs = conv(outputs)
            if self.norm_type == 'ln':
                outputs = outputs.transpose(1,3)  # [B, C, T, H] -> [B, H, T, C]
                outputs = norm(outputs)
                outputs = outputs.transpose(1,3)   # [B, H, T, C] -> [B, C, T, H]
            elif self.norm_type == 'bn':
                outputs = norm(outputs)
            outputs = F.relu(outputs)
        batch_size, channels, sequence_length, dim = outputs.size()
        outputs = outputs.transpose(1,2)
        outputs = outputs.contiguous().view(batch_size, sequence_length, dim*channels)
        return outputs
    
    def get_output_dim(self):
        output_dims = self.n_features
        for i in range(self.num_layers):
            output_dims = (output_dims + 1) // 2
        return output_dims * self.channels[-1]


