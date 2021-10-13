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

import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNN(nn.Module):
    def __init__(
        self,
        context: list,
        input_channels: int,
        output_channels: int,
        full_context: bool = True,
        stride: int = 1,
        padding: int = 0,
    ):
        """
        Implementation of a 'Fast' TDNN layer by exploiting the dilation argument of the PyTorch Conv1d class

        Due to its fastness the context has gained two constraints:
            * The context must be symmetric
            * The context must have equal spacing between each consecutive element

        For example: the non-full and symmetric context {-3, -2, 0, +2, +3} is not valid since it doesn't have
        equal spacing; The non-full context {-6, -3, 0, 3, 6} is both symmetric and has an equal spacing, this is
        considered valid.

        :param context: The temporal context
        :param input_channels: The number of input channels
        :param output_channels: The number of channels produced by the temporal convolution
        :param full_context: Indicates whether a full context needs to be used
        """
        super(TDNN, self).__init__()
        self.full_context = full_context
        self.input_dim = input_channels
        self.output_dim = output_channels

        context = sorted(context)
        self.check_valid_context(context, full_context)
        if full_context:
            kernel_size = context[-1] - context[0] + 1 if len(context) > 1 else 1
            self.temporal_conv = nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding)
            self.BN = nn.BatchNorm1d(output_channels)
        else:
            # use dilation
            delta = context[1] - context[0]
            self.temporal_conv = nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size=len(context),
                    dilation=delta,
                    stride=stride,
                    padding=padding,
                )
            self.BN = nn.BatchNorm1d(output_channels)
            

    def forward(self, x):
        """
        :param x: is one batch of data, x.size(): [batch_size, sequence_length, input_channels]
            sequence length is the dimension of the arbitrary length data
        :return: [batch_size, len(valid_steps), output_dim]
        """
        x = self.temporal_conv(torch.transpose(x, 1, 2))
        x = self.BN(x) 
        x = torch.transpose(x, 1, 2)
        return F.relu(x)

    @staticmethod
    def check_valid_context(context: list, full_context: bool) -> None:
        """
        Check whether the context is symmetrical and whether and whether the passed
        context can be used for creating a convolution kernel with dil

        :param full_context: indicates whether the full context (dilation=1) will be used
        :param context: The context of the model, must be symmetric if no full context and have an equal spacing.
        """
        if full_context:
            assert (
                len(context) <= 2
            ), "If the full context is given one must only define the smallest and largest"
            if len(context) == 2:
                assert context[0] + context[-1] == 0, "The context must be symmetric"
        else:
            assert len(context) % 2 != 0, "The context size must be odd"
            assert (
                context[len(context) // 2] == 0
            ), "The context contain 0 in the center"
            if len(context) > 1:
                delta = [context[i] - context[i - 1] for i in range(1, len(context))]
                assert all(
                    delta[0] == delta[i] for i in range(1, len(delta))
                ), "Intra context spacing must be equal!"
