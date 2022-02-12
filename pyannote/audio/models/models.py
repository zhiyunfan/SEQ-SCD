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
# Hervé BREDIN - http://herve.niderb.fr
# Juan Manuel Coria

from typing import Optional
from typing import Text

import torch
import torch.nn as nn

from .sincnet import SincNet
from .tdnn import TDNN

from .recurrent import Recurrent
from .linear import Linear

from pyannote.audio.train.model import Model
from pyannote.audio.train.model import Resolution
from pyannote.audio.train.model import RESOLUTION_CHUNK
from pyannote.audio.train.model import RESOLUTION_FRAME
from pyannote.audio.train.task import Task
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional
import logging
import math
import numpy as np
import time
from pyannote.audio.models.multihead_attention import MultiheadAttention 
from torch.nn.parameter import Parameter

class Encoder(nn.Module):
    """Recurrent layers

    Parameters
    ----------
    n_features : `int`
        Input feature shape.
    unit : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    hidden_size : `int`, optional
        Number of features in the hidden state h. Defaults to 16.
    num_layers : `int`, optional
        Number of recurrent layers. Defaults to 1.
    bias : `boolean`, optional
        If False, then the layer does not use bias weights. Defaults to True.
    dropout : `float`, optional
        If non-zero, introduces a Dropout layer on the outputs of each layer
        except the last layer, with dropout probability equal to dropout.
        Defaults to 0.
    bidirectional : `boolean`, optional
        If True, becomes a bidirectional RNN. Defaults to False.
    concatenate : `boolean`, optional
        Concatenate output of each layer instead of using only the last one
        (which is the default behavior).
    pool : {'sum', 'max', 'last', 'x-vector'}, optional
        Temporal pooling strategy. Defaults to no pooling.
    """

    def __init__(
        self,
        n_features,
        down_rate,
        unit="LSTM",
        hidden_size=256,
        num_layers=1,
        bias=True,
        dropout=0,
        bidirectional=False,
    ):
        super().__init__()

        self.n_features = n_features
        self.unit = unit
        Klass = getattr(nn, self.unit)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.down_rate = down_rate
        
        self.rnn_ = nn.ModuleList([])
        self.tdnn_ = nn.ModuleList([])
        
        if self.down_rate == 1:
            stride = [1,1,1,1]
        elif self.down_rate == 2:
            stride = [1,1,1,2]
        elif self.down_rate == 4:
            stride = [1,1,2,2]
        elif self.down_rate == 8:
            stride = [1,2,2,2]
        elif self.down_rate == 16:
            stride = [2,2,2,2]
        elif self.down_rate == 32:
            stride = [2,2,4,2]
        else:
            raise ValueError(f'wrong down rate')

        for i in range(len(stride)):
            if i == 0:
                input_dim = n_features
            else:
                input_dim = 2 * self.hidden_size
            
            out_dim = 2 * self.hidden_size

            tdnn = TDNN(
                context=[-2, 2],
                input_channels=input_dim,
                output_channels=out_dim,
                full_context=True,
                stride=stride[i],
                padding=2,
            )

            self.tdnn_.append(tdnn)
            self.n_features = out_dim
  
        for i in range(self.num_layers):
            if i + 1 == self.num_layers:
                dropout = 0
            else:
                dropout = self.dropout

            rnn = Klass(
                2 * self.hidden_size,
                self.hidden_size,
                num_layers=1,
                bias=self.bias,
                batch_first=True,
                dropout=dropout,
                bidirectional=self.bidirectional,
            )
          
            self.rnn_.append(rnn)
            
    def forward(self, features, return_intermediate=False):
        """Apply recurrent layer (and optional temporal pooling)

        Parameters
        ----------
        features : `torch.Tensor`
            Features shaped as (batch_size, n_frames, n_features)
        return_intermediate : `boolean`, optional
            Return intermediate RNN hidden state.

        Returns
        -------
        output : `torch.Tensor`
            TODO. Shape depends on parameters...
        intermediate : `torch.Tensor`
            (num_layers, batch_size, hidden_size * num_directions)
        """
        hidden = None
        output = features

        for j in range(3):
            output = self.tdnn_[j](output)

        output, hidden = self.rnn_[0](output, hidden)
        output = self.tdnn_[3](output)
        output, hidden = self.rnn_[1](output)

        return output

    def dimension():
        doc = "Output features dimension."

        def fget(self):
            if self.num_layers < 1:
                dimension = self.n_features
            else:
                dimension = self.hidden_size

            if self.bidirectional:
                dimension *= 2

            return dimension

        return locals()

    dimension = property(**dimension())

    def intermediate_dimension(self, layer):
        if self.num_layers < 1:
            dimension = self.n_features
        else:
            dimension = self.hidden_size

        if self.bidirectional:
            dimension *= 2

        return dimension

class CIF(nn.Module):
    def __init__(
        self,
        cif_weight_threshold=1,
        encoder_out_dim=256,
        max_history=10,
        weight_active='crelu',
        relu_threshold=1.0,
        using_scaling=True,
        nonlinear_act=None,
        using_kaiming_init=False,
        using_bias_constant_init=False,
        normalize_scalar=1.0,
        ):
        super().__init__()
        self.cif_weight_threshold = cif_weight_threshold
        self.max_history = max_history
        self.weight_active = weight_active
        self.relu_threshold = relu_threshold
        self.using_scaling = using_scaling
        self.nonlinear_act = nonlinear_act
        self.normalize_scalar = normalize_scalar

        n = 2

        self.weight_dense1 = nn.Linear(n * encoder_out_dim, n * encoder_out_dim)
        self.weight_dense2 = nn.Linear(n * encoder_out_dim, 1)

        if using_bias_constant_init == True:
            nn.init.constant_(self.weight_dense1.bias, 0.05)
            nn.init.constant_(self.weight_dense2.bias, 0.05)

        if using_kaiming_init == True:
            nn.init.kaiming_normal_(self.weight_dense1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.weight_dense2.weight, mode='fan_in', nonlinearity='relu')

        if self.nonlinear_act == 'relu':
            self.diff_act = torch.nn.ReLU()
        elif self.nonlinear_act == 'tanh':
            self.diff_act = torch.nn.Tanh()
        else:
            pass

        if weight_active == 'crelu':
            self.crelu = torch.nn.ReLU()

        if weight_active == 'leakyrelu' or weight_active == 'leakyrelu_nc' :
            self.leakyrelu = torch.nn.LeakyReLU(0.01)


    def forward(self, encoder_outputs, mask, is_training):
        """The CIF part in between the encoder and decoder,
           firing the representation of one acoustic unit
           when the information is accumulated over the given threshold.

        Args:
          encoder_outputs: a Tensor of shape [batch_size, down_sampled_length, hidden_size].
          not_padding: shows the effective and non-effective, with shape [batch_size, down_sampled_length].
          is_training: a boolean.
          targets: the batch of targets used for score scaling, with shape [batch_size, target_length]


        Returns:
          cif_outputs: the output of the CIF part, with shape [batch_size, cif_output_length, hidden_size]
          not_padding_after_cif: shows the effective and non-effective.
          sum_a: the sum of predicted score of each high-level frame, used for loss calculation.
          integrated_logits_on_encoder: the integrated logits by using the same weight estimated by CIF,
                                        with shape [batch_size, cif_output_length, vocab_size]
        """
        encoder_outputs_ori = encoder_outputs
        first_padding_pos = (torch.sum(torch.abs(encoder_outputs), -1) != 0.0).int().sum(1)

        threshold = self.cif_weight_threshold
        batch_size = encoder_outputs.size(0)
        encoder_output_length = encoder_outputs.size(1)
        hidden_size = encoder_outputs.size(2)

        accumulated_weights = torch.zeros([batch_size, 0]).cuda()
        accumulated_states = torch.zeros([batch_size, 0, hidden_size]).cuda()
        fired_states = torch.zeros([batch_size, 0, hidden_size]).cuda()
        fired_flag = torch.zeros([batch_size, 0, 1]).cuda()
        
        history_chunk_keep = torch.zeros(batch_size, 0 , hidden_size).cuda()
        history_std_keep = torch.zeros(batch_size, 0 , hidden_size).cuda()

        for t in range(encoder_output_length):
            if t == 0:
                history_chunk = encoder_outputs[:, 0, :]
            else:
                start = max(0, t-self.max_history)
                history_chunk = torch.mean(encoder_outputs[:, start:t, :], 1)
            
            # fuse future information
            history_chunk_keep = torch.cat([history_chunk_keep, history_chunk.unsqueeze(1)], 1)

        #计算差异量
        ht = encoder_outputs

        his_diff = ht - history_chunk_keep

        info = his_diff
        info = torch.cat([encoder_outputs, info], -1)
 
        weight = self.weight_dense1(info)

        if self.nonlinear_act != None:
            weight = self.diff_act(weight)
        
        weight = self.weight_dense2(weight)

        if self.weight_active == 'hardsigmoid':
            weight = torch.nn.functional.hardsigmoid(weight)
        elif self.weight_active == 'sigmoid':
            weight = torch.sigmoid(weight)
        elif self.weight_active == 'crelu':
            weight = self.crelu(weight)
            weight = torch.clip(weight, max=self.relu_threshold)
        else:
            pass
        weight = weight.squeeze(-1)

        # keep original weight
        ori_weight = weight

        # scaling
        if is_training and self.using_scaling and mask != None:
            sum_a = weight.sum(1)
            N = mask[:,:,0].sum(-1) - 1
            scale = N/torch.clip(sum_a, min=1e-8)
            scale = torch.where(sum_a == 0,
                                torch.zeros_like(sum_a),
                                scale)
            #print(N, sum_a, scale)
            weight = scale.unsqueeze(-1).repeat(1, encoder_output_length) * weight
        else:
            sum_a = weight.sum(1)


        for t in range(encoder_output_length):
            if t == 0:
                prev_accumulated_weight = torch.zeros([batch_size]).cuda()
                prev_accumulated_state = encoder_outputs[:, 0, :]
            else:
                prev_accumulated_weight = accumulated_weights[:, t-1]
                prev_accumulated_state = accumulated_states[:, t-1, :]
 
            cur_weight = weight[:,t]

            cur_weight = torch.where(torch.full([batch_size], t).cuda() > first_padding_pos,
                                       torch.zeros([batch_size]).cuda(),
                                       cur_weight)

            cur_is_fired = prev_accumulated_weight + cur_weight > threshold
            cur_is_fired = cur_is_fired.int()
            remained_weight = threshold - prev_accumulated_weight
            cur_accumulated_weight = torch.where(cur_is_fired == 1,
                                            cur_weight - remained_weight,
                                            cur_weight + prev_accumulated_weight)

            cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                                encoder_outputs[:, t, :],
                                                prev_accumulated_state + (1-cur_weight.unsqueeze(-1))*encoder_outputs[:, t, :])
            cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                        prev_accumulated_state + (1-cur_weight.unsqueeze(-1))*encoder_outputs[:, t, :],
                                        torch.zeros([batch_size, hidden_size]).cuda()) 
             
            # handling the speech tail by rounding up and down
            cur_fired_state = torch.where(torch.full([batch_size, 1], t).cuda() > first_padding_pos.unsqueeze(-1),
                                            torch.zeros([batch_size, hidden_size]).cuda(),
                                            cur_fired_state) 

            fired_flag = torch.cat([fired_flag, cur_is_fired.unsqueeze(-1).unsqueeze(-1)], 1)
            accumulated_weights = torch.cat([accumulated_weights, cur_accumulated_weight.unsqueeze(-1)], 1)
            accumulated_states = torch.cat([accumulated_states, cur_accumulated_state.unsqueeze(1)], 1)                     
            fired_states = torch.cat([fired_states, cur_fired_state.unsqueeze(1)], 1)

        tail_fired_states = cur_accumulated_state
        fired_states = torch.cat([fired_states, tail_fired_states.unsqueeze(1)], 1)

        fired_marks = (torch.sum(torch.abs(fired_states), -1) != 0.0).int()
        fired_utt_length = fired_marks.sum(-1)
        fired_max_length = torch.max(fired_utt_length).int()

        cif_outputs = torch.zeros([0, fired_max_length, hidden_size]).cuda()

        for j in range(batch_size):
            cur_utt_fired_mark = fired_marks[j, :]
            cur_utt_fired_state = fired_states[j, :, :]

            cur_utt_output = cur_utt_fired_state[cur_utt_fired_mark == 1]
            cur_utt_length = cur_utt_output.size(0)

            pad_length = fired_max_length - cur_utt_length

            cur_utt_output = torch.cat([cur_utt_output,
                                        torch.full([pad_length, hidden_size], 0.0).cuda()], 0)
            cur_utt_output = cur_utt_output.unsqueeze(0)
            cif_outputs = torch.cat([cif_outputs, cur_utt_output], 0)
        
        # calculate the not_padding according to the cif_outputs
        not_padding_after_cif = (torch.sum(torch.abs(cif_outputs), -1) != 0.0).int()
        cif_outputs = self.normalize_scalar * F.normalize(cif_outputs, p=2, dim=-1)

        return cif_outputs, ori_weight, fired_flag, not_padding_after_cif, sum_a

class SEQSCD(Model):
    """waveform -> SincNet -> RNN [-> merge] [-> time_pool] -> FC -> output

    Parameters
    ----------
    sincnet : `dict`, optional
        SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
        default parameters. Use {'skip': True} to use handcrafted features
        instead of waveforms: [ waveform -> SincNet -> RNN -> ... ] then
        becomes [ features -> RNN -> ...].
    rnn : `dict`, optional
        Recurrent network parameters. Defaults to `RNN` default parameters.
    ff : `dict`, optional
        Feed-forward layers parameters. Defaults to `FF` default parameters.
    """

    @staticmethod
    def get_alignment(task: Task, sincnet=None, **kwargs):
        """Get frame alignment"""

        if sincnet is None:
            sincnet = dict()

        if sincnet.get("skip", False):
            return "center"

        return SincNet.get_alignment(task, **sincnet)

    @staticmethod
    def get_resolution(
        task: Task,
        sincnet: Optional[dict] = None,
        rnn: Optional[dict] = None,
        **kwargs,
    ) -> Resolution:
        """Get sliding window used for feature extraction

        Parameters
        ----------
        task : Task
        sincnet : dict, optional
        rnn : dict, optional

        Returns
        -------
        sliding_window : `pyannote.core.SlidingWindow` or {`window`, `frame`}
            Returns RESOLUTION_CHUNK if model returns one vector per input
            chunk, RESOLUTION_FRAME if model returns one vector per input
            frame, and specific sliding window otherwise.
        """

        if rnn is None:
            rnn = {"pool": None}

        if rnn.get("pool", None) is not None:
            return RESOLUTION_CHUNK

        if sincnet is None:
            sincnet = {"skip": False}

        if sincnet.get("skip", False):
            return RESOLUTION_FRAME

        return SincNet.get_resolution(task, **sincnet)

    def init(
        self,
        decoder: Optional[dict] = None,
        sincnet: Optional[dict] = None,
        rnn: Optional[dict] = None,
        cif: Optional[dict] = None,
        codebook: Optional[dict] = None,
        loss_cfg: Optional[dict] = None,
        training: bool = True,
    ):
        """waveform -> SincNet -> RNN [-> merge] [-> time_pool] -> FC -> output

        Parameters
        ----------
        sincnet : `dict`, optional
            SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
            default parameters. Use {'skip': True} to use handcrafted features
            instead of waveforms: [ waveform -> SincNet -> RNN -> ... ] then
            becomes [ features -> RNN -> ...].
        rnn : `dict`, optional
            Recurrent network parameters. Defaults to `RNN` default parameters.
        """

        if sincnet is None:
            sincnet = dict()
        self.sincnet = sincnet
        self.loss_cfg = loss_cfg
        self.down_rate = loss_cfg["down_rate"]
        self.decoder = decoder
        self.rnn = rnn
        self.codebook = codebook
        self.training = training
        self.normalize_scalar = cif["normalize_scalar"]
        n_features = self.n_features
        
        if not sincnet.get("skip", False):
            if n_features != 1:
                msg = (
                    f"SincNet only supports mono waveforms. "
                    f"Here, waveform has {n_features} channels."
                )
                raise ValueError(msg)
            self.sincnet_ = SincNet(**sincnet)
            n_features = self.sincnet_.dimension

        if self.rnn is None:
            self.rnn = dict()

        self.rnn_ = Encoder(n_features, self.down_rate, **rnn)
        n_features = self.rnn_.dimension

        cif['encoder_out_dim'] = n_features
        self.cif = cif
        self.cif_ = CIF(**cif)

        if self.rnn['bidirectional'] == True:
            hidden_size = self.rnn['hidden_size'] * 2
        else:
            hidden_size = self.rnn['hidden_size']

        self.dec_linear = nn.Linear(hidden_size, self.rnn['hidden_size'], bias=True)
        self.activation_hid = torch.nn.ReLU()

        num_class = loss_cfg["num_spk_class"]
        self.activation_ = torch.nn.Sigmoid()
        self.linear_ = nn.Linear(self.rnn['hidden_size'], num_class, bias=True)

        # spk code
        if self.codebook["is_used"] == True:
            self.spkcode = Parameter(torch.Tensor(codebook["code_len"], codebook["code_size"]))
            nn.init.kaiming_uniform_(self.spkcode, a=math.sqrt(5))

            self.self_attn = MultiheadAttention(
                self.codebook["attention_size"],
                self.codebook["num_attention_heads"],
                dropout=self.codebook["attention_dropout"],
            )

    def forward(self, waveforms, mask=None):
        """Forward pass

        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1) `torch.Tensor`
            Batch of waveforms. In case SincNet is skipped, a tensor with shape
            (batch_size, n_samples, n_features) is expected.
        Returns
        -------
        output : `torch.Tensor`
            Final network output.
        """
        #waveforms.size()  [64, 32000, 1]
        if self.sincnet.get("skip", False):
            output = waveforms
        else:
            output = self.sincnet_(waveforms)

        # encoder 
        output = self.rnn_(output)

        # CIF
        output, weight_keep, fired_flag, not_padding_after_cif, sum_a = self.cif_(output, mask, self.training)
        weight_keep = weight_keep.unsqueeze(-1)
        cif_output = output

        # spk codebook
        batch_size = output.size(0)
        if self.codebook["is_used"]:
            spk_code = self.spkcode.unsqueeze(0).repeat(batch_size, 1, 1)
            output, _ = self.self_attn(
                query=output.transpose(0,1),
                key=spk_code.transpose(0,1),
                value=spk_code.transpose(0,1),
                need_weights=False,
                attn_mask=None,
                )
            output = output.transpose(0,1)
        # Decoder
        # hidden FC layer 
        output = self.dec_linear(output)
        output = self.activation_hid(output)

        # output FC layer
        output = self.linear_(output)
        output = self.activation_(output)

        return cif_output, output, not_padding_after_cif, sum_a, torch.cat([1-fired_flag, fired_flag], -1), self.loss_cfg

    @property
    def dimension(self):
        return Model.dimension.fget(self)

    def intermediate_dimension(self, layer):
        if layer == 0:
            return self.sincnet_.dimension
        return self.rnn_.intermediate_dimension(layer - 1)
