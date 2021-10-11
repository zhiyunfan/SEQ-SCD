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
from .tdnn import XVectorNet
from .pooling import TemporalPooling
from .tdnn import TDNN
from .conv2d import CONV2D_front

from .convolutional import Convolutional
from .recurrent import Recurrent
from .linear import Linear
from .pooling import Pooling
from .scaling import Scaling


from pyannote.audio.train.model import Model
from pyannote.audio.train.model import Resolution
from pyannote.audio.train.model import RESOLUTION_CHUNK
from pyannote.audio.train.model import RESOLUTION_FRAME
from pyannote.audio.train.task import Task
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional
import pdb
import logging
import math
from .modules import MultiheadAttention
import numpy as np
import time
from pyannote.audio.utils.AutomaticWeightedLoss import AutomaticWeightedLoss
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def is_xla_tensor(tensor):
    return torch.is_tensor(tensor) and tensor.device.type == 'xla'


def index_put(tensor, indices, value):
    if is_xla_tensor(tensor):
        for _ in range(indices.dim(), tensor.dim()):
            indices = indices.unsqueeze(-1)
        if indices.size(-1) < tensor.size(-1):
            indices = indices.expand_as(tensor)
        tensor = torch.mul(tensor, ~indices) + torch.mul(value, indices)
    else:
        tensor[indices] = value
    return tensor


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def get_activation_fn(activation: str) -> Callable:
    """ Returns the activation function corresponding to `activation` """
    from .modules import gelu, gelu_accurate

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        deprecation_warning(
            "--activation-fn=gelu_fast has been renamed to gelu_accurate"
        )
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))



class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn

class SA(nn.Module):
    def __init__(self, 
                  in_dim, 
                  encoder_embed_dim,
                  dropout,
                  conv_pos,
                  conv_pos_groups,
                  encoder_ffn_embed_dim,
                  encoder_attention_heads,
                  attention_dropout,
                  activation_dropout,
                  activation_fn,
                  layer_norm_first,
                  encoder_layers,
                  encoder_layerdrop,
                ):
        super().__init__()

        self.dropout = dropout
        self.in_dim = in_dim
        self.embedding_dim = encoder_embed_dim

        self.first_affine = nn.Linear(in_dim, encoder_embed_dim)

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=conv_pos,
            padding=conv_pos // 2,
            groups=conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(conv_pos), nn.GELU())
        

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=encoder_ffn_embed_dim,
                    num_attention_heads=encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    layer_norm_first=layer_norm_first,
                )
                for _ in range(encoder_layers)
            ]
        )

        self.layer_norm_first = layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = encoder_layerdrop


    def forward(self, x, is_training, padding_mask=None):
        x = self.extract_features(x, is_training, padding_mask)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x

    def extract_features(self, x, is_training, padding_mask=None):

        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x = self.first_affine(x)
        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=is_training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        #layer_results = []
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not is_training or (dropout_probability > self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                #layer_results.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x


    def dimension(self):
        return self.embedding_dim


class TDNNSA(nn.Module):
    def __init__(self, 
                  in_dim, 
                  down_rate,
                  encoder_embed_dim,
                  dropout,
                  conv_pos,
                  conv_pos_groups,
                  encoder_ffn_embed_dim,
                  encoder_attention_heads,
                  attention_dropout,
                  activation_dropout,
                  activation_fn,
                  layer_norm_first,
                  encoder_layers,
                  encoder_layerdrop,
                ):
        super().__init__()

        self.dropout = dropout
        self.in_dim = in_dim
        self.down_rate = down_rate

        self.tdnn_ = nn.ModuleList([])
        self.tdnn_layers = int(math.log(self.down_rate)/math.log(2))
        self.encoder_layers = encoder_layers


        if self.down_rate == 16 or self.down_rate == 8:
            for i in range(self.tdnn_layers):
                if i > 0:
                    input_dim = encoder_embed_dim
                else:
                    input_dim = in_dim
                tdnn = TDNN(
                    context=[-2, 2],
                    input_channels=input_dim,
                    output_channels=encoder_embed_dim,
                    full_context=True,
                    stride=2,
                    padding=2,
                )
                self.tdnn_.append(tdnn)
        elif self.down_rate == 2 or self.down_rate == 4:
            for i in range(self.tdnn_layers):
                tdnn = TDNN(
                    context=[-2, 2],
                    input_channels=encoder_embed_dim,
                    output_channels=encoder_embed_dim,
                    full_context=True,
                    stride=2,
                    padding=2,
                )
                self.tdnn_.append(tdnn)
        else:
            for i in range(self.tdnn_layers-1):
                if i > 0:
                    input_dim = encoder_embed_dim
                else:
                    input_dim = in_dim
                tdnn = TDNN(
                    context=[-2, 2],
                    input_channels=input_dim,
                    output_channels=encoder_embed_dim,
                    full_context=True,
                    stride=2,
                    padding=2,
                )
                self.tdnn_.append(tdnn)   

            tdnn = TDNN(
                    context=[-2, 2],
                    input_channels=encoder_embed_dim,
                    output_channels=encoder_embed_dim,
                    full_context=True,
                    stride=2,
                    padding=1,
                )

            self.tdnn_.append(tdnn)


        self.embedding_dim = encoder_embed_dim

        if self.tdnn_layers > self.encoder_layers:
            self.first_affine = nn.Linear(encoder_embed_dim, encoder_embed_dim)
        else:
            self.first_affine = nn.Linear(in_dim, encoder_embed_dim)


        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=conv_pos,
            padding=conv_pos // 2,
            groups=conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(conv_pos), nn.GELU())
        

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=encoder_ffn_embed_dim,
                    num_attention_heads=encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    layer_norm_first=layer_norm_first,
                )
                for _ in range(encoder_layers)
            ]
        )

        self.layer_norm_first = layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = encoder_layerdrop


    def forward(self, x, is_training, padding_mask=None):
        x = self.extract_features(x, is_training, padding_mask)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x

    def extract_features(self, x, is_training, padding_mask=None):

        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        if self.tdnn_layers > self.encoder_layers:
            low_tdnn_num = self.tdnn_layers - self.encoder_layers
            for i in range(low_tdnn_num):
                x = self.tdnn_[i](x)
        else:
            low_tdnn_num = 0
        x = self.first_affine(x)
        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=is_training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        #layer_results = []
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not is_training or (dropout_probability > self.layerdrop):
                if i < len(self.tdnn_):
                    x = x.transpose(0, 1)
                    x = self.tdnn_[i+low_tdnn_num](x)
                    x = x.transpose(0, 1)
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                #layer_results.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x


    def dimension(self):
        return self.embedding_dim



class RNN(nn.Module):
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
        unit="LSTM",
        hidden_size=16,
        num_layers=1,
        bias=True,
        dropout=0,
        bidirectional=False,
        concatenate=False,
        using_residual=False,
        pool=None,
        tdnn_norm_type='wn',
        mid_layer_insert_lstm='tdnn',
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
        self.concatenate = concatenate
        self.pool = pool
        self.pool_ = TemporalPooling.create(pool) if pool is not None else None
        self.using_residual = using_residual


        if num_layers < 1:
            msg = '"bidirectional" must be set to False when num_layers < 1'
            if bidirectional:
                raise ValueError(msg)
            msg = '"concatenate" must be set to False when num_layers < 1'
            if concatenate:
                raise ValueError(msg)
            return

        if self.concatenate:
            self.rnn_ = nn.ModuleList([])
            for i in range(self.num_layers):

                if i > 0:
                    input_dim = self.hidden_size
                    if self.bidirectional:
                        input_dim *= 2
                else:
                    input_dim = self.n_features

                if i + 1 == self.num_layers:
                    dropout = 0
                else:
                    dropout = self.dropout

                rnn = Klass(
                    input_dim,
                    self.hidden_size,
                    num_layers=1,
                    bias=self.bias,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=self.bidirectional,
                )

                self.rnn_.append(rnn)

        else:
            self.rnn_ = Klass(
                self.n_features,
                self.hidden_size,
                num_layers=self.num_layers,
                bias=self.bias,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            )

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

        if self.num_layers < 1:

            if return_intermediate:
                msg = (
                    '"return_intermediate" must be set to False ' "when num_layers < 1"
                )
                raise ValueError(msg)

            output = features

        else:

            if return_intermediate:
                num_directions = 2 if self.bidirectional else 1

            if self.concatenate:

                if return_intermediate:
                    msg = (
                        '"return_intermediate" is not supported '
                        'when "concatenate" is True'
                    )
                    raise NotADirectoryError(msg)

                outputs = []

                hidden = None
                output = None
                # apply each layer separately...
                for i, rnn in enumerate(self.rnn_):
                    if i > 0:
                        output, hidden = rnn(output, hidden)
                    else:
                        output, hidden = rnn(features)
                    outputs.append(output)

                # ... and concatenate their output
                output = torch.cat(outputs, dim=2)

            else:
                output, hidden = self.rnn_(features)

                if return_intermediate:
                    if self.unit == "LSTM":
                        h = hidden[0]
                    elif self.unit == "GRU":
                        h = hidden

                    # to (num_layers, batch_size, num_directions * hidden_size)
                    h = h.view(self.num_layers, num_directions, -1, self.hidden_size)
                    intermediate = (
                        h.transpose(2, 1)
                        .contiguous()
                        .view(self.num_layers, -1, num_directions * self.hidden_size)
                    )
        if self.pool_ is not None:
            output = self.pool_(output)

        if return_intermediate:
            return output, intermediate

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

            if self.concatenate:
                dimension *= self.num_layers

            if self.pool == "x-vector":
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


class RNN_maxp(nn.Module):
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
        unit="LSTM",
        hidden_size=16,
        num_layers=1,
        bias=True,
        dropout=0,
        bidirectional=False,
        concatenate=False,
        using_residual=False,
        pool=None,

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
        self.concatenate = concatenate
        self.pool = pool
        self.pool_ = TemporalPooling.create(pool) if pool is not None else None
        self.using_residual = using_residual

        if num_layers < 1:
            msg = '"bidirectional" must be set to False when num_layers < 1'
            if bidirectional:
                raise ValueError(msg)
            msg = '"concatenate" must be set to False when num_layers < 1'
            if concatenate:
                raise ValueError(msg)
            return

        self.rnn_ = nn.ModuleList([])
        for i in range(self.num_layers):

            if i > 0:
                input_dim = self.hidden_size
                if self.bidirectional:
                    input_dim *= 2
            else:
                input_dim = self.n_features

            if i + 1 == self.num_layers:
                dropout = 0
            else:
                dropout = self.dropout

            rnn = Klass(
                input_dim,
                self.hidden_size,
                num_layers=1,
                bias=self.bias,
                batch_first=True,
                dropout=dropout,
                bidirectional=self.bidirectional,
            )

            self.rnn_.append(rnn)
            self.max_pool1d = nn.MaxPool1d(3, stride=2, padding=1, dilation=1)

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
        output = None
        # apply each layer separately...
        for i, rnn in enumerate(self.rnn_):
            if i > 0:
                output, hidden = rnn(output, hidden)
            else:
                output, hidden = rnn(features)
            if i == 0:
                output = output.transpose(1,2)
                output = self.max_pool1d(output)
                output = output.transpose(1,2)

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

            if self.concatenate:
                dimension *= self.num_layers

            if self.pool == "x-vector":
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

class TDNNRNN(nn.Module):
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
        hidden_size=16,
        num_layers=1,
        bias=True,
        dropout=0,
        bidirectional=False,
        concatenate=False,
        using_residual=False,
        pool=None,
        tdnn_norm_type='wn',
        mid_layer_insert_lstm='tdnn',
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
        self.concatenate = concatenate
        self.pool = pool
        self.pool_ = TemporalPooling.create(pool) if pool is not None else None
        self.down_rate = down_rate
        self.using_residual = using_residual
        self.mid_layer_insert_lstm = mid_layer_insert_lstm

        if num_layers < 1:
            msg = '"bidirectional" must be set to False when num_layers < 1'
            if bidirectional:
                raise ValueError(msg)
            msg = '"concatenate" must be set to False when num_layers < 1'
            if concatenate:
                raise ValueError(msg)
            return
        
        self.rnn_ = nn.ModuleList([])
        self.tdnn_ = nn.ModuleList([])
        self.max_pool_ = nn.ModuleList([])
        
        num_down2_layer = int(math.log(self.down_rate)/math.log(2))
        low_tdnn_num = max(num_down2_layer - self.num_layers, 0)
        out_dim = self.hidden_size*2 if self.bidirectional else self.hidden_size
        self.low_tdnn_num = low_tdnn_num
        self.num_down2_layer = num_down2_layer

        if low_tdnn_num > 0:
            for i in range(low_tdnn_num):
                if i > 0:
                    input_dim = self.hidden_size
                    if self.bidirectional:
                        input_dim *= 2
                else:
                    input_dim = self.n_features

                tdnn = TDNN(
                    context=[-2, 2],
                    input_channels=input_dim,
                    output_channels=out_dim,
                    full_context=True,
                    stride=2,
                    padding=2,
                    norm_type=tdnn_norm_type,
                )
                self.tdnn_.append(tdnn)
                self.n_features = out_dim
        
        for i in range(low_tdnn_num, num_down2_layer):
            if self.mid_layer_insert_lstm == 'tdnn':
                if i > 0:
                    input_dim = self.hidden_size
                    if self.bidirectional:
                        input_dim *= 2
                else:
                    input_dim = self.n_features
                
                if i < 4:
                    tdnn = TDNN(
                        context=[-2, 2],
                        input_channels=input_dim,
                        output_channels=out_dim,
                        full_context=True,
                        stride=2,
                        padding=2,
                        norm_type=tdnn_norm_type,
                    )
                elif i == 4:
                    tdnn = TDNN(
                            context=[-2, 2],
                            input_channels=out_dim,
                            output_channels=out_dim,
                            full_context=True,
                            stride=2,
                            padding=1,
                            norm_type=tdnn_norm_type,
                        )
                else:
                    raise ValueError(f'too many tdnn layers')
                self.tdnn_.append(tdnn)
            elif self.mid_layer_insert_lstm == 'max_pooling':
                max_pool1d = nn.MaxPool1d(3, stride=2, padding=1, dilation=1)
                self.max_pool_.append(max_pool1d)

        for i in range(self.num_layers):

            if i > 0 or low_tdnn_num > 1:
                input_dim = self.hidden_size
                if self.bidirectional:
                    input_dim *= 2
            else:
                input_dim = self.n_features

            if i + 1 == self.num_layers:
                dropout = 0
            else:
                dropout = self.dropout
            if self.bidirectional:
                rnn = Klass(
                    self.hidden_size*2,
                    self.hidden_size,
                    num_layers=1,
                    bias=self.bias,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=self.bidirectional,
                )
            else:
                rnn = Klass(
                    self.hidden_size,
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

        if self.num_layers < 1:

            if return_intermediate:
                msg = (
                    '"return_intermediate" must be set to False ' "when num_layers < 1"
                )
                raise ValueError(msg)

            output = features

        else:
            if return_intermediate:
                num_directions = 2 if self.bidirectional else 1

            if self.concatenate:

                if return_intermediate:
                    msg = (
                        '"return_intermediate" is not supported '
                        'when "concatenate" is True'
                    )
                    raise NotADirectoryError(msg)

                outputs = []

                hidden = None
                output = None
                # apply each layer separately...
                for i, rnn in enumerate(self.rnn_):
                    if i > 0:
                        output, hidden = rnn(output, hidden)
                    else:
                        output, hidden = rnn(features)
                    outputs.append(output)

                # ... and concatenate their output
                output = torch.cat(outputs, dim=2)

            else:
                hidden = None
                output = features
                # apply each layer separately...
                for j in range(self.low_tdnn_num):
                    output = self.tdnn_[j](output)

                if self.mid_layer_insert_lstm == 'tdnn':
                    for i, rnn in enumerate(self.rnn_):
                        if i > self.num_down2_layer - self.low_tdnn_num or i+self.low_tdnn_num > self.num_down2_layer-1:
                            residual = output
                            output, hidden = rnn(output, hidden)
                            if self.using_residual:
                                output = output + residual
                                
                        else:
                            output = self.tdnn_[i+self.low_tdnn_num](output)
                            residual = output
                            output, hidden = rnn(output)
                            if self.using_residual:
                                output = output + residual
                elif self.mid_layer_insert_lstm == 'max_pooling':
                    for i, rnn in enumerate(self.rnn_):
                        if i == self.num_layers: 
                            output, hidden = rnn(output, hidden)
                        else:
                            output, hidden = rnn(output)
                            output = output.transpose(1,2)
                            output = self.max_pool_[i](output)
                            output = output.transpose(1,2)

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

            if self.concatenate:
                dimension *= self.num_layers

            if self.pool == "x-vector":
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

class TDNN4RNN(nn.Module):
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
        concatenate=False,
        using_residual=False,
        pool=None,
        tdnn_norm_type='bn',
        mid_layer_insert_lstm='tdnn',
    ):
        super().__init__()
        tdnn_norm_type='bn'
        mid_layer_insert_lstm='tdnn'

        self.n_features = n_features

        self.unit = unit
        Klass = getattr(nn, self.unit)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.concatenate = concatenate
        self.pool = pool
        self.pool_ = TemporalPooling.create(pool) if pool is not None else None
        self.down_rate = down_rate
        self.using_residual = using_residual
        self.mid_layer_insert_lstm = mid_layer_insert_lstm

        if num_layers < 1:
            msg = '"bidirectional" must be set to False when num_layers < 1'
            if bidirectional:
                raise ValueError(msg)
            msg = '"concatenate" must be set to False when num_layers < 1'
            if concatenate:
                raise ValueError(msg)
            return
        
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
                norm_type='bn',
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

        if self.num_layers < 1:

            if return_intermediate:
                msg = (
                    '"return_intermediate" must be set to False ' "when num_layers < 1"
                )
                raise ValueError(msg)

            output = features

        else:
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

            if self.concatenate:
                dimension *= self.num_layers

            if self.pool == "x-vector":
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


class CONV2DRNN(nn.Module):
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
        hidden_size=16,
        num_layers=1,
        bias=True,
        dropout=0,
        bidirectional=False,
        concatenate=False,
        using_residual=False,
        pool=None,
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
        self.concatenate = concatenate
        self.pool = pool
        self.pool_ = TemporalPooling.create(pool) if pool is not None else None
        self.down_rate = down_rate
        self.using_residual = using_residual

        if num_layers < 1:
            msg = '"bidirectional" must be set to False when num_layers < 1'
            if bidirectional:
                raise ValueError(msg)
            msg = '"concatenate" must be set to False when num_layers < 1'
            if concatenate:
                raise ValueError(msg)
            return

        self.rnn_ = nn.ModuleList([])
        self.conv_ = nn.ModuleList([])
        
        num_conv = int(math.log(self.down_rate)/math.log(2))
        low_conv_num = max(num_conv - self.num_layers, 0)
        out_dim = self.hidden_size*2 if self.bidirectional else self.hidden_size
        self.low_conv_num = low_conv_num
        self.num_conv = num_conv

        if low_conv_num > 0:
            for i in range(low_conv_num):
                if i > 0:
                    input_dim = self.hidden_size
                    if self.bidirectional:
                        input_dim *= 2
                else:
                    input_dim = self.n_features

                conv = CONV2D(
                    input_channels=input_dim,
                    output_channels=out_dim,
                    stride=2,
                    padding=1,
                    kernel_size=3,
                )
                self.conv_.append(conv)
                #self.n_features = out_dim
            
        for i in range(low_conv_num, num_conv):
            if i > 0:
                input_dim = self.hidden_size
                if self.bidirectional:
                    input_dim *= 2
            else:
                input_dim = self.n_features
            
            if i < 4:
                conv = CONV2D(
                    input_channels=input_dim,
                    output_channels=out_dim,
                    stride=2,
                    padding=1,
                    kernel_size=3,
                )
            elif i == 4:
                conv = CONV2D(
                    input_channels=out_dim,
                    output_channels=out_dim,
                    stride=2,
                    padding=1,
                    kernel_size=3,
                )
            else:
                raise ValueError(f'too many tdnn layers')
            self.conv_.append(conv)

        for i in range(self.num_layers):

            if i > 0:
                input_dim = self.hidden_size
                if self.bidirectional:
                    input_dim *= 2
            else:
                input_dim = self.n_features

            if i + 1 == self.num_layers:
                dropout = 0
            else:
                dropout = self.dropout
            if self.bidirectional:
                rnn = Klass(
                    input_dim,
                    self.hidden_size,
                    num_layers=1,
                    bias=self.bias,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=self.bidirectional,
                )
            else:
                rnn = Klass(
                    self.hidden_size,
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

        if self.num_layers < 1:

            if return_intermediate:
                msg = (
                    '"return_intermediate" must be set to False ' "when num_layers < 1"
                )
                raise ValueError(msg)

            output = features

        else:
            hidden = None
            output = features
            # apply each layer separately...
            for j in range(self.low_conv_num):
                output = self.conv_[j](output)

            for i, rnn in enumerate(self.rnn_):
                if i > self.num_conv - self.low_conv_num or i+self.low_conv_num > self.num_conv-1:
                    output, hidden = rnn(output, hidden)                   
                else:
                    output = self.conv_[i+self.low_conv_num](output)
                    output, hidden = rnn(output)

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

            if self.concatenate:
                dimension *= self.num_layers

            if self.pool == "x-vector":
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


class RNNP(nn.Module):
    def __init__(
        self,
        n_features,
        down_rate,
        unit="LSTM",
        hidden_size=16,
        num_layers=1,
        bias=True,
        dropout=0,
        bidirectional=False,
        concatenate=False,
        using_residual=False,
        pool=None,
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
        self.concatenate = concatenate
        self.pool = pool
        self.pool_ = TemporalPooling.create(pool) if pool is not None else None
        self.down_rate = down_rate
        self.using_residual = using_residual

        if num_layers < 1:
            msg = '"bidirectional" must be set to False when num_layers < 1'
            if bidirectional:
                raise ValueError(msg)
            msg = '"concatenate" must be set to False when num_layers < 1'
            if concatenate:
                raise ValueError(msg)
            return

        else:
            self.rnn_ = nn.ModuleList([])
            self.tdnn_ = nn.ModuleList([])
            self.max_pool1d_ = nn.ModuleList([])
            self.dropout_ = nn.ModuleList([])

           
            self.num_downsamp = int(math.log(self.down_rate)/math.log(2))

            if self.num_downsamp > 0:
                tdnn = TDNN(
                    context=[-2, 2],
                    input_channels=self.n_features,
                    output_channels=self.n_features,
                    full_context=True,
                    stride=2,
                    padding=2,
                )
                self.tdnn_.append(tdnn)
            for i in range(self.num_layers):

                if i == 0:
                    input_dim = self.n_features
                else:
                    input_dim = self.hidden_size
                    if self.bidirectional:
                        input_dim *= 2

                if i == self.num_layers - 1:
                    dropout = 0
                else:
                    dropout = self.dropout

                self.dropout_.append(nn.Dropout(dropout))
                if self.bidirectional:
                    rnn = Klass(
                        input_dim,
                        self.hidden_size,
                        num_layers=1,
                        bias=self.bias,
                        batch_first=True,
                        dropout=dropout,
                        bidirectional=self.bidirectional,
                    )
                    
                else:
                    rnn = Klass(
                        input_dim,
                        self.hidden_size,
                        num_layers=1,
                        bias=self.bias,
                        batch_first=True,
                        dropout=dropout,
                        bidirectional=self.bidirectional,
                    )

                self.rnn_.append(rnn)


                if i < self.num_layers-1:
                    max_pool1d = nn.MaxPool1d(3, stride=2, padding=1, dilation=1)
                    self.max_pool1d_.append(max_pool1d)
                

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

        if self.num_layers < 1:

            if return_intermediate:
                msg = (
                    '"return_intermediate" must be set to False ' "when num_layers < 1"
                )
                raise ValueError(msg)

            output = features

        else:
            hidden = None
            output = features
            # apply each layer separately...
            if self.num_downsamp > 0:
                output = self.tdnn_[0](output)

            for i, rnn in enumerate(self.rnn_):
                if i < self.num_downsamp-1 and i < self.num_layers-1:
                    output, hidden = rnn(output, hidden)
                    output = output.transpose(1,2)
                    output = self.max_pool1d_[i](output)
                    output = output.transpose(1,2)

                else:
                    output, hidden = rnn(output, hidden)
                
                output = self.dropout_[i](output)

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

            if self.concatenate:
                dimension *= self.num_layers

            if self.pool == "x-vector":
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



class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)

class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class CIF0(nn.Module):
    def __init__(
        self,
        conv_cif_num_filters=128,
        conv_cif_widths_string='1',
        produce_weights_type='conv',
        conv_cif_num_layers=1,
        dense_cif_units=128,
        cif_weight_threshold=1,
        dropout=0,
        print_timestamp=True,
        mode="layer_norm",
        conv_bias=False,
        leaky_rate=0.9,
        activation_diff='cos',
        norm_accumulate=True,
        scale=1.0,
        affine_cos=False,
        momentum_embed=0.0,
        integrate_mode='avg',
        context=0,
        b_conv_context=2,
        produce_diff_type='meandense',
        encoder_out_dim=256,
        norm_type=None,
        ):
        super().__init__()

        self.produce_weights_type = produce_weights_type
        self.conv_cif_num_filters = conv_cif_num_filters
        self.conv_cif_widths_string = conv_cif_widths_string
        self.conv_cif_num_layers = conv_cif_num_layers
        self.dense_cif_units = dense_cif_units
        self.cif_weight_threshold = cif_weight_threshold
        self.print_timestamp = print_timestamp
        # For visualizing attention heads.
        self.attention_weights = dict()
        self.leaky_rate = leaky_rate
        self.activation_diff = activation_diff
        self.norm_accumulate = norm_accumulate
        self.scale = scale
        self.affine_cos = affine_cos
        self.momentum_embed = momentum_embed
        self.integrate_mode = integrate_mode
        self.context = context
        self.norm_type = norm_type

        if self.produce_weights_type == 'conv':
            def block(
                n_in,
                n_out,
                conv_cif_width,
                stride,
                is_layer_norm=False,
                is_group_norm=False,
                conv_bias=False,
            ):
                def make_conv():
                    conv = nn.Conv1d(n_in, n_out, conv_cif_width, stride=1, bias=conv_bias)
                    nn.init.kaiming_normal_(conv.weight)
                    return conv
                assert(
                    is_layer_norm and is_group_norm
                ) == False, "layer norm adn group norm are exclusive"

                if is_layer_norm:
                    return nn.Sequential(
                        make_conv(),
                        nn.Dropout(p=dropout),
                        nn.Sequential(
                            TransposeLast(),
                            Fp32LayerNorm(dim, elementwise_affine=True),
                            TransposeLast(),
                        ),
                        nn.GELU(),
                    )
                elif is_group_norm:
                    return nn.Sequential(
                        make_conv(),
                        nn.Dropout(p=dropout),
                        Fp32GroupNorm(dim, dim, affine=True),
                        nn.GELU(),
                    )
                    
                else:
                    return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

            in_d = 256
            self.conv_cif_widths_list = self.conv_cif_widths_string.split(',')
            self.conv_layers = nn.ModuleList()
            for i in range(self.conv_cif_num_layers):
                dim = self.conv_cif_num_filters
                self.conv_layers.append(
                    block(
                        in_d,
                        dim,
                        int(self.conv_cif_widths_list[i]),
                        stride=1,
                        is_layer_norm=mode == "layer_norm",
                        is_group_norm=mode == "default" and i == 0,
                        conv_bias=conv_bias,
                    )
                )
                in_d = dim
        elif self.produce_weights_type == 'dense':
            self.cif_dense = nn.Linear(encoder_out_dim, self.dense_cif_units)

        #self.final_dense = nn.Linear(self.dense_cif_units, 1)
        self.final_dense = nn.Linear(encoder_out_dim, 1)
        if self.activation_diff == 'relutanh':
            self.cif_relu = torch.nn.ReLU()
            self.cif_tanh = torch.nn.Tanh()
        if self.affine_cos:
            self.dense_cos = nn.Linear(encoder_out_dim, encoder_out_dim)

        if self.norm_type == 'BN':
            self.cif_norm = nn.BatchNorm1d(
                encoder_out_dim, eps=1e-5, momentum=0.1, affine=False
            )

        elif self.norm_type == 'LN':
            self.cif_norm = LayerNorm(encoder_out_dim)
        else:
            pass
        # LM fusion part
        # (TODO)
    def forward(self, encoder_outputs, change_p, is_training):
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

        def produce_weights(diff, type):
            # if self.produce_weights_type == 'conv':
            #     # conv cif
            #     diff = diff.unsqueeze(1)
            #     a = diff.transpose(1,2)
            #     assert len(self.conv_cif_widths_list) == self.conv_cif_num_layers

            #     for conv in self.conv_layers:
            #         a = conv(a)
            #     a = a.transpose(1,2)

            # elif self.produce_weights_type == 'dense':
            #     # dense cif
            #     a = self.cif_dense(diff)
            # else:
            #     raise ValueError("No such weight-producing type!")
            
            a = self.final_dense(diff)
            #a = torch.sigmoid(a)
            if self.activation_diff == 'relutanh':
                a = self.cif_relu(a)
                a = self.cif_tanh(a)

            elif self.activation_diff == 'sigmoid':
                a = torch.sigmoid(a)
                #pdb.set_trace()

            else:
                raise Exception("wrong activation_diff") 
            #a = 0.5*(torch.tanh(a - 3) + 1)
            a = a.squeeze(-1)
            # mask = 1 - padding_mask.int()
            # a = a.mul(mask)
            
            return a

        def produce_a(diff): 
            a = self.final_dense(diff)
            #a = torch.sigmoid(a)
            if self.activation_diff == 'relutanh':
                a = self.cif_relu(a)
                a = self.cif_tanh(a)

            elif self.activation_diff == 'sigmoid':
                a = torch.sigmoid(a)

            else:
                raise Exception("wrong activation_diff") 

            a = a.squeeze(-1)
 
            return a


        # def process_change_p_t(change_p, is_fired, flag, t):
        #     #pdb.set_trace()
        #     change_p[:,t] = torch.where(flag==1, torch.zeros([batch_size]).int().cuda(), change_p[:, t].int())
        #     flag = torch.where((flag==0) & (is_fired==1) & (change_p[:,t]==1), torch.full([batch_size], 1).int().cuda(), flag)
        #     if t+1 < change_p.size(1):
        #         flag = torch.where((flag==1) & ~(change_p[:,t+1]==1),torch.zeros([batch_size]).int().cuda(),flag)

        #     return change_p, flag


        # if is_training and self.use_scaling_strategy:
        #     targets_mask = (targets != 0).float()
        #     targets_length = targets_mask.sum(-1)

        #     a_sum = a.sum(-1)
        #     normalize_scalar = (targets_length / a_sum).unsqueeze(-1)
        #     a_org = a
        #     a = a * normalize_scalar

        # used for the handling of tail
        #first_padding_pos = not_padding.sum(-1)
        first_padding_pos = (torch.sum(torch.abs(encoder_outputs), -1) != 0.0).int().sum(1)

        threshold = self.cif_weight_threshold
        batch_size = encoder_outputs.size(0)
        encoder_output_length = encoder_outputs.size(1)

        hidden_size = encoder_outputs.size(2)


        # accumulated_weights = torch.zeros([batch_size, 0]).cuda()
        # accumulated_states = torch.zeros([batch_size, 0, hidden_size]).cuda()
        # fired_states = torch.zeros([batch_size, 0, hidden_size]).cuda()

        accumulated_weights = torch.zeros([batch_size, 0]).cuda()
        accumulated_1_weights = torch.zeros([batch_size, 0]).cuda()
        accumulated_states = torch.zeros([batch_size, 0, hidden_size]).cuda()
        fired_states = torch.zeros([batch_size, 0, hidden_size]).cuda()
        fired_flag = torch.zeros([batch_size, 0, 1]).cuda()
        weight_keep = torch.zeros([batch_size, 0, 1]).cuda()
        sum_a = torch.zeros([batch_size]).cuda()
        integrate_steps = torch.zeros([batch_size, hidden_size]).cuda()

        flag = torch.zeros([batch_size]).int().cuda()
        if change_p != None:
            change_p = change_p.squeeze(-1)
        for t in range(encoder_output_length):
            if t == 0:
                # prev_accumulated_weight = torch.zeros([batch_size]).cuda()
                # prev_accumulated_state = torch.zeros([batch_size, hidden_size]).cuda()

                prev_accumulated_weight = torch.zeros([batch_size]).cuda()
                prev_accumulated_1_weight = torch.zeros([batch_size]).cuda()
                #prev_accumulated_state = torch.zeros([batch_size, hidden_size]).cuda()
                prev_accumulated_state = encoder_outputs[:, 0, :]

            else:
                prev_accumulated_weight = accumulated_weights[:, t-1]
                prev_accumulated_1_weight = accumulated_1_weights[:, t-1]
                prev_accumulated_state = accumulated_states[:, t-1, :]

            def cos_distance(v1, v2):
                return (1 - F.cosine_similarity(v1, v2, 1, 1e-8))/2
            def euc_distance(v1, v2):
                return torch.norm(v1 - v2)
            integrate_steps += 1
            #pdb.set_trace()
            #diff = cos_distance(prev_accumulated_state, encoder_outputs[:, t, :])
            #pdb.set_trace()
            context = self.context
            if context != 0:
                start = max(0, t - context)
                end = min(t+context, encoder_output_length)
                if self.activation_diff == 'cos':
                    if self.affine_cos:
                        diff = self.scale * cos_distance(prev_accumulated_state, self.dense_cos(torch.mean(encoder_outputs[:, start:end, :],dim=1)))
                    else:
                        diff = self.scale * cos_distance(prev_accumulated_state, torch.mean(encoder_outputs[:, start:end, :],dim=1))
                    a = diff
                else:
                    #diff = euc_distance(prev_accumulated_state, encoder_outputs[:, t, :])
                    diff = F.normalize(prev_accumulated_state, dim=-1) - F.normalize(encoder_outputs[:, t, :], dim=-1)
                    
                    a = produce_a(diff.unsqueeze(1))
            else:
                if self.activation_diff == 'cos':
                    if self.affine_cos:
                        diff = self.scale * cos_distance(prev_accumulated_state, self.dense_cos(encoder_outputs[:, t, :]))
                    else:
                        diff = self.scale * cos_distance(prev_accumulated_state, encoder_outputs[:, t, :])
                    a = diff
                else:
                    #diff = euc_distance(prev_accumulated_state, encoder_outputs[:, t, :])
                    diff = F.normalize(prev_accumulated_state, dim=-1) - F.normalize(encoder_outputs[:, t, :], dim=-1)

                    a = produce_a(diff.unsqueeze(1))
            a = a.squeeze(-1) + 1e-8

            a = torch.where(torch.full([batch_size], t).cuda() > first_padding_pos,
                                       torch.zeros([batch_size]).cuda(),
                                       a)
            weight_keep = torch.cat([weight_keep, a.unsqueeze(-1).unsqueeze(-1)], 1)

            cur_is_fired = self.leaky_rate * prev_accumulated_weight + a > threshold
            cur_is_fired = cur_is_fired.int()
            ## dell with change_p according to fired flag 
            #pdb.set_trace()
            if change_p != None:
                change_p[:,t] = torch.where(flag==1, torch.zeros([batch_size]).int().cuda(), change_p[:, t].int())
                flag = torch.where((flag==0) & (cur_is_fired==1) & (change_p[:,t]==1), torch.full([batch_size], 1).int().cuda(), flag)
                if t+1 < change_p.size(1):
                    flag = torch.where((flag==1) & (change_p[:,t+1]==0) ,torch.zeros([batch_size]).int().cuda(),flag)

            cur_weight = a
            remained_weight = 1.0 - self.leaky_rate * prev_accumulated_weight
            detal_sum_a = torch.where(cur_is_fired == 1,
                        self.leaky_rate * prev_accumulated_weight + remained_weight,
                        torch.zeros([batch_size]).cuda())
            detal_sum_a = torch.where(torch.full([batch_size], t).cuda() > first_padding_pos,
                                       torch.zeros([batch_size]).cuda(),
                                       detal_sum_a)
            sum_a += detal_sum_a
            cur_accumulated_weight = torch.where(cur_is_fired == 1,
                                                cur_weight - remained_weight,
                                                cur_weight + self.leaky_rate * prev_accumulated_weight)

            cur_accumulated_1_weight = torch.where(cur_is_fired == 1,
                                                1 - (cur_weight - remained_weight) + 1e-8,
                                                1 - cur_weight + prev_accumulated_1_weight + 1e-8)


            if self.integrate_mode == 'norm':
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                             (1 - (cur_weight - remained_weight)).unsqueeze(-1) * encoder_outputs[:, t, :],
                                             (prev_accumulated_1_weight.unsqueeze(-1) * prev_accumulated_state + (1 - cur_weight.unsqueeze(-1)) * encoder_outputs[:, t, :])/cur_accumulated_1_weight.unsqueeze(-1))

                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                        prev_accumulated_state,
                                       torch.zeros([batch_size, hidden_size]).cuda())

            elif self.integrate_mode == 'momentum':
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                             (cur_weight - remained_weight).unsqueeze(-1) * encoder_outputs[:, t, :],
                                             (1 - self.momentum_embed) * prev_accumulated_state + self.momentum_embed * (1 - cur_weight.unsqueeze(-1)) * encoder_outputs[:, t, :])

                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                            prev_accumulated_state,
                                          torch.zeros([batch_size, hidden_size]).cuda())


            elif self.integrate_mode == 'avg':  
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                             encoder_outputs[:, t, :],
                                             ((integrate_steps - 1) * prev_accumulated_state + encoder_outputs[:, t, :])/integrate_steps)

                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                            prev_accumulated_state,
                                          torch.zeros([batch_size, hidden_size]).cuda())                          

            else:
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                             (cur_weight - remained_weight).unsqueeze(-1) * encoder_outputs[:, t, :],
                                             prev_accumulated_state + (1 - cur_weight.unsqueeze(-1)) * encoder_outputs[:, t, :])

                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                            prev_accumulated_state,
                                          torch.zeros([batch_size, hidden_size]).cuda())


            integrate_steps = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size) == 1,
                                            torch.zeros([batch_size, hidden_size]).cuda(),
                                            integrate_steps)
            # handling the speech tail by rounding up and down

            cur_fired_state = torch.where(torch.full([batch_size, 1], t).cuda() > first_padding_pos.unsqueeze(-1),
                                       torch.zeros([batch_size, hidden_size]).cuda(),
                                       cur_fired_state)


            fired_flag = torch.cat([fired_flag, cur_is_fired.unsqueeze(-1).unsqueeze(-1)], 1)
            accumulated_weights = torch.cat([accumulated_weights, cur_accumulated_weight.unsqueeze(-1)], 1)
            accumulated_1_weights = torch.cat([accumulated_1_weights, cur_accumulated_1_weight.unsqueeze(-1)], 1)
            accumulated_states = torch.cat([accumulated_states, cur_accumulated_state.unsqueeze(1)], 1)
                                           
            fired_states = torch.cat([fired_states, cur_fired_state.unsqueeze(1)], 1)

        
        # tail fire

        tail_fired_states = cur_accumulated_state/(cur_accumulated_weight.unsqueeze(-1).repeat(1, hidden_size)+1e-05)
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


        if change_p != None:
            change_p = change_p.unsqueeze(-1)
        #pdb.set_trace()
        if self.norm_type == 'BN':
            cif_outputs = cif_outputs.transpose(2,1)
            cif_outputs = self.cif_norm(cif_outputs)
            cif_outputs = cif_outputs.transpose(2,1)
        elif self.norm_type == 'LN':
            cif_outputs = self.cif_norm(cif_outputs)
        else:
            pass
        return cif_outputs, weight_keep, fired_flag, not_padding_after_cif, sum_a, change_p





class CIF1(nn.Module):
    def __init__(
        self,
        conv_cif_num_filters=128,
        conv_cif_widths_string='1',
        produce_weights_type='conv',
        conv_cif_num_layers=1,
        dense_cif_units=1,
        cif_weight_threshold=1,
        dropout=0,
        print_timestamp=True,
        mode="layer_norm",
        conv_bias=False,
        leaky_rate=0.9,
        activation_diff='cos',
        norm_accumulate=True,
        scale=1.0,
        affine_cos=False,
        momentum_embed=0.0,
        integrate_mode='avg',
        context=0,
        b_conv_context=2,
        produce_diff_type='meandense',
        encoder_out_dim=256,
        norm_type=None,
        ):
        super().__init__()

        self.produce_weights_type = produce_weights_type
        self.conv_cif_num_filters = conv_cif_num_filters
        self.conv_cif_widths_string = conv_cif_widths_string
        self.conv_cif_num_layers = conv_cif_num_layers
        self.dense_cif_units = dense_cif_units
        self.cif_weight_threshold = cif_weight_threshold
        self.print_timestamp = print_timestamp
        # For visualizing attention heads.
        self.attention_weights = dict()
        self.leaky_rate = leaky_rate
        self.activation_diff = activation_diff
        self.norm_accumulate = norm_accumulate
        self.scale = scale
        self.affine_cos = affine_cos
        self.momentum_embed = momentum_embed
        self.integrate_mode = integrate_mode
        self.b_conv_context = b_conv_context
        self.produce_diff_type = produce_diff_type
        self.context = context
        self.norm_type = norm_type
        if self.produce_weights_type == 'conv':
            self.cif_conv = TDNN(
                    context=[-self.b_conv_context, self.b_conv_context],
                    input_channels=encoder_out_dim,
                    output_channels=1,
                    full_context=True,
                    stride=1,
                    padding=self.b_conv_context,
                )

 
        elif self.produce_weights_type == 'dense':
            self.cif_dense = nn.Linear(encoder_out_dim, self.dense_cif_units)

        #self.final_dense = nn.Linear(self.dense_cif_units, 1)
        self.final_dense = nn.Linear(encoder_out_dim, 1)
        if self.activation_diff == 'relutanh':
            self.cif_relu = torch.nn.ReLU()
            self.cif_tanh = torch.nn.Tanh()
        if self.affine_cos:
            self.dense_cos = nn.Linear(encoder_out_dim, encoder_out_dim)

            if self.produce_diff_type == 'conv':
                self.conv_cos = TDNN(
                    context=[-self.context, self.context],
                    input_channels=encoder_out_dim,
                    output_channels=encoder_out_dim,
                    full_context=True,
                    stride=1,
                    padding=self.context,
                )

        if self.norm_type == 'BN':
            self.cif_norm = nn.BatchNorm1d(
                encoder_out_dim, eps=1e-5, momentum=0.1, affine=False
            )

        elif self.norm_type == 'LN':
            self.cif_norm = LayerNorm(encoder_out_dim)
        else:
            pass
        # LM fusion part
        # (TODO)
    def forward(self, encoder_outputs, change_p, is_training):
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

        # 先计算生成每一个时间步的speaker的信息量，这个信息量是用来加权ht的，从而生成seg级别的说话人表示，这个权重b和speaker change点关系不大
        if self.produce_weights_type == 'conv':
            b = self.cif_conv(encoder_outputs)
        elif self.produce_weights_type == 'dense':
            b = self.cif_dense(encoder_outputs)
        else:
            raise ValueError("No such weight-producing type!")

        if self.produce_diff_type == 'conv':
            conv_ht = self.conv_cos(encoder_outputs)

        b = torch.sigmoid(b)

        def produce_a(diff): 
            a = self.final_dense(diff)
            #a = torch.sigmoid(a)
            if self.activation_diff == 'relutanh':
                a = self.cif_relu(a)
                a = self.cif_tanh(a)

            elif self.activation_diff == 'sigmoid':
                a = torch.sigmoid(a)

            else:
                raise Exception("wrong activation_diff") 

            a = a.squeeze(-1)
 
            return a


        # used for the handling of tail
        #first_padding_pos = not_padding.sum(-1)
        first_padding_pos = (torch.sum(torch.abs(encoder_outputs), -1) != 0.0).int().sum(1)

        threshold = self.cif_weight_threshold
        batch_size = encoder_outputs.size(0)
        encoder_output_length = encoder_outputs.size(1)

        hidden_size = encoder_outputs.size(2)


        accumulated_weights = torch.zeros([batch_size, 0]).cuda()
        accumulated_b = torch.zeros([batch_size, 0, 1]).cuda()
        accumulated_states = torch.zeros([batch_size, 0, hidden_size]).cuda()
        fired_states = torch.zeros([batch_size, 0, hidden_size]).cuda()
        fired_flag = torch.zeros([batch_size, 0, 1]).cuda()
        weight_keep = torch.zeros([batch_size, 0, 1]).cuda()
        sum_a = torch.zeros([batch_size]).cuda()
        integrate_steps = torch.zeros([batch_size, hidden_size]).cuda()

        flag = torch.zeros([batch_size]).int().cuda()
        if change_p != None:
            change_p = change_p.squeeze(-1)
        for t in range(encoder_output_length):
            bt = b[:,t,:]
            if t == 0:
                prev_accumulated_weight = torch.zeros([batch_size]).cuda()
                prev_accumulated_b = torch.zeros([batch_size, 1]).cuda()
                #prev_accumulated_state = torch.zeros([batch_size, hidden_size]).cuda()
                prev_accumulated_state = encoder_outputs[:, 0, :]

            else:
                prev_accumulated_weight = accumulated_weights[:, t-1]
                prev_accumulated_b = accumulated_b[:, t-1, :]
                prev_accumulated_state = accumulated_states[:, t-1, :]

            def cos_distance(v1, v2):
                tmp = F.cosine_similarity(v1, v2, 1, 1e-8)
                return (1 - tmp)/2

            def euc_distance(v1, v2):
                return torch.norm(v1 - v2)
            integrate_steps += 1
            context = self.context
            if context != 0:
                start = max(0, t - context)
                end = min(t+context, encoder_output_length)

                if self.activation_diff == 'cos':
                    if self.affine_cos:
                        if self.produce_diff_type == 'meandense':
                            diff = self.scale * cos_distance(prev_accumulated_state, self.dense_cos(torch.mean(encoder_outputs[:, start:end, :],dim=1)))
                        elif self.produce_diff_type == 'conv':
                            diff = self.scale * cos_distance(prev_accumulated_state, conv_ht[:,t,:])
                        else:
                            raise ValueError(f'wrong produce diff type')

                    else:
                        diff = self.scale * cos_distance(prev_accumulated_state, torch.mean(encoder_outputs[:, start:end, :],dim=1))
                        #print('max, min', torch.max(diff), torch.min(diff))
                    
                    a = diff
                else:
                    #diff = euc_distance(prev_accumulated_state, torch.mean(encoder_outputs[:, start:end, :],dim=1))
                    diff = F.normalize(prev_accumulated_state, dim=-1) - F.normalize(encoder_outputs[:, t, :], dim=-1)
                    a = produce_a(diff.unsqueeze(1))
            else:
                if self.activation_diff == 'cos':
                    if self.affine_cos:
                        diff = self.scale * cos_distance(prev_accumulated_state, self.dense_cos(encoder_outputs[:, t, :]))
                    else:
                        diff = self.scale * cos_distance(prev_accumulated_state, encoder_outputs[:, t, :])
                    a = diff
                else:
                    #diff = euc_distance(prev_accumulated_state, encoder_outputs[:, t, :])
                    diff = F.normalize(prev_accumulated_state, dim=-1) - F.normalize(encoder_outputs[:, t, :], dim=-1)
                    a = produce_a(diff.unsqueeze(1))

            a = a.squeeze(-1)

            a = torch.where(torch.full([batch_size], t).cuda() > first_padding_pos,
                                       torch.zeros([batch_size]).cuda(),
                                       a)
            weight_keep = torch.cat([weight_keep, a.unsqueeze(-1).unsqueeze(-1)], 1)

            cur_is_fired = self.leaky_rate * prev_accumulated_weight + a > threshold
            cur_is_fired = cur_is_fired.int()




            ## dell with change_p according to fired flag 
            if change_p != None:
                
                change_p[:,t] = torch.where(flag==1, torch.zeros([batch_size]).int().cuda(), change_p[:, t].int())
                flag = torch.where((flag==0) & (cur_is_fired==1) & (change_p[:,t]==1), torch.full([batch_size], 1).int().cuda(), flag)
                if t+1 < change_p.size(1):
                    flag = torch.where((flag==1) & (change_p[:,t+1]==0), torch.zeros([batch_size]).int().cuda(),flag)


            cur_weight = a
            remained_weight = 1.0 - self.leaky_rate * prev_accumulated_weight
            detal_sum_a = torch.where(cur_is_fired == 1,
                        self.leaky_rate * prev_accumulated_weight + remained_weight,
                        torch.zeros([batch_size]).cuda())
            detal_sum_a = torch.where(torch.full([batch_size], t).cuda() > first_padding_pos,
                                       torch.zeros([batch_size]).cuda(),
                                       detal_sum_a)
            sum_a += detal_sum_a
            cur_accumulated_weight = torch.where(cur_is_fired == 1,
                                                cur_weight - remained_weight,
                                                cur_weight + self.leaky_rate * prev_accumulated_weight)

            cur_accumulated_b = torch.where(cur_is_fired.unsqueeze(-1) == 1,
                                            bt,
                                            bt + prev_accumulated_b,
                                            )
            

            if self.integrate_mode == 'norm':
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                             bt * encoder_outputs[:, t, :],
                                             (prev_accumulated_b * prev_accumulated_state + bt * encoder_outputs[:, t, :])/cur_accumulated_b)

                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                        prev_accumulated_state,
                                       torch.zeros([batch_size, hidden_size]).cuda())

            elif self.integrate_mode == 'momentum':
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                             bt * encoder_outputs[:, t, :],
                                             (1 - self.momentum_embed) * prev_accumulated_state + self.momentum_embed * bt * encoder_outputs[:, t, :])

                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                            prev_accumulated_state,
                                          torch.zeros([batch_size, hidden_size]).cuda())


            elif self.integrate_mode == 'avg':  
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                             encoder_outputs[:, t, :],
                                             ((integrate_steps - 1) * prev_accumulated_state + encoder_outputs[:, t, :])/integrate_steps)

                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                             prev_accumulated_state,
                                          torch.zeros([batch_size, hidden_size]).cuda())                          

            else:
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                             bt * encoder_outputs[:, t, :],
                                             prev_accumulated_state + bt * encoder_outputs[:, t, :])

                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                            prev_accumulated_state,
                                          torch.zeros([batch_size, hidden_size]).cuda())


            integrate_steps = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size) == 1,
                                            torch.zeros([batch_size, hidden_size]).cuda(),
                                            integrate_steps)
            # handling the speech tail by rounding up and down

            cur_fired_state = torch.where(torch.full([batch_size, 1], t).cuda() > first_padding_pos.unsqueeze(-1),
                                       torch.zeros([batch_size, hidden_size]).cuda(),
                                       cur_fired_state)


            fired_flag = torch.cat([fired_flag, cur_is_fired.unsqueeze(-1).unsqueeze(-1)], 1)
            accumulated_weights = torch.cat([accumulated_weights, cur_accumulated_weight.unsqueeze(-1)], 1)
            accumulated_b = torch.cat([accumulated_b, cur_accumulated_b.unsqueeze(-1)], 1)
            accumulated_states = torch.cat([accumulated_states, cur_accumulated_state.unsqueeze(1)], 1)
                                           
            fired_states = torch.cat([fired_states, cur_fired_state.unsqueeze(1)], 1)

        # tail fire

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

        # for the calculation of num char
        # if is_training and self.use_scaling_strategy:
        #     sum_a = a_org.sum(1)
        # else:
        #     sum_a = a.sum(1)
        if change_p != None:
            change_p = change_p.unsqueeze(-1)
        #pdb.set_trace()
        if self.norm_type == 'BN':
            cif_outputs = cif_outputs.transpose(2,1)
            cif_outputs = self.cif_norm(cif_outputs)
            cif_outputs = cif_outputs.transpose(2,1)
        elif self.norm_type == 'LN':
            cif_outputs = self.cif_norm(cif_outputs)
        else:
            pass


        return cif_outputs, weight_keep, fired_flag, not_padding_after_cif, sum_a, change_p

class CIF2(nn.Module):
    def __init__(
        self,
        cif_weight_threshold=1,
        leaky_rate=0.9,
        encoder_out_dim=256,
        cif_output_norm_type='LN',
        max_history=10,
        weight_active='hardsigmoid',
        relu_threshold=1.0,
        using_L2norm=False,
        norm_et_mode=None,
        reset_et='ht',
        suma_mode='1',
        ):
        super().__init__()

        self.cif_weight_threshold = cif_weight_threshold

        self.attention_weights = dict()
        self.leaky_rate = leaky_rate
        self.cif_output_norm_type = cif_output_norm_type
        self.max_history = max_history
        self.weight_active = weight_active
        self.relu_threshold = relu_threshold
        self.using_L2norm = using_L2norm
        self.norm_et_mode = norm_et_mode
        self.reset_et = reset_et
        self.suma_mode = suma_mode

        self.weight_dense1 = nn.Linear(2 * encoder_out_dim, 2 * encoder_out_dim)
        self.weight_dense2 = nn.Linear(2 * encoder_out_dim, 1)

        if weight_active == 'crelu':
            self.crelu = torch.nn.ReLU()


        if self.cif_output_norm_type == 'BN':
            self.cif_norm = nn.BatchNorm1d(
                encoder_out_dim, eps=1e-5, momentum=0.1, affine=False
            )

        elif self.cif_output_norm_type == 'LN':
            self.cif_norm = LayerNorm(encoder_out_dim)
        else:
            pass


    def sequence_mask(self, lengths, max_len=None):
        lengths_shape = lengths.shape  
        lengths = lengths.reshape(-1)
        
        batch_size = lengths.numel()
        max_len = max_len or int(lengths.max())
        lengths_shape += (max_len,)
        
        return (torch.arange(0,max_len,device=lengths.device)
        .type_as(lengths)
        .unsqueeze(0).expand(batch_size,max_len)
        .lt(lengths.unsqueeze(1))).reshape(lengths_shape)

    def forward(self, encoder_outputs, is_training):
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

        if self.using_L2norm == True:
            encoder_outputs = F.normalize(encoder_outputs, p=2, dim=-1)

        first_padding_pos = (torch.sum(torch.abs(encoder_outputs), -1) != 0.0).int().sum(1)

        threshold = self.cif_weight_threshold
        batch_size = encoder_outputs.size(0)
        encoder_output_length = encoder_outputs.size(1)

        hidden_size = encoder_outputs.size(2)

        accumulated_1_weights = torch.zeros([batch_size, 0]).cuda()
        accumulated_weights = torch.zeros([batch_size, 0]).cuda()
        accumulated_states = torch.zeros([batch_size, 0, hidden_size]).cuda()
        fired_states = torch.zeros([batch_size, 0, hidden_size]).cuda()
        fired_flag = torch.zeros([batch_size, 0, 1]).cuda()
        weight_keep = torch.zeros([batch_size, 0, 1]).cuda()
        sum_a = torch.zeros([batch_size]).cuda()
        integrate_steps = torch.zeros([batch_size, hidden_size]).cuda()
        
        start_flag = torch.zeros([batch_size]).int().cuda()


        for t in range(encoder_output_length):

            if t == 0:
                prev_accumulated_weight = torch.zeros([batch_size]).cuda()
                prev_accumulated_state = encoder_outputs[:, 0, :]
                prev_accumulated_1_weight = torch.zeros([batch_size]).cuda()

            else:
                prev_accumulated_weight = accumulated_weights[:, t-1]
                prev_accumulated_state = accumulated_states[:, t-1, :]
                prev_accumulated_1_weight = accumulated_1_weights[:, t-1]

            integrate_steps +=1 
            history_chunk_len = torch.full([batch_size], t).cuda() - start_flag

            history_chunk = encoder_outputs[:, t, :]

            history_chunk = torch.where(history_chunk_len.unsqueeze(-1).repeat(1,hidden_size)==1,
                                    encoder_outputs[:, t-1, :], history_chunk)

            history_chunk_len = torch.clip(history_chunk_len, max=self.max_history)
            his_len = encoder_outputs[:, max(t-self.max_history,0):t, :].size(1)


            history_mask = self.sequence_mask(history_chunk_len, his_len).int()
            history_mask = torch.flip(history_mask, dims=[1])

            history_chunk_tmp = history_mask.unsqueeze(-1).repeat(1, 1, hidden_size)
            #pdb.set_trace()
            history_chunk_tmp = (encoder_outputs[:, max(t-self.max_history,0):t, :] * history_chunk_tmp).sum(1) \
                                    / (history_mask.sum(1).unsqueeze(-1).repeat(1, hidden_size) + 1e-8)

            history_chunk = torch.where(history_chunk_len.unsqueeze(-1).repeat(1, hidden_size)>1,
                                    history_chunk_tmp, history_chunk)
             
            if self.using_L2norm == True:
                diff = encoder_outputs[:, t, :] - F.normalize(history_chunk, p=2, dim=-1)
            else:
                diff = encoder_outputs[:, t, :] - history_chunk

            info = torch.cat([encoder_outputs[:, t, :], diff], -1)
            weight = self.weight_dense1(info)
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

            weight = torch.where(torch.full([batch_size], t).cuda() > first_padding_pos,
                                       torch.zeros([batch_size]).cuda(),
                                       weight)

            weight_keep = torch.cat([weight_keep, weight.unsqueeze(-1).unsqueeze(-1)], 1)

            cur_is_fired = self.leaky_rate * prev_accumulated_weight + weight > threshold
            cur_is_fired = cur_is_fired.int()
            remained_weight = threshold - self.leaky_rate * prev_accumulated_weight

            start_flag = torch.where(cur_is_fired==1, torch.full([batch_size], t).int().cuda(), start_flag)

            cur_weight = weight


            if self.suma_mode == '1':
                cur_accumulated_weight = torch.where(cur_is_fired == 1,
                                                torch.zeros([batch_size]).cuda(),
                                                cur_weight + self.leaky_rate * prev_accumulated_weight)
                cur_accumulated_1_weight = torch.where(cur_is_fired == 1,
                                                torch.ones([batch_size]).cuda(),
                                                1 - cur_weight + prev_accumulated_1_weight + 1e-8)
                detal_sum_a = torch.where(cur_is_fired == 1,
                            torch.ones([batch_size]).cuda(),
                            torch.zeros([batch_size]).cuda())

            elif self.suma_mode == 'full':
                cur_accumulated_weight = torch.where(cur_is_fired == 1,
                                                cur_weight - remained_weight,
                                                cur_weight + self.leaky_rate * prev_accumulated_weight)
                cur_accumulated_1_weight = torch.where(cur_is_fired == 1,
                                                1 - (cur_weight - remained_weight),
                                                1 - (cur_weight + self.leaky_rate * prev_accumulated_weight))
                detal_sum_a = torch.where(cur_is_fired == 1,
                            self.leaky_rate * prev_accumulated_weight + cur_weight,
                            torch.zeros([batch_size]).cuda())
            else:
                raise ValueError(f'wrong sum a mode')


            detal_sum_a = torch.where(torch.full([batch_size], t).cuda() > first_padding_pos,
                                       torch.zeros([batch_size]).cuda(),
                                       detal_sum_a)

            sum_a += detal_sum_a

            # 发放位置embedding重置成0 或者 ht
            if self.reset_et == 'ht':
                fired_et =  encoder_outputs[:, t, :]

            elif self.reset_et == '0':
                fired_et = torch.zeros([batch_size, hidden_size]).cuda()
            else:
                raise ValueError(f'error reset et setting')       


            if self.norm_et_mode == 'avg_norm':
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                             fired_et,
                                             ((integrate_steps - 1) * prev_accumulated_state + \
                                             (1-cur_weight.unsqueeze(-1)) * \
                                             encoder_outputs[:, t, :]) / \
                                             integrate_steps)

                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                             ((integrate_steps - 1) * prev_accumulated_state + \
                                             (1-cur_weight.unsqueeze(-1)) * \
                                             encoder_outputs[:, t, :]) / integrate_steps,
                                             torch.zeros([batch_size, hidden_size]).cuda()) 

            elif self.norm_et_mode == 'weight_norm':
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                             fired_et,
                                             (prev_accumulated_1_weight.unsqueeze(-1) * prev_accumulated_state + (1 - cur_weight.unsqueeze(-1)) * encoder_outputs[:, t, :])/cur_accumulated_1_weight.unsqueeze(-1))

                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                        prev_accumulated_state,
                                       torch.zeros([batch_size, hidden_size]).cuda())

            elif self.norm_et_mode == 'et_norm': 
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                            fired_et,
                                            cur_weight.unsqueeze(-1) * prev_accumulated_state + \
                                            (1-cur_weight.unsqueeze(-1))*encoder_outputs[:, t, :])

                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                        cur_weight.unsqueeze(-1) * prev_accumulated_state + (1-cur_weight.unsqueeze(-1))*encoder_outputs[:, t, :],
                                        torch.zeros([batch_size, hidden_size]).cuda())
            elif self.norm_et_mode == None:      
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                                    fired_et,
                                                    prev_accumulated_state + (1-cur_weight.unsqueeze(-1))*encoder_outputs[:, t, :])
                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                            prev_accumulated_state + (1-cur_weight.unsqueeze(-1))*encoder_outputs[:, t, :],
                                            torch.zeros([batch_size, hidden_size]).cuda())   
 
            else:
                raise ValueError(f'error norm et mode')                   

            integrate_steps = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size) == 1,
                                            torch.zeros([batch_size, hidden_size]).cuda(),
                                            integrate_steps)

            # handling the speech tail by rounding up and down
            cur_fired_state = torch.where(torch.full([batch_size, 1], t).cuda() > first_padding_pos.unsqueeze(-1),
                                            torch.zeros([batch_size, hidden_size]).cuda(),
                                            cur_fired_state) 

            fired_flag = torch.cat([fired_flag, cur_is_fired.unsqueeze(-1).unsqueeze(-1)], 1)
            accumulated_weights = torch.cat([accumulated_weights, cur_accumulated_weight.unsqueeze(-1)], 1)
            accumulated_states = torch.cat([accumulated_states, cur_accumulated_state.unsqueeze(1)], 1)                     
            fired_states = torch.cat([fired_states, cur_fired_state.unsqueeze(1)], 1)
            accumulated_1_weights = torch.cat([accumulated_1_weights, cur_accumulated_1_weight.unsqueeze(-1)], 1)
        # tail fire
        #pdb.set_trace()

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

        if self.cif_output_norm_type == 'BN':
            cif_outputs = cif_outputs.transpose(2,1)
            cif_outputs = self.cif_norm(cif_outputs)
            cif_outputs = cif_outputs.transpose(2,1)
        elif self.cif_output_norm_type == 'LN':
            cif_outputs = self.cif_norm(cif_outputs)
        elif self.cif_output_norm_type == 'L2':
            cif_outputs = F.normalize(cif_outputs, p=2, dim=-1)
        else:
            pass

        return cif_outputs, weight_keep, fired_flag, not_padding_after_cif, sum_a



class CIF3(nn.Module):
    def __init__(
        self,
        cif_weight_threshold=1,
        encoder_out_dim=256,
        cif_output_norm_type='LN',
        max_history=10,
        weight_active='hardsigmoid',
        relu_threshold=1.0,
        using_L2norm=False,
        norm_et_mode=None,
        reset_et='ht',
        suma_mode='1',
        using_scaling=True,
        nonlinear_act=None,
        using_kaiming_init=False,
        using_zero_order=True,
        using_bias_constant_init=False,
        history_chunk_mode='avg',
        diff_mode='-',
        diff_conv_context=1,
        using_two_order=False,
        using_std_deviation=None,
        dt_pow=1,
        add_noise=False,
        noise_am=0.1,
        aux_change=False,
        scale_1_weight_position=None,
        var_norm_with_history=False,
        using_gt_change=False,
        scaling_b=False,
        future_len=0,
        diff_tdnn_norm_type=None,
        integrate_emb_without_L2=False,
        conv_one_order=False,
        force_no_fire=False,
        normalize_scalar=1.0,
        spk_embedding_weight='1_dt',
        ):
        super().__init__()
        self.cif_weight_threshold = cif_weight_threshold
        self.attention_weights = dict()
        self.cif_output_norm_type = cif_output_norm_type
        self.max_history = max_history
        self.weight_active = weight_active
        self.relu_threshold = relu_threshold
        self.using_L2norm = using_L2norm
        self.norm_et_mode = norm_et_mode
        self.reset_et = reset_et
        self.suma_mode = suma_mode
        self.using_scaling = using_scaling
        self.nonlinear_act = nonlinear_act
        self.using_zero_order = using_zero_order
        self.history_chunk_mode = history_chunk_mode
        self.diff_mode = diff_mode
        self.using_two_order = using_two_order
        self.using_std_deviation = using_std_deviation
        self.dt_pow = dt_pow
        self.add_noise = add_noise
        self.noise_am = noise_am
        self.aux_change = aux_change
        self.scale_1_weight_position = scale_1_weight_position
        self.var_norm_with_history = var_norm_with_history
        self.using_gt_change = using_gt_change
        self.scaling_b = scaling_b
        self.diff_conv_context = diff_conv_context
        self.future_len = future_len
        self.integrate_emb_without_L2 = integrate_emb_without_L2
        self.conv_one_order = conv_one_order
        self.force_no_fire = force_no_fire
        self.normalize_scalar = normalize_scalar
        self.spk_embedding_weight = spk_embedding_weight

        if self.history_chunk_mode == 'avg_context_chunk':
            n = 2
        else:
            n = 1

        n = int(using_zero_order) + int(using_two_order) + n

        if using_std_deviation != None:
            n += 1
        
        self.weight_dense1 = nn.Linear(n * encoder_out_dim, n * encoder_out_dim)
        self.weight_dense2 = nn.Linear(n * encoder_out_dim, 1)

        if aux_change == True:
            self.aux_change_dense = nn.Linear(n * encoder_out_dim, 2)
            self.aux_change_softmax = nn.Softmax(dim=-1)

        if spk_embedding_weight == 'l0+2fc+scalar':
            self.spk_embedding_weight_dense1 = nn.Linear(n * encoder_out_dim, n * encoder_out_dim)
            self.spk_embedding_weight_relu = torch.nn.ReLU()
            self.spk_embedding_weight_dense2 = nn.Linear(n * encoder_out_dim, 1)
        elif spk_embedding_weight == 'l0+2fc+vec':
            self.spk_embedding_weight_dense1 = nn.Linear(n * encoder_out_dim, n * encoder_out_dim)
            self.spk_embedding_weight_relu = torch.nn.ReLU()
            self.spk_embedding_weight_dense2 = nn.Linear(n * encoder_out_dim, encoder_out_dim)
        elif spk_embedding_weight == 'l1+1fc+scalar':
            self.spk_embedding_weight_dense1 = nn.Linear(n * encoder_out_dim, 1)
        elif spk_embedding_weight == 'l1+1fc+vec':
            self.spk_embedding_weight_dense1 = nn.Linear(n * encoder_out_dim, encoder_out_dim)
        elif spk_embedding_weight == '1_dt':
            pass
        else:
            raise ValueError(f'wrong spk embedding weight method')

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

        if self.cif_output_norm_type == 'BN':
            self.cif_norm = nn.BatchNorm1d(
                encoder_out_dim, eps=1e-5, momentum=0.1, affine=False
            )

        elif self.cif_output_norm_type == 'LN':
            self.cif_norm = LayerNorm(encoder_out_dim)
        else:
            pass
        
        if self.norm_et_mode == 'forget':
            self.forget_weight = nn.Linear(2 * encoder_out_dim, 1)

        if self.norm_et_mode == 'var_norm':
            if self.var_norm_with_history:
                self.var_weight = nn.Linear(2 * encoder_out_dim, 1)
            else:
                self.var_weight = nn.Linear(encoder_out_dim, 1)

        if self.norm_et_mode == 'nn':
            self.differ_nn_layer = nn.Linear(encoder_out_dim + 1, 1)

        if self.history_chunk_mode == 'tdnn':
            wid = self.max_history // 2
            if wid != 0:
                self.history_tdnn = TDNN(
                        context=[-wid, wid],
                        input_channels=encoder_out_dim,
                        output_channels=encoder_out_dim,
                        full_context=True,
                        stride=1,
                        padding=int(2*wid),
                    )  
            else:
                self.history_tdnn = TDNN(
                        context=[0],
                        input_channels=encoder_out_dim,
                        output_channels=encoder_out_dim,
                        full_context=True,
                        stride=1,
                        padding=0,
                    )  

        if self.diff_mode == 'conv':
            self.diff_conv = TDNN(
                        context=[-self.diff_conv_context, self.diff_conv_context],
                        input_channels=encoder_out_dim,
                        output_channels=encoder_out_dim,
                        full_context=True,
                        stride=1,
                        padding=self.diff_conv_context,
                    )
            if self.conv_one_order:
                self.one_order_conv = TDNN(
                        context=[-self.diff_conv_context, self.diff_conv_context],
                        input_channels=encoder_out_dim,
                        output_channels=encoder_out_dim,
                        full_context=True,
                        stride=1,
                        padding=self.diff_conv_context,
                )
                self.diff_dense = nn.Linear(2*encoder_out_dim, 1)
            else:
                self.diff_dense = nn.Linear(encoder_out_dim, 1)
        
    def compute_std_deviation(self, inputs):
        shape = inputs.size()
        if len(shape) == 2:
            inputs = inputs.unsqueeze(1)
        elif len(shape) == 3:
            pass
        else:
            raise ValueError('bad input shape {0}'.format(shape))
        mean = torch.mean(inputs, 1)
        square = torch.square(inputs)
        std = torch.sqrt(torch.clip(torch.mean(square, 1)-mean*mean, min=1e-8))
        return std

    def get_avg_context_chunk(self, inputs):
        batch_size = inputs.size(0)
        hidden_size = inputs.size(-1)
        history_chunk_keep = torch.zeros(batch_size, 0 , hidden_size).cuda()
        future_chunk_keep = torch.zeros(batch_size, 0 , hidden_size).cuda()

        for t in range(inputs.size(1)):
            if t == 0:
                end = min(inputs.size(1), t+self.max_history+1)
                history_chunk = inputs[:, 0, :]
                future_chunk = torch.mean(inputs[:, t+1:end, :], 1)
            elif t == inputs.size(1) - 1:
                start = max(0, t-self.max_history)
                history_chunk = torch.mean(inputs[:, start:t, :], 1)
                future_chunk = inputs[:, t, :]
            else:
                start = max(0, t-self.max_history)
                end = min(inputs.size(1), t+self.max_history+1)
                history_chunk = torch.mean(inputs[:, start:t, :], 1)
                future_chunk = torch.mean(inputs[:, t+1:end, :], 1)
         
            history_chunk_keep = torch.cat([history_chunk_keep, history_chunk.unsqueeze(1)], 1)
            future_chunk_keep = torch.cat([future_chunk_keep, future_chunk.unsqueeze(1)], 1)
        
        return history_chunk_keep, future_chunk_keep

    def forward(self, encoder_outputs, mask, is_training, change_p=None):
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
        if not self.using_gt_change:
            change_p = None
        encoder_outputs_ori = encoder_outputs
        if self.using_L2norm == True:
            encoder_outputs = self.normalize_scalar * F.normalize(encoder_outputs, p=2, dim=-1)
        first_padding_pos = (torch.sum(torch.abs(encoder_outputs), -1) != 0.0).int().sum(1)

        threshold = self.cif_weight_threshold
        batch_size = encoder_outputs.size(0)
        encoder_output_length = encoder_outputs.size(1)
        hidden_size = encoder_outputs.size(2)

        accumulated_weights = torch.zeros([batch_size, 0]).cuda()
        accumulated_states = torch.zeros([batch_size, 0, hidden_size]).cuda()
        fired_states = torch.zeros([batch_size, 0, hidden_size]).cuda()
        fired_flag = torch.zeros([batch_size, 0, 1]).cuda()
        start_flag = torch.zeros([batch_size]).int().cuda()
        
        if self.history_chunk_mode == 'avg':
            history_chunk_keep = torch.zeros(batch_size, 0 , hidden_size).cuda()
            history_std_keep = torch.zeros(batch_size, 0 , hidden_size).cuda()
            if self.future_len != 0:
                ht_with_future_keep = torch.zeros(batch_size, 0 , hidden_size).cuda()

            for t in range(encoder_output_length):
                if t == 0:
                    history_chunk = encoder_outputs[:, 0, :]
                    if self.using_std_deviation == 'std1':
                        history_std = self.compute_std_deviation(history_chunk)
                    elif self.using_std_deviation == 'std2':
                        history_std = torch.zeros_like(history_chunk).cuda()
                    elif self.using_std_deviation == None:
                        pass
                    else:
                        raise ValueError(f'wrong std deviation')
                else:
                    start = max(0, t-self.max_history)
                    history_chunk = torch.mean(encoder_outputs[:, start:t, :], 1)
                    if self.using_std_deviation == 'std1':
                        history_std = self.compute_std_deviation(encoder_outputs[:, start:t+1, :])
                    elif self.using_std_deviation == 'std2':
                        history_std = torch.std(encoder_outputs[:, start:t+1, :], 1)
                    elif self.using_std_deviation == None:
                        pass
                    else:
                        raise ValueError(f'wrong std deviation')
                
                # fuse future information
                future_end = min(t + self.future_len, encoder_output_length)
                ht_with_future = torch.mean(encoder_outputs[:, t:future_end, :], 1)
                if self.future_len != 0:
                    ht_with_future_keep = torch.cat([ht_with_future_keep, ht_with_future.unsqueeze(1)], 1)

                history_chunk_keep = torch.cat([history_chunk_keep, history_chunk.unsqueeze(1)], 1)
                if self.using_std_deviation:
                    history_std_keep = torch.cat([history_std_keep, history_std.unsqueeze(1)], 1)
        elif self.history_chunk_mode == 'tdnn':
            history_chunk_keep = self.history_tdnn(encoder_outputs)
            history_chunk_keep = history_chunk_keep[:,:encoder_output_length-1,:]
            history_chunk_keep = torch.cat([encoder_outputs[:,0,:].unsqueeze(1), history_chunk_keep], 1)
        elif self.history_chunk_mode == 'avg_context_chunk':
            history_chunk_keep, future_chunk_keep = self.get_avg_context_chunk(encoder_outputs)
        else:
            raise ValueError(f'wrong history chunk mode')

        #计算差异量
        if self.diff_mode == '-':
            if self.future_len != 0:
                ht = ht_with_future_keep
            else:
                ht = encoder_outputs

            if self.using_L2norm == True:
                his_diff = ht - self.normalize_scalar * F.normalize(history_chunk_keep, p=2, dim=-1)
                if self.history_chunk_mode == 'avg_context_chunk':
                    fut_diff = ht - self.normalize_scalar * F.normalize(future_chunk_keep, p=2, dim=-1)
            else:
                his_diff = ht - history_chunk_keep
                if self.history_chunk_mode == 'avg_context_chunk':
                    fut_diff = ht - future_chunk_keep

            info = his_diff
            if self.using_zero_order == True:
                info = torch.cat([encoder_outputs, info], -1)

            if self.history_chunk_mode == 'avg_context_chunk':
                info = torch.cat([fut_diff, info], -1)

            if self.using_two_order == True:
                tmp_diff = diff[:,1:,:]
                tmp_diff = torch.cat([tmp_diff, diff[:,-1,:].unsqueeze(1)], 1)
                diff_diff = tmp_diff - diff
                info = torch.cat([info, diff_diff], -1)

            if self.using_std_deviation != None:
                info = torch.cat([info, history_std_keep], -1)
            
            weight = self.weight_dense1(info)

            if self.nonlinear_act != None:
                weight = self.diff_act(weight)
                
            if self.aux_change == True:
                aux_change = self.aux_change_dense(weight)
                aux_change = self.aux_change_softmax(aux_change)
            else:
                aux_change = None

            ## diff info used to weight spk embedding
            if self.spk_embedding_weight == 'l0+2fc+scalar' or self.spk_embedding_weight == 'l0+2fc+vec':
                info2spk_intergrate = self.spk_embedding_weight_dense1(info)
                info2spk_intergrate = self.spk_embedding_weight_relu(info2spk_intergrate)
                info2spk_intergrate = self.spk_embedding_weight_dense2(info2spk_intergrate)
                info2spk_intergrate = torch.sigmoid(info2spk_intergrate)
            elif self.spk_embedding_weight == 'l1+1fc+scalar' or self.spk_embedding_weight == 'l1+1fc+vec':
                info2spk_intergrate = self.spk_embedding_weight_dense1(info)
                info2spk_intergrate = torch.sigmoid(info2spk_intergrate)
            elif self.spk_embedding_weight == '1_dt':
                pass
            else:
                raise ValueError(f'wrong spk embedding weight')
            
            weight = self.weight_dense2(weight)

            if self.weight_active == 'hardsigmoid':
                weight = torch.nn.functional.hardsigmoid(weight)
            elif self.weight_active == 'sigmoid':
                weight = torch.sigmoid(weight)
            elif self.weight_active == 'crelu':
                weight = self.crelu(weight)
                weight = torch.clip(weight, max=self.relu_threshold)
            elif self.weight_active == 'leakyrelu':
                weight_c = self.leakyrelu(weight)
                weight = torch.clip(weight_c, min=0, max=self.relu_threshold)
            elif self.weight_active == 'leakyrelu_nc':
                weight_c = self.leakyrelu(weight)
                weight = torch.clip(weight_c, min=0)
            else:
                pass
            weight = weight.squeeze(-1)

        elif self.diff_mode == 'cos':
            weight = F.cosine_similarity(history_chunk_keep, encoder_outputs, -1, 1e-8)
            weight = (1 - weight)/2

        elif self.diff_mode == 'conv':
            aux_change = None
            weight_hid = self.diff_conv(encoder_outputs)
            if self.conv_one_order:
                one_order = encoder_outputs[:,1:,:] - encoder_outputs[:,:-1,:]
                zero_pad = torch.zeros([encoder_outputs.size(0), 1, encoder_outputs.size(-1)]).cuda()
                one_order = torch.cat([zero_pad, one_order], 1)
                one_order_diff = self.one_order_conv(one_order)
                weight_hid = torch.cat([weight_hid, one_order_diff], -1)

            weight = self.diff_dense(weight_hid)

            if self.weight_active == 'hardsigmoid':
                weight = torch.nn.functional.hardsigmoid(weight)
            elif self.weight_active == 'sigmoid':
                weight = torch.sigmoid(weight)
            elif self.weight_active == 'crelu':
                weight = self.crelu(weight)
                weight = torch.clip(weight, max=self.relu_threshold)
            elif self.weight_active == 'leakyrelu':
                weight_c = self.leakyrelu(weight)
                weight = torch.clip(weight_c, min=0, max=self.relu_threshold)
            elif self.weight_active == 'leakyrelu_nc':
                weight_c = self.leakyrelu(weight)
                weight = torch.clip(weight_c, min=0)
            else:
                pass
            weight = weight.squeeze(-1)

        else:
            raise ValueError(f'wrong diff mode')

        if self.norm_et_mode == 'nn':
            b = torch.cat([encoder_outputs, weight.unsqueeze(-1)], -1)
            b = self.differ_nn_layer(b)
            b = torch.sigmoid(b)
            b = b.squeeze(-1)
            if is_training and self.scaling_b:
                sum_b = b.sum(1)
                N = mask[:,:,0].sum(-1)-1
                scale_b = (N+1)/torch.clip(sum_b, min=1e-8)
                scale_b = torch.where(sum_b == 0,
                                torch.zeros_like(scale_b),
                                scale_b)
                b = scale_b.unsqueeze(-1).repeat(1, encoder_output_length) * b

        if is_training and self.add_noise:
            weight = weight + self.noise_am*torch.randn(weight.size()).cuda()

        # keep original weight
        ori_weight = weight
        w1_weight = 1 - weight

        if is_training and self.using_scaling and mask != None:
            sum_a = weight.sum(1)
            N = mask[:,:,0].sum(-1) - 1
            scale = N/torch.clip(sum_a, min=1e-8)
            scale = torch.where(sum_a == 0,
                                torch.zeros_like(sum_a),
                                scale)
            #print(N, sum_a, scale)
            weight = scale.unsqueeze(-1).repeat(1, encoder_output_length) * weight

            if self.scale_1_weight_position == 'after':
                w1_weight = 1 - weight
                sum_1_weight = w1_weight.sum(1)
                scale_1_weight = (N+1)/torch.clip(sum_1_weight, min=1e-8)
                scale_1_weight = torch.where(sum_1_weight == 0,
                                torch.zeros_like(sum_1_weight),
                                scale_1_weight)

                w1_weight = scale_1_weight.unsqueeze(-1).repeat(1, encoder_output_length) * w1_weight
            elif self.scale_1_weight_position == 'before':
                w1_weight = 1 - ori_weight
                sum_1_weight = w1_weight.sum(1)
                scale_1_weight = (N+1)/torch.clip(sum_1_weight, min=1e-8)
                scale_1_weight = torch.where(sum_1_weight == 0,
                                torch.zeros_like(sum_1_weight),
                                scale_1_weight)

                w1_weight = scale_1_weight.unsqueeze(-1).repeat(1, encoder_output_length) * w1_weight
            elif self.scale_1_weight_position == None:
                pass
            else:
                raise ValueError(f'wrong scale 1 - weight position')
        else:
            sum_a = weight.sum(1)

        if change_p != None:
            weight = change_p.squeeze(-1)
            w1_weight = 1 - weight

        if self.integrate_emb_without_L2:
            encoder_outputs = encoder_outputs_ori

        if self.force_no_fire:
            weight = torch.zeros([batch_size, encoder_output_length]).cuda().float()
            w1_weight = 1 - weight
            ori_weight = weight
            sum_a = weight.sum(1)

        for t in range(encoder_output_length):
            if t == 0:
                prev_accumulated_weight = torch.zeros([batch_size]).cuda()
                prev_accumulated_state = encoder_outputs[:, 0, :]
            else:
                prev_accumulated_weight = accumulated_weights[:, t-1]
                prev_accumulated_state = accumulated_states[:, t-1, :]
 
            cur_weight = weight[:,t]
            cur_1_weight = w1_weight[:,t]

            cur_weight = torch.where(torch.full([batch_size], t).cuda() > first_padding_pos,
                                       torch.zeros([batch_size]).cuda(),
                                       cur_weight)

            cur_is_fired = prev_accumulated_weight + cur_weight > threshold
            cur_is_fired = cur_is_fired.int()
            remained_weight = threshold - prev_accumulated_weight

            start_flag = torch.where(cur_is_fired==1, torch.full([batch_size], t).int().cuda(), start_flag)
            if self.suma_mode == '1':
                cur_accumulated_weight = torch.where(cur_is_fired == 1,
                                                torch.zeros([batch_size]).cuda(),
                                                cur_weight + prev_accumulated_weight)

            elif self.suma_mode == 'full':
                cur_accumulated_weight = torch.where(cur_is_fired == 1,
                                                cur_weight - remained_weight,
                                                cur_weight + prev_accumulated_weight)

            else:
                raise ValueError(f'wrong sum a mode')

            cur_weight = torch.pow(cur_weight+1e-8, self.dt_pow)

            # 发放位置embedding重置成0 或者 ht
            if self.reset_et == 'ht':
                fired_et = encoder_outputs[:, t, :]
            elif self.reset_et == '0':
                fired_et = torch.zeros([batch_size, hidden_size]).cuda()
            elif self.reset_et == '1_w_ht':
                if self.norm_et_mode == 'weight_norm':
                    fired_et = cur_1_weight.unsqueeze(-1)*encoder_outputs[:, t, :]
                elif  self.norm_et_mode == 'nn':
                    fired_et = b[:,t].unsqueeze(-1)*encoder_outputs[:, t, :]
                else:
                    fired_et = (1-cur_weight.unsqueeze(-1))*encoder_outputs[:, t, :]
            else:
                raise ValueError(f'error reset et setting')       

            integrate_steps = torch.full([batch_size], t).cuda() - start_flag + 1
            integrate_steps = integrate_steps.unsqueeze(-1)
            if self.norm_et_mode == 'avg_norm':
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                             fired_et,
                                             ((integrate_steps - 1) * prev_accumulated_state + \
                                             (1-cur_weight.unsqueeze(-1)) * \
                                             encoder_outputs[:, t, :]) / \
                                             integrate_steps)

                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                             ((integrate_steps - 1) * prev_accumulated_state + \
                                             (1-cur_weight.unsqueeze(-1)) * \
                                             encoder_outputs[:, t, :]) / integrate_steps,
                                             torch.zeros([batch_size, hidden_size]).cuda()) 

            elif self.norm_et_mode == 'weight_norm':
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                             fired_et,
                                             prev_accumulated_state + cur_1_weight.unsqueeze(-1) * encoder_outputs[:, t, :])

                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                        prev_accumulated_state + cur_1_weight.unsqueeze(-1) * encoder_outputs[:, t, :],
                                       torch.zeros([batch_size, hidden_size]).cuda())

            elif self.norm_et_mode == 'et_norm': 
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                            fired_et,
                                            cur_weight.unsqueeze(-1) * prev_accumulated_state + \
                                            (1-cur_weight.unsqueeze(-1))*encoder_outputs[:, t, :])

                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                        cur_weight.unsqueeze(-1) * prev_accumulated_state + (1-cur_weight.unsqueeze(-1))*encoder_outputs[:, t, :],
                                        torch.zeros([batch_size, hidden_size]).cuda())

            elif self.norm_et_mode == 'forget':
                forget = torch.sigmoid(self.forget_weight(torch.cat([encoder_outputs[:,t,:], prev_accumulated_state], -1)))
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                                    fired_et,
                                                    forget * prev_accumulated_state + (1-cur_weight.unsqueeze(-1))*encoder_outputs[:, t, :])
                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                            forget * prev_accumulated_state + (1-cur_weight.unsqueeze(-1))*encoder_outputs[:, t, :],
                                            torch.zeros([batch_size, hidden_size]).cuda())

            elif self.norm_et_mode == 'var_norm':
                if self.var_norm_with_history:
                    wb = torch.sigmoid(self.var_weight(torch.cat([encoder_outputs[:,t,:], prev_accumulated_state], -1)))
                else:
                    wb = torch.sigmoid(self.var_weight(encoder_outputs[:,t,:]))

                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                                    fired_et,
                                                    prev_accumulated_state + (wb-cur_weight.unsqueeze(-1))*encoder_outputs[:, t, :])
                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                            prev_accumulated_state + (wb-cur_weight.unsqueeze(-1))*encoder_outputs[:, t, :],
                                            torch.zeros([batch_size, hidden_size]).cuda())  
            elif self.norm_et_mode == 'nn':
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                                    fired_et,
                                                    prev_accumulated_state + b[:,t].unsqueeze(-1)*encoder_outputs[:, t, :])
                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                            prev_accumulated_state + b[:,t].unsqueeze(-1)*encoder_outputs[:, t, :],
                                            torch.zeros([batch_size, hidden_size]).cuda())   
            elif self.norm_et_mode == '1':
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                                    fired_et,
                                                    prev_accumulated_state + encoder_outputs[:, t, :])
                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                            prev_accumulated_state + encoder_outputs[:, t, :],
                                            torch.zeros([batch_size, hidden_size]).cuda())  
            elif self.norm_et_mode == 'dis_fired_frame':
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                                    torch.zeros([batch_size, hidden_size]).cuda(),
                                                    prev_accumulated_state + (1-cur_weight.unsqueeze(-1))*encoder_outputs[:, t, :])
                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                            prev_accumulated_state,
                                            torch.zeros([batch_size, hidden_size]).cuda())   
            elif self.norm_et_mode == 'learnable_weight':
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                                    fired_et,
                                                    prev_accumulated_state + info2spk_intergrate[:,t,:]*encoder_outputs[:, t, :])
                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                            prev_accumulated_state + info2spk_intergrate[:,t,:]*encoder_outputs[:, t, :],
                                            torch.zeros([batch_size, hidden_size]).cuda())  
            elif self.norm_et_mode == None:      
                cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                                    fired_et,
                                                    prev_accumulated_state + (1-cur_weight.unsqueeze(-1))*encoder_outputs[:, t, :])
                cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                            prev_accumulated_state + (1-cur_weight.unsqueeze(-1))*encoder_outputs[:, t, :],
                                            torch.zeros([batch_size, hidden_size]).cuda()) 
            else:
                raise ValueError(f'error norm et mode')                   

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

        if self.cif_output_norm_type == 'BN':
            cif_outputs = cif_outputs.transpose(2,1)
            cif_outputs = self.cif_norm(cif_outputs)
            cif_outputs = cif_outputs.transpose(2,1)
        elif self.cif_output_norm_type == 'LN':
            cif_outputs = self.cif_norm(cif_outputs)
        elif self.cif_output_norm_type == 'L2':
            cif_outputs = self.normalize_scalar * F.normalize(cif_outputs, p=2, dim=-1)
        else:
            pass
        #pdb.set_trace()
        if self.weight_active == 'leakyrelu':
            return cif_outputs, weight_c, fired_flag, not_padding_after_cif, sum_a, aux_change
        else:
            return cif_outputs, ori_weight, fired_flag, not_padding_after_cif, sum_a, aux_change


class CIF4(nn.Module):
    def __init__(
        self,
        cif_weight_threshold=1,
        encoder_out_dim=256,
        cif_output_norm_type='LN',
        max_history=10,
        weight_active='hardsigmoid',
        relu_threshold=1.0,
        using_L2norm=False,
        norm_et_mode=None,
        reset_et='ht',
        suma_mode='1',
        using_scaling=True,
        nonlinear_act=None,
        using_kaiming_init=False,
        using_zero_order=True,
        using_bias_constant_init=False,
        history_chunk_mode='avg',
        diff_mode='-',
        diff_conv_context=1,
        using_two_order=False,
        using_std_deviation=None,
        dt_pow=1,
        add_noise=False,
        noise_am=0.1,
        aux_change=False,
        scale_1_weight_position=None,
        var_norm_with_history=False,
        using_gt_change=False,
        scaling_b=False,
        future_len=0,
        diff_tdnn_norm_type='wn',
        ):
        super().__init__()

        self.cif_weight_threshold = cif_weight_threshold

        self.attention_weights = dict()
        self.cif_output_norm_type = cif_output_norm_type
        self.max_history = max_history
        self.weight_active = weight_active
        self.relu_threshold = relu_threshold
        self.using_L2norm = using_L2norm
        self.norm_et_mode = norm_et_mode
        self.reset_et = reset_et
        self.suma_mode = suma_mode
        self.using_scaling = using_scaling
        self.nonlinear_act = nonlinear_act
        self.using_zero_order = using_zero_order
        self.history_chunk_mode = history_chunk_mode
        self.diff_mode = diff_mode
        self.using_two_order = using_two_order
        self.using_std_deviation = using_std_deviation
        self.dt_pow = dt_pow
        self.add_noise = add_noise
        self.noise_am = noise_am
        self.aux_change = aux_change
        self.scale_1_weight_position = scale_1_weight_position
        self.var_norm_with_history = var_norm_with_history
        self.using_gt_change = using_gt_change
        self.diff_conv_context = diff_conv_context
        self.diff_tdnn_norm_type = diff_tdnn_norm_type

        self.diff_conv = TDNN(
                    context=[-self.diff_conv_context, self.diff_conv_context],
                    input_channels=encoder_out_dim,
                    output_channels=encoder_out_dim,
                    full_context=True,
                    stride=1,
                    padding=self.diff_conv_context,
                    norm_type=diff_tdnn_norm_type,
        )

        self.diff_dense = nn.Linear(encoder_out_dim, 1)
              
    def forward(self, encoder_outputs, mask, is_training, change_p=None):
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
        if not self.using_gt_change:
            change_p = None

        first_padding_pos = (torch.sum(torch.abs(encoder_outputs), -1) != 0.0).int().sum(1)

        threshold = self.cif_weight_threshold
        batch_size = encoder_outputs.size(0)
        encoder_output_length = encoder_outputs.size(1)
        hidden_size = encoder_outputs.size(2)

        accumulated_weights = torch.zeros([batch_size, 0]).cuda()
        accumulated_1_weights = torch.zeros([batch_size, 0]).cuda()
        accumulated_states = torch.zeros([batch_size, 0, hidden_size]).cuda()
        fired_states = torch.zeros([batch_size, 0, hidden_size]).cuda()
        fired_flag = torch.zeros([batch_size, 0, 1]).cuda()
        
        aux_change = None
            
        weight = self.diff_conv(encoder_outputs)
        weight = self.diff_dense(weight)

        weight = torch.sigmoid(weight)
        weight = weight.squeeze(-1)

        # keep original weight
        ori_weight = weight

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

        if change_p != None:
            weight = change_p.squeeze(-1)
            w1_weight = 1 - weight

        for t in range(encoder_output_length):
            if t == 0:
                prev_accumulated_weight = torch.zeros([batch_size]).cuda()
                prev_accumulated_state = encoder_outputs[:, 0, :]
                prev_accumulated_1_weight = torch.zeros([batch_size]).cuda()
            else:
                prev_accumulated_weight = accumulated_weights[:, t-1]
                prev_accumulated_state = accumulated_states[:, t-1, :]
                prev_accumulated_1_weight = accumulated_1_weights[:, t-1]
 
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
      
            cur_accumulated_1_weight = torch.where(cur_is_fired == 1,
                                                1 - (cur_weight - remained_weight) + 1e-8,
                                                1 - cur_weight + prev_accumulated_1_weight + 1e-8)
            norm_1_weight = 1 - cur_weight + prev_accumulated_1_weight + 1e-8
            # 直接在发放的位置 weight norm 一下
            cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                            (1-(cur_weight - remained_weight)).unsqueeze(-1)*encoder_outputs[:, t, :],
                                            prev_accumulated_state + (1 - cur_weight.unsqueeze(-1)) * encoder_outputs[:, t, :])
            #pdb.set_trace()
            cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                    (prev_accumulated_state + (1 - (cur_weight - remained_weight).unsqueeze(-1)) * encoder_outputs[:, t, :])/norm_1_weight.unsqueeze(-1),
                                    torch.zeros([batch_size, hidden_size]).cuda())
                
            # handling the speech tail by rounding up and down
            cur_fired_state = torch.where(torch.full([batch_size, 1], t).cuda() > first_padding_pos.unsqueeze(-1),
                                            torch.zeros([batch_size, hidden_size]).cuda(),
                                            cur_fired_state) 

            fired_flag = torch.cat([fired_flag, cur_is_fired.unsqueeze(-1).unsqueeze(-1)], 1)
            accumulated_weights = torch.cat([accumulated_weights, cur_accumulated_weight.unsqueeze(-1)], 1)
            accumulated_1_weights = torch.cat([accumulated_1_weights, cur_accumulated_1_weight.unsqueeze(-1)], 1)
            accumulated_states = torch.cat([accumulated_states, cur_accumulated_state.unsqueeze(1)], 1)                     
            fired_states = torch.cat([fired_states, cur_fired_state.unsqueeze(1)], 1)

        tail_fired_states = cur_accumulated_state/cur_accumulated_1_weight.unsqueeze(-1)
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

        return cif_outputs, ori_weight, fired_flag, not_padding_after_cif, sum_a, aux_change



class FF(nn.Module):
    """Feedforward layers

    Parameters
    ----------
    n_features : `int`
        Input dimension.
    hidden_size : `list` of `int`, optional
        Linear layers hidden dimensions. Defaults to [16, ].
    """

    def __init__(self, n_features, hidden_size=[16,]):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size

        self.linear_ = nn.ModuleList([])
        for hidden_size in self.hidden_size:
            linear = nn.Linear(n_features, hidden_size, bias=True)
            self.linear_.append(linear)
            n_features = hidden_size

    def forward(self, features):
        """

        Parameters
        ----------
        features : `torch.Tensor`
            (batch_size, n_samples, n_features) or (batch_size, n_features)

        Returns
        -------
        output : `torch.Tensor`
            (batch_size, n_samples, hidden_size[-1]) or (batch_size, hidden_size[-1])
        """

        output = features
        for linear in self.linear_:
            output = linear(output)
            output = torch.tanh(output)
        return output

    def dimension():
        doc = "Output dimension."

        def fget(self):
            if self.hidden_size:
                return self.hidden_size[-1]
            return self.n_features

        return locals()

    dimension = property(**dimension())


class Embedding(nn.Module):
    """Embedding

    Parameters
    ----------
    n_features : `int`
        Input dimension.
    batch_normalize : `boolean`, optional
        Apply batch normalization. This is more or less equivalent to
        embedding whitening.
    scale : {"fixed", "logistic"}, optional
        Scaling method. Defaults to no scaling.
    unit_normalize : deprecated in favor of 'scale'
    """

    def __init__(
        self,
        n_features: int,
        batch_normalize: bool = False,
        scale: Text = None,
        unit_normalize: bool = False,
    ):
        super().__init__()

        self.n_features = n_features

        self.batch_normalize = batch_normalize
        if self.batch_normalize:
            self.batch_normalize_ = nn.BatchNorm1d(
                n_features, eps=1e-5, momentum=0.1, affine=False
            )

        self.scale = scale
        self.scaling = Scaling(n_features, method=scale)

        if unit_normalize is True:
            msg = f"'unit_normalize' has been deprecated in favor of 'scale'."
            raise ValueError(msg)

    def forward(self, embedding):

        if self.batch_normalize:
            embedding = self.batch_normalize_(embedding)

        return self.scaling(embedding)

    @property
    def dimension(self):
        """Output dimension."""
        return self.n_features


class PyanNet3(Model):
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
    embedding : `dict`, optional
        Embedding parameters. Defaults to `Embedding` default parameters. This
        only has effect when model is used for representation learning.
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
        encoder: Optional[dict] = None,
        decoder: Optional[dict] = None,
        sincnet: Optional[dict] = None,
        conv2d: Optional[dict] = None,
        rnn: Optional[dict] = None,
        sa: Optional[dict] = None,
        ff: Optional[dict] = None,
        cif: Optional[dict] = None,
        loss_cfg: Optional[dict] = None,
        embedding: Optional[dict] = None,
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
        ff : `dict`, optional
            Feed-forward layers parameters. Defaults to `FF` default parameters.
        embedding : `dict`, optional
            Embedding parameters. Defaults to `Embedding` default parameters. This
            only has effect when model is used for representation learning.
        """

        if sincnet is None:
            sincnet = dict()
        if conv2d is None:
            conv2d = dict()
        self.sincnet = sincnet
        self.conv2d = conv2d
        self.loss_cfg = loss_cfg
        self.down_rate = loss_cfg["down_rate"]
        self.down_rate_mode = loss_cfg["down_rate_mode"]
        self.tdnn_context = loss_cfg["tdnn_context"]
        self.encoder = encoder
        self.decoder = decoder
        self.sa = sa
        self.rnn = rnn
        self.training = training
        self.pretraining = loss_cfg["pretraining"]
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

        self.rnn_ = TDNN4RNN(n_features, self.down_rate, **rnn)
        n_features = self.rnn_.dimension

        cif['encoder_out_dim'] = n_features
        self.cif = cif
        if loss_cfg['cif_type'] == 'cif0':
            self.cif_ = CIF0(**cif)
        elif loss_cfg['cif_type'] == 'cif1':
            self.cif_ = CIF1(**cif)
        elif loss_cfg['cif_type'] == 'cif2':
            self.cif_ = CIF2(**cif)
        elif loss_cfg['cif_type'] == 'cif3':
            self.cif_ = CIF3(**cif)
        elif loss_cfg['cif_type'] == 'cif4':
            self.cif_ = CIF4(**cif)
        else:
            raise ValueError(f'wrong cif type')

        if self.encoder['struction'] == 'rnn':
            if self.rnn['bidirectional'] == True:
                hidden_size = self.rnn['hidden_size'] * 2
            else:
                hidden_size = self.rnn['hidden_size']
        else:
            hidden_size = self.sa['encoder_embed_dim']

        #self.linear_ = nn.Linear(n_features, len(self.classes), bias=True)
        if self.decoder['struction'] == 'fc':
            self.dec_linear = nn.Linear(hidden_size, self.rnn['hidden_size'], bias=True)
            if decoder.get('extra_fc', False):
                self.dec_linear_extra = nn.Linear(hidden_size, hidden_size, bias=True)
                self.activation_hid_extra = torch.nn.ReLU()

        elif self.decoder['struction'] == 'tdnn':
            self.dec_tdnn = TDNN(
                    context=[-self.decoder['tdnn_context'], self.decoder['tdnn_context']],
                    input_channels=hidden_size,
                    output_channels=self.rnn['hidden_size'],
                    full_context=True,
                    stride=1,
                    padding=self.decoder['tdnn_context'],
                )     

        elif self.decoder['struction'] == 'rnn':
            Klass = getattr(nn, self.rnn_.unit)
            self.dec_rnn = Klass(
                hidden_size,
                self.rnn['hidden_size'],
                num_layers=1,
                bias=self.rnn_.bias,
                batch_first=True,
                dropout=self.rnn_.dropout,
                bidirectional=self.rnn_.bidirectional,
            )
        elif self.decoder['struction'] == 'sa':
            dec_sa = self.sa
            dec_sa['encoder_layers'] = 1
            self.dec_sa = SA(hidden_size, **dec_sa)
        elif self.decoder['struction'] == 'none':
            pass
        else:
            raise ValueError(f'error decoder struction')

        if self.decoder['struction'] != 'none':
            self.activation_hid = torch.nn.ReLU()

        if self.loss_cfg["spk_loss_type"] == 'bce' or self.loss_cfg["spk_loss_type"] == 'focal_bce':
            num_class = loss_cfg["num_spk_class"]
            self.activation_ = torch.nn.Sigmoid()
        elif self.loss_cfg["spk_loss_type"] == 'softmax':
            num_class = 141
            self.activation_ = torch.nn.LogSoftmax(dim=-1)
        else:
            raise ValueError(f'wrong spk loss type')


        if self.decoder['struction'] == 'rnn' or self.decoder['struction'] == 'none':
            self.linear_ = nn.Linear(hidden_size, num_class, bias=True)
        else:
            self.linear_ = nn.Linear(self.rnn['hidden_size'], num_class, bias=True)
        #self.activation_ = self.task.default_activation
        
        if self.loss_cfg["using_DPCL"]:
            self.spk_linear = nn.Linear(hidden_size, hidden_size, bias=True)
        if self.loss_cfg["using_frame_spk"] or self.pretraining:
            self.spk_linear2 = nn.Linear(hidden_size, hidden_size, bias=True)
            self.spk_activation = torch.nn.ReLU()
            self.spk_linear3 = nn.Linear(hidden_size, num_class, bias=True)
            self.activation_2 = torch.nn.Sigmoid()

        log_var = torch.zeros(self.loss_cfg['num_loss']).cuda()
        self.log_var = torch.nn.Parameter(log_var)


    def forward(self, waveforms, change_p=None, mask=None, return_intermediate=None):
        """Forward pass

        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1) `torch.Tensor`
            Batch of waveforms. In case SincNet is skipped, a tensor with shape
            (batch_size, n_samples, n_features) is expected.
        return_intermediate : `int`, optional
            Index of RNN layer. Returns RNN intermediate hidden state.
            Defaults to only return the final output.

        Returns
        -------
        output : `torch.Tensor`
            Final network output.
        intermediate : `torch.Tensor`
            Intermediate network output (only when `return_intermediate`
            is provided).
        """
        #waveforms.size()  [64, 32000, 1]
        if self.sincnet.get("skip", False):
            output = waveforms
        else:
            output = self.sincnet_(waveforms)
        #output.size()  [64, 1174, 60]

        output = self.rnn_(output)
        #speaker probability for frame-level speaker classification
        if self.loss_cfg["using_frame_spk"] or self.pretraining:
            rep = self.spk_linear2(output)
            rep = self.spk_activation(rep)
            prob = self.activation_2(self.spk_linear3(rep))
        else: 
            prob = None
        #speaker embedding for DPCL
        if not self.pretraining:
            if self.loss_cfg["using_DPCL"]:
                speaker_embedding = F.normalize(torch.tanh(self.spk_linear(output)), p=2, dim=-1)
            else:
                speaker_embedding = None
            if self.loss_cfg['cif_type'] == 'cif0' or self.loss_cfg['cif_type'] == 'cif1':
                output, weight_keep, fired_flag, not_padding_after_cif, sum_a, change_p = self.cif_(output, change_p, self.training)
            elif self.loss_cfg['cif_type'] == 'cif2':
                output, weight_keep, fired_flag, not_padding_after_cif, sum_a = self.cif_(output, self.training)
            elif self.loss_cfg['cif_type'] == 'cif3' or self.loss_cfg['cif_type'] == 'cif4':
                output, weight_keep, fired_flag, not_padding_after_cif, sum_a, aux_change = self.cif_(output, mask, self.training, change_p)
                weight_keep = weight_keep.unsqueeze(-1)

            cif_output = output

            if self.loss_cfg["using_ff"]:
                output = self.ff_(output)

            #output.size()   [64, 1174, 128]
            if self.task.is_representation_learning:
                return self.embedding_(output)

            if self.decoder['struction'] == 'fc':
                if self.decoder.get('extra_fc', False):
                    output = self.dec_linear_extra(output)
                    output = self.activation_hid_extra(output)
                output = self.dec_linear(output)
            elif self.decoder['struction'] == 'tdnn':
                output = self.dec_tdnn(output)
            elif self.decoder['struction'] == 'rnn':
                output, hidden = self.dec_rnn(output)
            elif self.decoder['struction'] == 'sa':
                output = self.dec_sa(output, self.training)
            elif self.decoder['struction'] == 'none':
                pass
            else:
                raise ValueError(f'error decoder struction')
            
            if self.decoder['struction'] != 'none':
                output = self.activation_hid(output)

            # length norm
            if self.decoder.get('length_norm', False):
                output = self.normalize_scalar * F.normalize(output)

            output = self.linear_(output)
            #output.size()   [64, 1174, 2]
            output = self.activation_(output)

            fired_flag_ = 1 - fired_flag
            #print(fired_flag.sum())
            #if return_intermediate is None:
            return cif_output, output, not_padding_after_cif, sum_a, weight_keep, torch.cat([fired_flag_, fired_flag], -1), speaker_embedding, self.loss_cfg, prob, change_p, self.log_var, aux_change
            #return output, intermediate
        else:
            return prob

    @property
    def dimension(self):
        if self.task.is_representation_learning:
            return self.embedding_.dimension

        return Model.dimension.fget(self)

    def intermediate_dimension(self, layer):
        if layer == 0:
            return self.sincnet_.dimension
        return self.rnn_.intermediate_dimension(layer - 1)


class PyanNet(Model):
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
    embedding : `dict`, optional
        Embedding parameters. Defaults to `Embedding` default parameters. This
        only has effect when model is used for representation learning.
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
        encoder: Optional[dict] = None,
        decoder: Optional[dict] = None,
        sincnet: Optional[dict] = None,
        conv2d: Optional[dict] = None,
        rnn: Optional[dict] = None,
        sa: Optional[dict] = None,
        ff: Optional[dict] = None,
        cif: Optional[dict] = None,
        loss_cfg: Optional[dict] = None,
        embedding: Optional[dict] = None,
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
        ff : `dict`, optional
            Feed-forward layers parameters. Defaults to `FF` default parameters.
        embedding : `dict`, optional
            Embedding parameters. Defaults to `Embedding` default parameters. This
            only has effect when model is used for representation learning.
        """

        if sincnet is None:
            sincnet = dict()
        if conv2d is None:
            conv2d = dict()
        self.sincnet = sincnet
        self.conv2d = conv2d
        self.loss_cfg = loss_cfg
        self.down_rate = loss_cfg["down_rate"]
        self.down_rate_mode = loss_cfg["down_rate_mode"]
        self.tdnn_context = loss_cfg["tdnn_context"]
        self.encoder = encoder
        self.decoder = decoder
        self.sa = sa
        self.rnn = rnn
        self.training = training
        self.pretraining = loss_cfg["pretraining"]
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

        # if not conv2d.get("skip", True):
        #     logger.info('skip 2d conv front end')
        #     self.conv2d_ = 
        #     n_features = 


        if self.down_rate != 1:
            if self.down_rate_mode == 'tdnn8':
                #for i in range(nun_tdnn_layers):
                frame1 = TDNN(
                    context=[-self.tdnn_context, self.tdnn_context],
                    input_channels=n_features,
                    output_channels=rnn["hidden_size"],
                    full_context=True,
                    stride=2,
                    padding=self.tdnn_context,
                )

                frame2 = TDNN(
                    context=[-self.tdnn_context, self.tdnn_context],
                    input_channels=rnn["hidden_size"],
                    output_channels=rnn["hidden_size"],
                    full_context=True,
                    stride=2,
                    padding=self.tdnn_context,
                )
                frame3 = TDNN(
                    context=[-self.tdnn_context, self.tdnn_context],
                    input_channels=rnn["hidden_size"],
                    output_channels=rnn["hidden_size"],
                    full_context=True,
                    stride=2,
                    padding=self.tdnn_context,
                )    

                frame4 = TDNN(
                    context=[-self.tdnn_context, self.tdnn_context],
                    input_channels=rnn["hidden_size"],
                    output_channels=rnn["hidden_size"],
                    full_context=True,
                    stride=2,
                    padding=self.tdnn_context,
                )   
                frame5 = TDNN(
                    context=[-self.tdnn_context, self.tdnn_context],
                    input_channels=rnn["hidden_size"],
                    output_channels=rnn["hidden_size"],
                    full_context=True,
                    stride=2,
                    padding=1,
                )              
                #self.tdnn_.append(frame)

                #self.tdnn_.append(StatsPool())
                if self.down_rate == 32:
                    self.tdnn = nn.Sequential(frame1, frame2, frame3, frame4, frame5)
                elif self.down_rate == 16:
                    self.tdnn = nn.Sequential(frame1, frame2, frame3, frame4)
                elif self.down_rate == 8:
                    self.tdnn = nn.Sequential(frame1, frame2, frame3)
                elif self.down_rate == 4:
                    self.tdnn = nn.Sequential(frame1, frame2)
                elif  self.down_rate == 2:
                    self.tdnn = nn.Sequential(frame1)
                else:
                    msg = (f"error down rate")
                    raise ValueError(msg)


                if self.encoder['struction'] == 'rnn':
                    n_features = rnn["hidden_size"]
                    if self.rnn is None:
                        self.rnn = dict()

                    self.rnn_ = RNN(n_features, **rnn)
                    n_features = self.rnn_.dimension

                elif self.encoder['struction'] == 'sa':
                    n_features = sa["encoder_embed_dim"]
                    if self.sa is None:
                        self.sa = dict()

                    self.sa_ = SA(n_features, **sa)
                    n_features = self.sa_.dimension

                else:
                    msg = (f"struction error")
                    raise ValueError(msg)


            elif self.down_rate_mode == 'trtrt':
                if self.encoder['struction'] == 'rnn':

                    if self.rnn is None:
                        self.rnn = dict()

                    self.rnn_ = TDNNRNN(n_features, self.down_rate, **rnn)
                    n_features = self.rnn_.dimension

                elif self.encoder['struction'] == 'sa':
                    if self.sa is None:
                        self.sa = dict()

                    self.sa_ = TDNNSA(n_features, self.down_rate, **sa)
                    n_features = self.sa_.dimension()


                else:
                    msg = (f"struction error")
                    raise ValueError(msg)                    

            elif self.down_rate_mode == 'crcrc':
                if self.rnn is None:
                    self.rnn = dict()

                self.rnn_ = CONV2DRNN(n_features, self.down_rate, **rnn)
                n_features = self.rnn_.dimension

            elif self.down_rate_mode == 'concat':
                n_features = self.down_rate * self.n_features
                if self.encoder['struction'] == 'rnn':
                    if self.rnn is None:
                        self.rnn = dict()
                    self.rnn_ = RNN(n_features, **rnn)
                    n_features = self.rnn_.dimension

                elif self.encoder['struction'] == 'sa':
                    if self.sa is None:
                        self.sa = dict()
                    self.sa_ = SA(n_features, **sa)
                    n_features = self.sa_.dimension

                else:
                    msg = (f"struction error")
                    raise ValueError(msg)
            elif self.down_rate_mode == 'max_pooling':
                if self.rnn is None:
                    self.rnn = dict()

                self.rnn_ = RNNP(n_features, self.down_rate, **rnn)
                n_features = self.rnn_.dimension

            else:
                raise ValueError(f'error down rate mode')


        else:
            if self.encoder['struction'] == 'rnn':
                if self.rnn is None:
                    self.rnn = dict()

                self.rnn_ = RNN(n_features, **rnn)
                n_features = self.rnn_.dimension

            elif self.encoder['struction'] == 'sa':
                if self.sa is None:
                    self.sa = dict()

                self.sa_ = SA(n_features, **sa)
                n_features = self.sa_.dimension
            else:
                msg = (f"struction error")
                raise ValueError(msg)
            
        cif['encoder_out_dim'] = n_features
        self.cif = cif
        if loss_cfg['cif_type'] == 'cif0':
            self.cif_ = CIF0(**cif)
        elif loss_cfg['cif_type'] == 'cif1':
            self.cif_ = CIF1(**cif)
        elif loss_cfg['cif_type'] == 'cif2':
            self.cif_ = CIF2(**cif)
        elif loss_cfg['cif_type'] == 'cif3':
            self.cif_ = CIF3(**cif)
        elif loss_cfg['cif_type'] == 'cif4':
            self.cif_ = CIF4(**cif)
        else:
            raise ValueError(f'wrong cif type')

        if ff is None:
            ff = dict()
        self.ff = ff
        if self.loss_cfg["using_ff"]:
            self.ff_ = FF(n_features, **ff)
            n_features = self.ff_.dimension

        if self.task.is_representation_learning:
            if embedding is None:
                embedding = dict()
            self.embedding = embedding
            self.embedding_ = Embedding(n_features, **embedding)
            return


        if self.encoder['struction'] == 'rnn':
            if self.rnn['bidirectional'] == True:
                hidden_size = self.rnn['hidden_size'] * 2
            else:
                hidden_size = self.rnn['hidden_size']
        else:
            hidden_size = self.sa['encoder_embed_dim']

        #self.linear_ = nn.Linear(n_features, len(self.classes), bias=True)
        if self.decoder['struction'] == 'fc':
            self.dec_linear = nn.Linear(hidden_size, self.rnn['hidden_size'], bias=True)
            if decoder.get('extra_fc', False):
                self.dec_linear_extra = nn.Linear(hidden_size, hidden_size, bias=True)
                self.activation_hid_extra = torch.nn.ReLU()

        elif self.decoder['struction'] == 'tdnn':
            self.dec_tdnn = TDNN(
                    context=[-self.decoder['tdnn_context'], self.decoder['tdnn_context']],
                    input_channels=hidden_size,
                    output_channels=self.rnn['hidden_size'],
                    full_context=True,
                    stride=1,
                    padding=self.decoder['tdnn_context'],
                )     

        elif self.decoder['struction'] == 'rnn':
            Klass = getattr(nn, self.rnn_.unit)
            self.dec_rnn = Klass(
                hidden_size,
                self.rnn['hidden_size'],
                num_layers=1,
                bias=self.rnn_.bias,
                batch_first=True,
                dropout=self.rnn_.dropout,
                bidirectional=self.rnn_.bidirectional,
            )
        elif self.decoder['struction'] == 'sa':
            dec_sa = self.sa
            dec_sa['encoder_layers'] = 1
            self.dec_sa = SA(hidden_size, **dec_sa)
        elif self.decoder['struction'] == 'none':
            pass
        else:
            raise ValueError(f'error decoder struction')

        if self.decoder['struction'] != 'none':
            self.activation_hid = torch.nn.ReLU()

        if self.loss_cfg["spk_loss_type"] == 'bce' or self.loss_cfg["spk_loss_type"] == 'focal_bce':
            num_class = loss_cfg["num_spk_class"]
            self.activation_ = torch.nn.Sigmoid()
        elif self.loss_cfg["spk_loss_type"] == 'softmax':
            num_class = 141
            self.activation_ = torch.nn.LogSoftmax(dim=-1)
        else:
            raise ValueError(f'wrong spk loss type')


        if self.decoder['struction'] == 'rnn' or self.decoder['struction'] == 'none':
            self.linear_ = nn.Linear(hidden_size, num_class, bias=True)
        else:
            self.linear_ = nn.Linear(self.rnn['hidden_size'], num_class, bias=True)
        #self.activation_ = self.task.default_activation
        
        if self.loss_cfg["using_DPCL"]:
            self.spk_linear = nn.Linear(hidden_size, hidden_size, bias=True)
        if self.loss_cfg["using_frame_spk"] or self.pretraining:
            self.spk_linear2 = nn.Linear(hidden_size, hidden_size, bias=True)
            self.spk_activation = torch.nn.ReLU()
            self.spk_linear3 = nn.Linear(hidden_size, num_class, bias=True)
            self.activation_2 = torch.nn.Sigmoid()

        log_var = torch.zeros(self.loss_cfg['num_loss']).cuda()
        self.log_var = torch.nn.Parameter(log_var)


    def forward(self, waveforms, change_p=None, mask=None, return_intermediate=None):
        """Forward pass

        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1) `torch.Tensor`
            Batch of waveforms. In case SincNet is skipped, a tensor with shape
            (batch_size, n_samples, n_features) is expected.
        return_intermediate : `int`, optional
            Index of RNN layer. Returns RNN intermediate hidden state.
            Defaults to only return the final output.

        Returns
        -------
        output : `torch.Tensor`
            Final network output.
        intermediate : `torch.Tensor`
            Intermediate network output (only when `return_intermediate`
            is provided).
        """
        #waveforms.size()  [64, 32000, 1]
        if self.sincnet.get("skip", False):
            output = waveforms
        else:
            output = self.sincnet_(waveforms)
        #output.size()  [64, 1174, 60]

        if self.down_rate == 1:
            if self.encoder['struction'] == 'rnn':
                if return_intermediate is None:
                    output = self.rnn_(output)
                else:
                    if return_intermediate == 0:
                        intermediate = output
                        output = self.rnn_(output)
                    else:
                        return_intermediate -= 1
                        # get RNN final AND intermediate outputs
                        output, intermediate = self.rnn_(output, return_intermediate=True)
                        # only keep hidden state of requested layer
                        intermediate = intermediate[return_intermediate]
                #output.size()   [64, 1174, 256]
            else:
                output = self.sa_(output, self.training)

        else:
            if self.down_rate_mode == 'concat':
                if output.size(1) % self.down_rate != 0:
                    n = output.size(1) // self.down_rate
                    output = output[:, :n*self.down_rate, :]
                output = torch.reshape(output, (output.size(0), output.size(1)//self.down_rate, self.down_rate, output.size(2)))
                output = torch.reshape(output, (output.size(0), output.size(1), self.down_rate*output.size(-1)))

            elif self.down_rate_mode == 'tdnn8':
                output = self.tdnn(output)
            elif self.down_rate_mode == 'trtrt':
                if self.encoder['struction'] == 'rnn':
                    output = self.rnn_(output)
                elif self.encoder['struction'] == 'sa':
                    output = self.sa_(output, self.training)
            elif self.down_rate_mode == 'crcrc':
                output = self.rnn_(output)
            elif self.down_rate_mode == 'max_pooling':
                output = self.rnn_(output)
            else:
                raise ValueError(f"error down rate mode")

 
            if self.down_rate_mode != 'trtrt' and self.down_rate_mode != 'max_pooling' and self.down_rate_mode != 'crcrc':
                if self.encoder['struction'] == 'rnn':
                    if return_intermediate is None:
                        output = self.rnn_(output)
                    else:
                        if return_intermediate == 0:
                            intermediate = output
                            output = self.rnn_(output)
                        else:
                            return_intermediate -= 1
                            # get RNN final AND intermediate outputs
                            output, intermediate = self.rnn_(output, return_intermediate=True)
                            # only keep hidden state of requested layer
                            intermediate = intermediate[return_intermediate]
                    #output.size()   [64, 1174, 256]
                else:
                    output = self.sa_(output, self.training)

        #speaker probability for frame-level speaker classification
        if self.loss_cfg["using_frame_spk"] or self.pretraining:
            rep = self.spk_linear2(output)
            rep = self.spk_activation(rep)
            prob = self.activation_2(self.spk_linear3(rep))
        else: 
            prob = None

        #speaker embedding for DPCL
        if not self.pretraining:
            if self.loss_cfg["using_DPCL"]:
                speaker_embedding = F.normalize(torch.tanh(self.spk_linear(output)), p=2, dim=-1)
            else:
                speaker_embedding = None
            if self.loss_cfg['cif_type'] == 'cif0' or self.loss_cfg['cif_type'] == 'cif1':
                output, weight_keep, fired_flag, not_padding_after_cif, sum_a, change_p = self.cif_(output, change_p, self.training)
            elif self.loss_cfg['cif_type'] == 'cif2':
                output, weight_keep, fired_flag, not_padding_after_cif, sum_a = self.cif_(output, self.training)
            elif self.loss_cfg['cif_type'] == 'cif3' or self.loss_cfg['cif_type'] == 'cif4':
                output, weight_keep, fired_flag, not_padding_after_cif, sum_a, aux_change = self.cif_(output, mask, self.training, change_p)
                weight_keep = weight_keep.unsqueeze(-1)

            cif_output = output

            if self.loss_cfg["using_ff"]:
                output = self.ff_(output)

            #output.size()   [64, 1174, 128]
            if self.task.is_representation_learning:
                return self.embedding_(output)

            if self.decoder['struction'] == 'fc':
                if self.decoder.get('extra_fc', False):
                    output = self.dec_linear_extra(output)
                    output = self.activation_hid_extra(output)
                output = self.dec_linear(output)
            elif self.decoder['struction'] == 'tdnn':
                output = self.dec_tdnn(output)
            elif self.decoder['struction'] == 'rnn':
                output, hidden = self.dec_rnn(output)
            elif self.decoder['struction'] == 'sa':
                output = self.dec_sa(output, self.training)
            elif self.decoder['struction'] == 'none':
                pass
            else:
                raise ValueError(f'error decoder struction')
            
            if self.decoder['struction'] != 'none':
                output = self.activation_hid(output)

            # length norm
            if self.decoder.get('length_norm', False):
                output = self.normalize_scalar * F.normalize(output)

            output = self.linear_(output)
            #output.size()   [64, 1174, 2]
            output = self.activation_(output)

            fired_flag_ = 1 - fired_flag
            #print(fired_flag.sum())
            #if return_intermediate is None:
            return cif_output, output, not_padding_after_cif, sum_a, weight_keep, torch.cat([fired_flag_, fired_flag], -1), speaker_embedding, self.loss_cfg, prob, change_p, self.log_var, aux_change
            #return output, intermediate
        else:
            return prob

    @property
    def dimension(self):
        if self.task.is_representation_learning:
            return self.embedding_.dimension

        return Model.dimension.fget(self)

    def intermediate_dimension(self, layer):
        if layer == 0:
            return self.sincnet_.dimension
        return self.rnn_.intermediate_dimension(layer - 1)





class PyanNet2(Model):
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
    embedding : `dict`, optional
        Embedding parameters. Defaults to `Embedding` default parameters. This
        only has effect when model is used for representation learning.
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
        encoder: Optional[dict] = None,
        decoder: Optional[dict] = None,
        sincnet: Optional[dict] = None,
        conv2d: Optional[dict] = None,
        rnn: Optional[dict] = None,
        sa: Optional[dict] = None,
        ff: Optional[dict] = None,
        cif: Optional[dict] = None,
        loss_cfg: Optional[dict] = None,
        embedding: Optional[dict] = None,
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
        ff : `dict`, optional
            Feed-forward layers parameters. Defaults to `FF` default parameters.
        embedding : `dict`, optional
            Embedding parameters. Defaults to `Embedding` default parameters. This
            only has effect when model is used for representation learning.
        """

        if sincnet is None:
            sincnet = dict()
        if conv2d is None:
            conv2d = dict()
        self.sincnet = sincnet
        self.conv2d = conv2d
        self.loss_cfg = loss_cfg
        self.down_rate = loss_cfg["down_rate"]
        self.down_rate_mode = loss_cfg["down_rate_mode"]
        self.tdnn_context = loss_cfg["tdnn_context"]
        self.encoder = encoder
        self.decoder = decoder
        self.sa = sa
        self.rnn = rnn
        self.training = training
        self.pretraining = loss_cfg["pretraining"]
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


        conv2d["n_features"] = n_features
        self.front_end = CONV2D_front(**conv2d)
        n_features = self.front_end.get_output_dim()
        if encoder["using_max_pooling_rnn"]:
            self.rnn_ = RNN_maxp(n_features, **self.rnn)
        else:
            self.rnn_ = RNN(n_features, **rnn)
        n_features = self.rnn_.dimension
        cif['encoder_out_dim'] = n_features
        self.cif = cif
        if loss_cfg['cif_type'] == 'cif0':
            self.cif_ = CIF0(**cif)
        elif loss_cfg['cif_type'] == 'cif1':
            self.cif_ = CIF1(**cif)
        elif loss_cfg['cif_type'] == 'cif2':
            self.cif_ = CIF2(**cif)
        elif loss_cfg['cif_type'] == 'cif3':
            self.cif_ = CIF3(**cif)
        elif loss_cfg['cif_type'] == 'cif4':
            self.cif_ = CIF4(**cif)
        else:
            raise ValueError(f'wrong cif type')


        if self.encoder['struction'] == 'rnn':
            if self.rnn['bidirectional'] == True:
                hidden_size = self.rnn['hidden_size'] * 2
            else:
                hidden_size = self.rnn['hidden_size']
        else:
            hidden_size = self.sa['encoder_embed_dim']

        #self.linear_ = nn.Linear(n_features, len(self.classes), bias=True)
        if self.decoder['struction'] == 'fc':
            self.dec_linear = nn.Linear(hidden_size, self.rnn['hidden_size'], bias=True)
        elif self.decoder['struction'] == 'tdnn':
            self.dec_tdnn = TDNN(
                    context=[-self.decoder['tdnn_context'], self.decoder['tdnn_context']],
                    input_channels=hidden_size,
                    output_channels=self.rnn['hidden_size'],
                    full_context=True,
                    stride=1,
                    padding=self.decoder['tdnn_context'],
                )     

        elif self.decoder['struction'] == 'rnn':
            Klass = getattr(nn, self.rnn_.unit)
            self.dec_rnn = Klass(
                hidden_size,
                self.rnn['hidden_size'],
                num_layers=1,
                bias=self.rnn_.bias,
                batch_first=True,
                dropout=self.rnn_.dropout,
                bidirectional=self.rnn_.bidirectional,
            )
        elif self.decoder['struction'] == 'sa':
            dec_sa = self.sa
            dec_sa['encoder_layers'] = 1
            self.dec_sa = SA(hidden_size, **dec_sa)
        else:
            raise ValueError(f'error decoder struction')


        self.activation_hid = torch.nn.ReLU()


        if self.loss_cfg["spk_loss_type"] == 'bce' or self.loss_cfg["spk_loss_type"] == 'focal_bce':
            num_class = loss_cfg["num_spk_class"]
            self.activation_ = torch.nn.Sigmoid()
        elif self.loss_cfg["spk_loss_type"] == 'softmax':
            num_class = 141
            self.activation_ = torch.nn.LogSoftmax(dim=-1)
        else:
            raise ValueError(f'wrong spk loss type')


        if self.decoder['struction'] != 'rnn':
            self.linear_ = nn.Linear(self.rnn['hidden_size'], num_class, bias=True)
        else:
            self.linear_ = nn.Linear(hidden_size, num_class, bias=True)
        #self.activation_ = self.task.default_activation
        
        if self.loss_cfg["using_DPCL"]:
            self.spk_linear = nn.Linear(hidden_size, hidden_size, bias=True)
        if self.loss_cfg["using_frame_spk"] or self.pretraining:
            self.spk_linear2 = nn.Linear(hidden_size, hidden_size, bias=True)
            self.spk_activation = torch.nn.ReLU()
            self.spk_linear3 = nn.Linear(hidden_size, num_class, bias=True)
            self.activation_2 = torch.nn.Sigmoid()

        log_var = torch.zeros(self.loss_cfg['num_loss']).cuda()
        self.log_var = torch.nn.Parameter(log_var)


    def forward(self, waveforms, change_p=None, mask=None, return_intermediate=None):
        """Forward pass

        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1) `torch.Tensor`
            Batch of waveforms. In case SincNet is skipped, a tensor with shape
            (batch_size, n_samples, n_features) is expected.
        return_intermediate : `int`, optional
            Index of RNN layer. Returns RNN intermediate hidden state.
            Defaults to only return the final output.

        Returns
        -------
        output : `torch.Tensor`
            Final network output.
        intermediate : `torch.Tensor`
            Intermediate network output (only when `return_intermediate`
            is provided).
        """
        #waveforms.size()  [64, 32000, 1]
        if self.sincnet.get("skip", False):
            output = waveforms
        else:
            output = self.sincnet_(waveforms)
        #output.size()  [64, 1174, 60]

        output = self.front_end(output)
        output = self.rnn_(output, return_intermediate=False)

        prob = None
        speaker_embedding = None

        output, weight_keep, fired_flag, not_padding_after_cif, sum_a, aux_change = self.cif_(output, mask, self.training, change_p)
        weight_keep = weight_keep.unsqueeze(-1)

        cif_output = output

        #output.size()   [64, 1174, 128]
        if self.task.is_representation_learning:
            return self.embedding_(output)

        output = self.dec_linear(output)
        output = self.activation_hid(output)

        output = self.linear_(output)
        #output.size()   [64, 1174, 2]
        output = self.activation_(output)

        fired_flag_ = 1 - fired_flag
        #print(fired_flag.sum())
        #if return_intermediate is None:
        return cif_output, output, not_padding_after_cif, sum_a, weight_keep, torch.cat([fired_flag_, fired_flag], -1), speaker_embedding, self.loss_cfg, prob, change_p, self.log_var, aux_change

    @property
    def dimension(self):
        if self.task.is_representation_learning:
            return self.embedding_.dimension

        return Model.dimension.fget(self)

    def intermediate_dimension(self, layer):
        if layer == 0:
            return self.sincnet_.dimension
        return self.rnn_.intermediate_dimension(layer - 1)


class SincTDNN(Model):
    """waveform -> SincNet -> XVectorNet (TDNN -> FC) -> output

    Parameters
    ----------
    sincnet : `dict`, optional
        SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
        default parameters.
    tdnn : `dict`, optional
        X-Vector Time-Delay neural network parameters.
        Defaults to `pyannote.audio.models.tdnn.XVectorNet` default parameters.
    embedding : `dict`, optional
        Embedding parameters. Defaults to `Embedding` default parameters. This
        only has effect when model is used for representation learning.
    """

    @staticmethod
    def get_alignment(task: Task, sincnet=None, **kwargs):
        """Get frame alignment"""

        if sincnet is None:
            sincnet = dict()

        return SincNet.get_alignment(task, **sincnet)

    supports_packed = False

    @staticmethod
    def get_resolution(
        task: Task, sincnet: Optional[dict] = None, **kwargs
    ) -> Resolution:
        """Get sliding window used for feature extraction

        Parameters
        ----------
        task : Task
        sincnet : dict, optional

        Returns
        -------
        sliding_window : `pyannote.core.SlidingWindow` or {`window`, `frame`}
        """

        # TODO add support for frame-wise and sequence labeling tasks
        # TODO https://github.com/pyannote/pyannote-audio/issues/290
        return RESOLUTION_CHUNK

    def init(
        self,
        sincnet: Optional[dict] = None,
        tdnn: Optional[dict] = None,
        embedding: Optional[dict] = None,
    ):
        """waveform -> SincNet -> XVectorNet (TDNN -> FC) -> output

        Parameters
        ----------
        sincnet : `dict`, optional
            SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
            default parameters.
        tdnn : `dict`, optional
            X-Vector Time-Delay neural network parameters.
            Defaults to `pyannote.audio.models.tdnn.XVectorNet` default parameters.
        embedding : `dict`, optional
            Embedding parameters. Defaults to `Embedding` default parameters. This
            only has effect when model is used for representation learning.
        """

        n_features = self.n_features

        if sincnet is None:
            sincnet = dict()
        self.sincnet = sincnet

        if n_features != 1:
            raise ValueError(
                "SincNet only supports mono waveforms. "
                f"Here, waveform has {n_features} channels."
            )
        self.sincnet_ = SincNet(**sincnet)
        n_features = self.sincnet_.dimension

        if tdnn is None:
            tdnn = dict()
        self.tdnn = tdnn
        self.tdnn_ = XVectorNet(n_features, **tdnn)
        n_features = self.tdnn_.dimension

        if self.task.is_representation_learning:
            if embedding is None:
                embedding = dict()
            self.embedding = embedding
            self.embedding_ = Embedding(n_features, **embedding)
        else:
            self.linear_ = nn.Linear(n_features, len(self.classes), bias=True)
            self.activation_ = self.task.default_activation

    def forward(self, waveforms: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1) `torch.Tensor`
            Batch of waveforms

        Returns
        -------
        output : `torch.Tensor`
            Final network output or intermediate network output
            (only when `return_intermediate` is provided).
        """

        output = self.sincnet_(waveforms)

        return_intermediate = (
            "segment6" if self.task.is_representation_learning else None
        )
        output = self.tdnn_(output, return_intermediate=return_intermediate)

        if self.task.is_representation_learning:
            return self.embedding_(output)

        return self.activation_(self.linear_(output))

    @property
    def dimension(self):
        if self.task.is_representation_learning:
            return self.embedding_.dimension

        return Model.dimension.fget(self)


class ACRoPoLiS(Model):
    """Audio -> Convolutional -> Recurrent (-> optional Pooling) -> Linear -> Scores

    Parameters
    ----------
    specifications : dict
        Task specifications.
    convolutional : dict, optional
        Definition of convolutional layers.
        Defaults to convolutional.Convolutional default hyper-parameters.
    recurrent : dict, optional
        Definition of recurrent layers.
        Defaults to recurrent.Recurrent default hyper-parameters.
    pooling : {"last", ""}, optional
        Definition of pooling layer. Only used when self.task.returns_vector is
        True, in which case it defaults to "last" pooling.
    linear : dict, optional
        Definition of linear layers.
        Defaults to linear.Linear default hyper-parameters.
    scale : dict, optional
    """

    def init(
        self,
        convolutional: dict = None,
        recurrent: dict = None,
        linear: dict = None,
        pooling: Text = None,
        scale: dict = None,
    ):

        self.normalize = nn.InstanceNorm1d(self.n_features)

        if convolutional is None:
            convolutional = dict()
        self.convolutional = convolutional
        self.cnn = Convolutional(self.n_features, **convolutional)

        if recurrent is None:
            recurrent = dict()
        self.recurrent = recurrent
        self.rnn = Recurrent(self.cnn.dimension, **recurrent)

        if pooling is None and self.task.returns_vector:
            pooling = "last"
        if pooling is not None and self.task.returns_sequence:
            msg = f"'pooling' should not be used for labeling tasks (is: {pooling})."
            raise ValueError(msg)

        self.pooling = pooling
        self.pool = Pooling(
            self.rnn.dimension,
            method=self.pooling,
            bidirectional=self.rnn.bidirectional,
        )

        if linear is None:
            linear = dict()
        self.linear = linear
        self.ff = Linear(self.pool.dimension, **linear)

        if self.task.is_representation_learning and scale is None:
            scale = dict()
        if not self.task.is_representation_learning and scale is not None:
            msg = (
                f"'scale' should not be used for representation learning (is: {scale})."
            )
            raise ValueError(msg)

        self.scale = scale
        self.scaling = Scaling(self.ff.dimension, **scale)

        if not self.task.is_representation_learning:
            self.final_linear = nn.Linear(
                self.scaling.dimension, len(self.classes), bias=True
            )
            self.final_activation = self.task.default_activation

    def forward(self, waveforms: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1) `torch.Tensor`
            Batch of waveforms

        Returns
        -------
        output : `torch.Tensor`
            Final network output or intermediate network output
            (only when `return_intermediate` is provided).
        """

        output = self.normalize(waveforms.transpose(1, 2)).transpose(1, 2)
        output = self.cnn(output)
        output = self.rnn(output)
        output = self.pool(output)
        output = self.ff(output)

        if not self.task.is_representation_learning:
            output = self.final_linear(output)
            output = self.final_activation(output)
        return output

    @property
    def dimension(self):
        if self.task.is_representation_learning:
            return self.ff.dimension
        else:
            return len(self.classes)

    @staticmethod
    def get_alignment(task: Task, convolutional: Optional[dict] = None, **kwargs):
        """Get frame alignment"""

        if convolutional is None:
            convolutional = dict()

        return Convolutional.get_alignment(task, **convolutional)

    @staticmethod
    def get_resolution(
        task: Task, convolutional: Optional[dict] = None, **kwargs
    ) -> Resolution:
        """Get frame resolution

        Parameters
        ----------
        task : Task
        convolutional : dict, optional

        Returns
        -------
        sliding_window : `pyannote.core.SlidingWindow` or {`window`, `frame`}
        """

        if task.returns_vector:
            return RESOLUTION_CHUNK

        if convolutional is None:
            convolutional = dict()

        return Convolutional.get_resolution(task, **convolutional)
