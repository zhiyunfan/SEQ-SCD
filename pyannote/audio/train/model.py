#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019-2020 CNRS

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

"""Models

## Parts

>>> model.parts
["ff.1", "ff.2", "ff.3"]

## Probes

>>> model.probes = ["ff.1", "ff.2"]
>>> output, probes = model(input)
>>> ff1 = probes["ff.1"]
>>> ff2 = probes["ff.2"]

>>> del model.probes
>>> output = model(input)

## Freeze/unfreeze layers

>>> model.freeze(["ff.1", "ff.2"])
>>> model.unfreeze(["ff.2"])

"""

from typing import Union
from typing import List
from typing import Text
from typing import Tuple
from typing import Dict

try:
    from typing import Literal
except ImportError as e:
    from typing_extensions import Literal
from typing import Callable
from pyannote.core import SlidingWindow
from pyannote.core import SlidingWindowFeature

RESOLUTION_FRAME = "frame"
RESOLUTION_CHUNK = "chunk"
Resolution = Union[SlidingWindow, Literal[RESOLUTION_FRAME, RESOLUTION_CHUNK]]

ALIGNMENT_CENTER = "center"
ALIGNMENT_STRICT = "strict"
ALIGNMENT_LOOSE = "loose"
Alignment = Literal[ALIGNMENT_CENTER, ALIGNMENT_STRICT, ALIGNMENT_LOOSE]

from pyannote.audio.train.task import Task
import numpy as np
import pescador
import torch
from torch.nn import Module
from functools import partial
import torch.nn.functional as F
from .callback import Callback
from .callback import Callbacks
from .logging import Logging
from tqdm import tqdm
import copy

class Model(Module):
    """Model

    A `Model` is nothing but a `torch.nn.Module` instance with a bunch of
    additional methods and properties specific to `pyannote.audio`.

    It is expected to be instantiated with a unique `specifications` positional
    argument describing the task addressed by the model, and a user-defined
    number of keyword arguments describing the model architecture.

    Parameters
    ----------
    specifications : `dict`
        Task specifications.
    **architecture_params : `dict`
        Architecture hyper-parameters.
    """

    def __init__(self, specifications: dict, **architecture_params):
        super().__init__()
        self.specifications = specifications
        training = specifications['training']
        architecture_params['training'] = training
        self.resolution_ = self.get_resolution(self.task, **architecture_params)
        self.alignment_ = self.get_alignment(self.task, **architecture_params)
        self.init(**architecture_params)

        ### for logging development
        # self.batches_pbar_ = tqdm(
        #         desc=f"Epoch #{'development'}",
        #         leave=False,
        #         ncols=80,
        #         unit="batch",
        #         position=1,
        #     )

        self.n_batches_ = 0
        self.loss_moving_avg_ = dict()
        self.beta_ = 0.98


    def init(self, **architecture_params):
        """Initialize model architecture

        This method is called by Model.__init__ after attributes
        'specifications', 'resolution_', and 'alignment_' have been set.

        Parameters
        ----------
        **architecture_params : `dict`
            Architecture hyper-parameters

        """
        msg = 'Method "init" must be overriden.'
        raise NotImplementedError(msg)

    @property
    def probes(self):
        """Get list of probes"""
        return list(getattr(self, "_probes", []))

    @probes.setter
    def probes(self, names: List[Text]):
        """Set list of probes

        Parameters
        ----------
        names : list of string
            Names of modules to probe.
        """

        for handle in getattr(self, "handles_", []):
            handle.remove()

        self._probes = []

        if not names:
            return

        handles = []

        def _init(module, input):
            self.probed_ = dict()

        handles.append(self.register_forward_pre_hook(_init))

        def _append(name, module, input, output):
            self.probed_[name] = output

        for name, module in self.named_modules():
            if name in names:
                handles.append(module.register_forward_hook(partial(_append, name)))
                self._probes.append(name)

        def _return(module, input, output):
            return output, self.probed_

        handles.append(self.register_forward_hook(_return))

        self.handles_ = handles

    @probes.deleter
    def probes(self):
        """Remove all probes"""
        for handle in getattr(self, "handles_", []):
            handle.remove()
        self._probes = []

    @property
    def parts(self):
        """Names of (freezable / probable) modules"""
        return [n for n, _ in self.named_modules()]

    def freeze(self, names: List[Text]):
        """Freeze parts of the model

        Parameters
        ----------
        names : list of string
            Names of modules to freeze.
        """
        for name, module in self.named_modules():
            if name in names:
                for parameter in module.parameters(recurse=True):
                    parameter.requires_grad = False

    def unfreeze(self, names: List[Text]):
        """Unfreeze parts of the model

        Parameters
        ----------
        names : list of string
            Names of modules to unfreeze.
        """

        for name, module in self.named_modules():
            if name in names:
                for parameter in module.parameters(recurse=True):
                    parameter.requires_grad = True

    def forward(
        self, sequences: torch.Tensor, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[Text, torch.Tensor]]]:
        """TODO

        Parameters
        ----------
        sequences : (batch_size, n_samples, n_features) `torch.Tensor`
        **kwargs : `dict`

        Returns
        -------
        output : (batch_size, ...) `torch.Tensor`
        probes : dict, optional
        """

        # TODO
        msg = "..."
        raise NotImplementedError(msg)

    @property
    def task(self) -> Task:
        """Type of task addressed by the model

        Shortcut for self.specifications['task']
        """
        return self.specifications["task"]

    def get_resolution(self, task: Task, **architecture_params) -> Resolution:
        """Get frame resolution

        This method is called by `BatchGenerator` instances to determine how
        target tensors should be built.

        Depending on the task and the architecture, the output of a model will
        have different resolution. The default behavior is to return
        - `RESOLUTION_CHUNK` if the model returns just one output for the whole
          input sequence
        - `RESOLUTION_FRAME` if the model returns one output for each frame of
          the input sequence

        In case neither of these options is valid, this method needs to be
        overriden to return a custom `SlidingWindow` instance.

        Parameters
        ----------
        task : Task
        **architecture_params
            Parameters used for instantiating the model architecture.

        Returns
        -------
        resolution : `Resolution`
            - `RESOLUTION_CHUNK` if the model returns one single output for the
              whole input sequence;
            - `RESOLUTION_FRAME` if the model returns one output for each frame
               of the input sequence.
        """

        if task.returns_sequence:
            return RESOLUTION_FRAME

        elif task.returns_vector:
            return RESOLUTION_CHUNK

        else:
            # this should never happened
            msg = f"{task} tasks are not supported."
            raise NotImplementedError(msg)

    @property
    def resolution(self) -> Resolution:
        return self.resolution_

    def get_alignment(self, task: Task, **architecture_params) -> Alignment:
        """Get frame alignment

        This method is called by `BatchGenerator` instances to dermine how
        target tensors should be aligned with the output of the model.

        Default behavior is to return 'center'. In most cases, you should not
        need to worry about this but if you do, this method can be overriden to
        return 'strict' or 'loose'.

        Parameters
        ----------
        task : Task
        architecture_params : dict
            Architecture hyper-parameters.

        Returns
        -------
        alignment : `Alignment`
            Target alignment. Must be one of 'center', 'strict', or 'loose'.
            Always returns 'center'.
        """

        return ALIGNMENT_CENTER

    @property
    def alignment(self) -> Alignment:
        return self.alignment_

    @property
    def n_features(self) -> int:
        """Number of input features

        Shortcut for self.specifications['X']['dimension']

        Returns
        -------
        n_features : `int`
            Number of input features
        """
        return self.specifications["X"]["dimension"]

    @property
    def dimension(self) -> int:
        """Output dimension

        This method needs to be overriden for representation learning tasks,
        because output dimension cannot be inferred from the task
        specifications.

        Returns
        -------
        dimension : `int`
            Dimension of model output.

        Raises
        ------
        AttributeError
            If the model addresses a classification or regression task.
        """

        if self.task.is_representation_learning:
            msg = (
                f"Class {self.__class__.__name__} needs to define "
                f"'dimension' property."
            )
            raise NotImplementedError(msg)

        msg = f"{self.task} tasks do not define attribute 'dimension'."
        raise AttributeError(msg)

    @property
    def classes(self) -> List[str]:
        """Names of classes

        Shortcut for self.specifications['y']['classes']

        Returns
        -------
        classes : `list` of `str`
            List of names of classes.


        Raises
        ------
        AttributeError
            If the model does not address a classification task.
        """

        if not self.task.is_representation_learning:
            return self.specifications["y"]["classes"]

        msg = f"{self.task} tasks do not define attribute 'classes'."
        raise AttributeError(msg)


    def crop_y(self, y, segment):
        """Extract y for specified segment

        Parameters
        ----------
        y : `pyannote.core.SlidingWindowFeature`
            Output of `initialize_y` above.
        segment : `pyannote.core.Segment`
            Segment for which to obtain y.

        Returns
        -------
        cropped_y : (n_samples, dim) `np.ndarray`
            y for specified `segment`
        """

        return y.crop(segment, mode="center", fixed=4)

    def on_batch_end(self, batch_loss):

        # time spent in forward/backward

        self.n_batches_ += 1

        self.loss = dict()
        for key in batch_loss:
            # if not key.startswith("loss"):
            #     continue
            loss = batch_loss[key].detach().cpu().item()
            self.loss_moving_avg_[key] = (
                self.beta_ * self.loss_moving_avg_.setdefault(key, 0.0)
                + (1 - self.beta_) * loss
            )
            self.loss[key] = self.loss_moving_avg_[key] / (
                1 - self.beta_ ** self.n_batches_
            )
            
        # self.batches_pbar_.set_postfix(ordered_dict=self.loss)
        # self.batches_pbar_.update(1)


    def slide(
        self,
        features: SlidingWindowFeature,
        labels,
        sliding_window: SlidingWindow,
        batch_size: int = 32,
        device: torch.device = None,
        skip_average: bool = None,
        postprocess: Callable[[np.ndarray], np.ndarray] = None,
        return_intermediate=None,
        progress_hook=None,
        down_rate=1,
        writer=None,
    ) -> SlidingWindowFeature:
        """Slide and apply model on features

        Parameters
        ----------
        features : SlidingWindowFeature
            Input features.
        sliding_window : SlidingWindow
            Sliding window used to apply the model.
        batch_size : int
            Batch size. Defaults to 32. Use large batch for faster inference.
        device : torch.device
            Device used for inference.
        skip_average : bool, optional
            For sequence labeling tasks (i.e. when model outputs a sequence of
            scores), each time step may be scored by several consecutive
            locations of the sliding window. Default behavior is to average
            those multiple scores. Set `skip_average` to False to return raw
            scores without averaging them.
        postprocess : callable, optional
            Function applied to the predictions of the model, for each batch
            separately. Expects a (batch_size, n_samples, n_features) np.ndarray
            as input, and returns a (batch_size, n_samples, any) np.ndarray.
        return_intermediate :
            Experimental. Not documented yet.
        progress_hook : callable
            Experimental. Not documented yet.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        if skip_average is None:
            skip_average = (self.resolution == RESOLUTION_CHUNK) or (
                return_intermediate is not None
            )

        try:
            dimension = self.dimension
        except AttributeError:
            dimension = len(self.classes)

        resolution = self.resolution

        # model returns one vector per input frame
        if resolution == RESOLUTION_FRAME:
            resolution = features.sliding_window

        # model returns one vector per input window
        if resolution == RESOLUTION_CHUNK:
            resolution = sliding_window

        support = features.extent
        if support.duration < sliding_window.duration:
            chunks = [support]
            fixed = support.duration
        else:
            chunks = list(sliding_window(support, align_last=True))
            fixed = sliding_window.duration

        if progress_hook is not None:
            n_chunks = len(chunks)
            n_done = 0
            progress_hook(n_done, n_chunks)


        batches = pescador.maps.buffer_stream(
            iter(
                {"X": features.crop(window, mode="center", fixed=fixed),
                'y': self.crop_y(labels, window)}
                for window in chunks
            ),
            batch_size,
            partial=True,
        )
        

        fX = []
        for batch in batches:
            tX = torch.tensor(batch["X"], dtype=torch.float32, device=device)

            # FIXME: fix support for return_intermediate
            with torch.no_grad():
                cif_output, output, not_padding_after_cif, sum_a, fired_flag, loss_cfg = self(tX, mask=None)

            tfX = torch.cat([fired_flag[:,:,-1].unsqueeze(-1), fired_flag[:,:,-1].unsqueeze(-1)], -1)

            if loss_cfg["plot_speaker"] == True:
                self.plot_speaker(cif_output, not_padding_after_cif, batch["y"], loss_cfg)

            tfX_npy = tfX.detach().to("cpu").numpy()
            if postprocess is not None:
                tfX_npy = postprocess(tfX_npy)

            fX.append(tfX_npy)

            if progress_hook is not None:
                n_done += len(batch["X"])
                progress_hook(n_done, n_chunks)

        fX = np.vstack(fX)

        # get total number of frames (based on last window end time)
        resolution_ = copy.copy(resolution)
        if down_rate != 1:
            resolution_._SlidingWindow__duration = resolution._SlidingWindow__duration + (down_rate - 1) * resolution._SlidingWindow__step
            resolution_._SlidingWindow__step = resolution._SlidingWindow__step * down_rate

        n_frames = resolution_.samples(chunks[-1].end, mode="center")
        # data[i] is the sum of all predictions for frame #i
        data = np.zeros((n_frames, dimension), dtype=np.float32)

        # k[i] is the number of chunks that overlap with frame #i
        k = np.zeros((n_frames, 1), dtype=np.int8)

        for chunk, fX_ in zip(chunks, fX):
            # indices of frames overlapped by chunk
            indices = resolution_.crop(chunk, mode=self.alignment, fixed=fixed)

            # accumulate the outputs
            data[indices] += fX_

            # keep track of the number of overlapping sequence
            # TODO - use smarter weights (e.g. Hamming window)
            k[indices] += 1
        # compute average embedding of each frame
        data = data / np.maximum(k, 1)

        def process(data):
            for t in range(1, data.shape[0]-1):
                if data[t+1][-1] != 0 and data[t][-1] != 0:
                    value = data[t]
                    u = t
                    while (u < data.shape[0] -1 and data[u+1][-1]!=0):
                        u += 1   
                    peak = (t + u) // 2
                    tmp1 = np.sum(data[t:u+1, 0])
                    tmp2 = np.sum(data[t:u+1, 1])
                    data[t:u+1,0] = 1
                    data[t:u+1,1] = 0
                    data[peak,0] = tmp1
                    data[peak,1] = tmp2
            return None

        process(data)
        return SlidingWindowFeature(data, resolution_)

    def compute_acc(self, prob, label, mask=None):
        batch_size = label.size(0)
        lens = label.size(1)
        classes = label.size(2)
        non_sil = (label.sum(-1) > 0).int()

        if mask is None:
            index = prob.argmax(-1)
            pred = torch.zeros_like(label).scatter_(2,index.unsqueeze(-1),1)
            correct = torch.sum((pred + label) > 1)
            acc = correct/non_sil.sum()
        else:
            index = prob.argmax(-1)
            pred = torch.zeros_like(label).scatter_(2,index.unsqueeze(-1),1)
            correct = torch.sum(((pred + label) > 1)*mask)
            non_sil = non_sil * mask[:,:,0]
            acc = correct/non_sil.sum()
        return acc

    def plot_speaker(self, X, not_padding_after_cif, Y, loss_cfg):
        target, mask, change_p = self.process_sil_rep(Y[:,:,:-1])
        X, not_padding_after_cif = self.fix_fX(X, not_padding_after_cif, target)
        fired_len = not_padding_after_cif.sum(-1).int()
        X = X.detach().to("cpu").numpy()
        with open(loss_cfg["speaker_embedding_path"], 'a') as f:
            for i in range(X.shape[0]):
                li = fired_len[i]
                xi = X[i]
                yi = target[i]
                for j in range(li):
                    if sum(yi[j]) != 1:
                        continue
                    else:
                        speaker_id = np.expand_dims(np.argmax(yi[j], -1),0)
                        line = np.concatenate([np.expand_dims(speaker_id, 0),np.expand_dims(xi[j],0)], axis=-1)
                        np.savetxt(f, line,fmt='%f',delimiter=' ')
        print("save speaker embedding done !")

    #####function for computing loss
    def compute_loss(self, batch, output, not_padding_after_cif, sum_a, change_p, weight_keep, fired_flag, speaker_embedding, loss_cfg, prob, device):
        frame_target = batch["y"]
        target, mask, change_p = self.process_sil_rep(frame_target)

        output, not_padding_after_cif = self.fix_fX(output, not_padding_after_cif, target)
        
        target = torch.tensor(target, dtype=torch.float32, device=device)
        mask = torch.tensor(
                mask, dtype=torch.float32, device=device
            )

        acc = self.compute_acc(output, target, mask)
        not_padding_after_cif = not_padding_after_cif.unsqueeze(-1).repeat(1, 1, mask.size(-1))

        if "mask" in batch:
            mask = torch.tensor(batch["mask"], dtype=torch.float32, device=device)

        loss, count_loss = self.loss_func(loss_cfg, output, target, sum_a=sum_a, mask=mask, not_padding_after_cif=not_padding_after_cif)
        return {"loss": loss, "count_loss": count_loss, "acc": acc}
 
    def process_sil_rep(self, y):
        batch_size = y.shape[0]
        length = y.shape[1]
        num_class = y.shape[2]
        y_ = []
        change_p_ = []
        max_len = 0
        for i in range(batch_size):
            cur_y = y[i]
            mask = np.sum(np.abs(cur_y), -1) != 0
            cur_y_ = cur_y[mask]
            # all sil
            change_p = np.sum(np.abs(np.diff(cur_y, axis=0)), axis=1, keepdims=True)
            change_p = np.vstack(([[False]], change_p > 0)).astype(float)
            if cur_y_.shape[0] == 0:
                cur_y =np.expand_dims(cur_y[0], 0)
            else:
                cur_y = cur_y_
            index = np.sum(np.abs(np.diff(cur_y, axis=0)), axis=1, keepdims=True)
            
            index = np.vstack(([[True]], index > 0)).squeeze(-1)
            index = np.where(index == True)
            cur_y = cur_y[index]
            max_len = max([max_len, len(cur_y)])
            y_.append(cur_y)
            change_p_.append(np.expand_dims(change_p,0))

        # padding
        y_o = []
        mask = np.ones([batch_size, max_len])
        for i in range(batch_size):
            pad_len = max_len - len(y_[i])
            mask[i][max_len-pad_len:] = 0
            pad = np.zeros([pad_len,num_class])
            cur_y = np.expand_dims(np.concatenate([y_[i], pad], axis=0), 0)
            
            y_o.append(cur_y)

        y_o = np.concatenate(y_o, axis=0)
        change_p_ = np.concatenate(change_p_, axis=0)
        mask = np.expand_dims(mask, -1).repeat(y_o.shape[-1], -1)
        return y_o, mask, change_p_

    def fix_fX(self, fX, not_padding_after_cif, y):
        batch_size = y.shape[0]
        max_len = y.shape[1]
        dim = fX.size(2)
        if fX.size(1) > max_len:
            return fX[:, :max_len, :], not_padding_after_cif[:, :max_len]
        else:
            pad_len = max_len - fX.size(1) 
            pad = torch.zeros([batch_size, pad_len, dim]).cuda()
            not_padding_after_cif = torch.cat([not_padding_after_cif, torch.zeros([batch_size, pad_len]).cuda()], 1)
            return torch.cat([fX, pad], 1), not_padding_after_cif   

        def count_loss_func(mask, sum_a):
            count = torch.sum(mask[:,:,0], dim=-1) - 1
            return torch.mean(torch.abs(count-sum_a))

    def loss_func(self, loss_cfg, input, target, sum_a, mask=None, not_padding_after_cif=None):
        count_loss = self.count_loss_func(mask, sum_a)

        if mask is None:
            loss = F.binary_cross_entropy(
                input, target, weight=None, reduction="mean"
            ) + loss_cfg["count_weight"] * count_loss


            count_loss = loss_cfg["count_weight"] * count_loss
            return loss, count_loss
        else:
            
            loss = torch.mean(
                mask
                * F.binary_cross_entropy(
                    input, target, weight=None, reduction="none"
                )
            ) + loss_cfg["count_weight"] * count_loss
            
            count_loss = loss_cfg["count_weight"] * count_loss
            return loss, count_loss
