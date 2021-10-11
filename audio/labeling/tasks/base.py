#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2020 CNRS

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

from typing import Optional
from typing import Text

import torch
import torch.nn.functional as F

import numpy as np
import scipy.signal

from tqdm import tqdm

from pyannote.core import Segment
from pyannote.core import SlidingWindow
from pyannote.core import Timeline
from pyannote.core import Annotation
from pyannote.core import SlidingWindowFeature

from pyannote.database import get_unique_identifier
from pyannote.database import get_annotated
from pyannote.database import Protocol
from pyannote.database import Subset

from pyannote.core.utils.numpy import one_hot_encoding


from pyannote.audio.features import RawAudio
from pyannote.audio.features.wrapper import Wrapper, Wrappable

from pyannote.core.utils.random import random_segment
from pyannote.core.utils.random import random_subsegment

from pyannote.audio.train.trainer import Trainer
from pyannote.audio.train.generator import BatchGenerator

from pyannote.audio.train.task import Task, TaskType, TaskOutput

from pyannote.audio.train.model import Resolution
from pyannote.audio.train.model import RESOLUTION_CHUNK
from pyannote.audio.train.model import RESOLUTION_FRAME
from pyannote.audio.train.model import Alignment

from pyannote.audio.utils.AutomaticWeightedLoss import AutomaticWeightedLoss
from pyannote.audio.utils import wer

SECONDS_IN_A_DAY = 24 * 60 * 60
torch.set_printoptions(profile="full")

import pdb
import logging
import copy

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class LabelingTaskGenerator(BatchGenerator):
    """Base batch generator for various labeling tasks

    This class should be inherited from: it should not be used directy

    Parameters
    ----------
    task : Task
        Task
    feature_extraction : Wrappable
        Describes how features should be obtained.
        See pyannote.audio.features.wrapper.Wrapper documentation for details.
    protocol : Protocol
    subset : {'train', 'development', 'test'}, optional
        Protocol and subset.
    resolution : `pyannote.core.SlidingWindow`, optional
        Override `feature_extraction.sliding_window`. This is useful for
        models that include the feature extraction step (e.g. SincNet) and
        therefore output a lower sample rate than that of the input.
        Defaults to `feature_extraction.sliding_window`
    alignment : {'center', 'loose', 'strict'}, optional
        Which mode to use when cropping labels. This is useful for models that
        include the feature extraction step (e.g. SincNet) and therefore use a
        different cropping mode. Defaults to 'center'.
    duration : float, optional
        Duration of audio chunks. Defaults to 2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Force total audio duration per epoch, in days.
        Defaults to total duration of protocol subset.
    exhaustive : bool, optional
        Ensure training files are covered exhaustively (useful in case of
        non-uniform label distribution).
    step : `float`, optional
        Ratio of audio chunk duration used as step between two consecutive
        audio chunks. Defaults to 0.1. Has not effect when exhaustive is False.
    mask : str, optional
        When provided, protocol files are expected to contain a key named after
        this `mask` variable and providing a `SlidingWindowFeature` instance.
        Generated batches will contain an additional "mask" key (on top of
        existing "X" and "y" keys) computed as an excerpt of `current_file[mask]`
        time-aligned with "y". Defaults to not add any "mask" key.
    local_labels : bool, optional
        Set to True to yield samples with local (file-level) labels.
        Defaults to use global (protocol-level) labels.
    """

    def __init__(
        self,
        task: Task,
        feature_extraction: Wrappable,
        protocol: Protocol,
        subset: Subset = "train",
        resolution: Optional[Resolution] = None,
        alignment: Optional[Alignment] = None,
        duration: float = 2.0,
        batch_size: int = 32,
        per_epoch: float = None,
        exhaustive: bool = False,
        step: float = 0.1,
        mask: Text = None,
        local_labels: bool = False,
        down_rate: int = 1,
        spk_list: list = [],
        collar: float = 0.1,
    ):

        self.task = task
        self.feature_extraction = Wrapper(feature_extraction)
        self.duration = duration
        self.exhaustive = exhaustive
        self.step = step
        self.mask = mask
        self.local_labels = local_labels

        self.resolution_ = resolution

        if alignment is None:
            alignment = "center"
        self.alignment = alignment

        self.batch_size = batch_size
        self.down_rate = down_rate

        # load metadata and estimate total duration of training data
        total_duration = self._load_metadata(protocol, subset=subset, spk_list=spk_list)
        #
        if per_epoch is None:

            # 1 epoch = covering the whole training set once
            #
            per_epoch = total_duration / SECONDS_IN_A_DAY

            # when exhaustive is False, this is not completely correct.
            # in practice, it will randomly sample audio chunk until their
            # overall duration reaches the duration of the training set.
            # but nothing guarantees that every single part of the training set
            # has been seen exactly once: it might be more than once, it might
            # be less than once. on average, however, after a certain amount of
            # epoch, this will be correct

            # when exhaustive is True, however, we can actually make sure every
            # single part of the training set has been seen. we just have to
            # make sur we account for the step used by the exhaustive sliding
            # window
            if self.exhaustive:
                per_epoch *= np.ceil(1 / self.step)
        self.per_epoch = per_epoch

    # TODO. use cached property (Python 3.8 only)
    # https://docs.python.org/fr/3/library/functools.html#functools.cached_property
    @property
    def resolution(self):

        if self.resolution_ in [None, RESOLUTION_FRAME]:
            return self.feature_extraction.sliding_window

        if self.resolution_ == RESOLUTION_CHUNK:
            return self.SlidingWindow(
                duration=self.duration, step=self.step * self.duration
            )

        return self.resolution_

    def postprocess_y(self, Y: np.ndarray) -> np.ndarray:
        """This function does nothing but return its input.
        It should be overriden by subclasses.

        Parameters
        ----------
        Y : (n_samples, n_speakers) numpy.ndarray

        Returns
        -------
        postprocessed :

        """
        return Y

    def initialize_y(self, current_file, spk_list):
        """Precompute y for the whole file

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol.

        Returns
        -------
        y : `SlidingWindowFeature`
            Precomputed y for the whole file
        """
        if self.local_labels:
            labels = current_file["annotation"].labels()
        else:
            labels = self.segment_labels_
        ###fzy
        resolution_ = copy.copy(self.resolution)
        if self.down_rate != 1:
            resolution_._SlidingWindow__duration = resolution_._SlidingWindow__duration + (self.down_rate - 1) * resolution_._SlidingWindow__step
            resolution_._SlidingWindow__step = resolution_._SlidingWindow__step * self.down_rate


        y = one_hot_encoding(
            current_file["annotation"],
            get_annotated(current_file),
            resolution_,
            labels=spk_list,
            mode="center",
            is_training=True,
        )
        y_change = self.postprocess_y(y.data)
        y.data = np.concatenate([y.data, y_change], -1)
        return y

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

        return y.crop(segment, mode=self.alignment, fixed=self.duration)

    def _load_metadata(self, protocol, subset: Subset = "train", spk_list = []) -> float:
        """Load training set metadata

        This function is called once at instantiation time, returns the total
        training set duration, and populates the following attributes:

        Attributes
        ----------
        data_ : dict

            {'segments': <list of annotated segments>,
             'duration': <total duration of annotated segments>,
             'current_file': <protocol dictionary>,
             'y': <labels as numpy array>}

        segment_labels_ : list
            Sorted list of (unique) labels in protocol.

        file_labels_ : dict of list
            Sorted lists of (unique) file labels in protocol

        Returns
        -------
        duration : float
            Total duration of annotated segments, in seconds.
        """
        self.data_ = {}
        segment_labels, file_labels = set(), dict()

        # loop once on all files
        files = getattr(protocol, subset)()
        for current_file in tqdm(files, desc="Loading labels", unit="file"):

            # ensure annotation/annotated are cropped to actual file duration
            support = Segment(start=0, end=current_file["duration"])
            current_file["annotated"] = get_annotated(current_file).crop(
                support, mode="intersection"
            )
            current_file["annotation"] = current_file["annotation"].crop(
                support, mode="intersection"
            )

            # keep track of unique segment labels
            segment_labels.update(current_file["annotation"].labels())

            # keep track of unique file labels
            for key, value in current_file.items():
                if isinstance(value, (Annotation, Timeline, SlidingWindowFeature)):
                    continue
                if key not in file_labels:
                    file_labels[key] = set()
                file_labels[key].add(value)

            segments = [
                s for s in current_file["annotated"] if s.duration > self.duration
            ]

            # corner case where no segment is long enough
            # and we removed them all...
            if not segments:
                continue

            # total duration of label in current_file (after removal of
            # short segments).
            duration = sum(s.duration for s in segments)

            # store all these in data_ dictionary
            datum = {
                "segments": segments,
                "duration": duration,
                "current_file": current_file,
            }
            uri = get_unique_identifier(current_file)
            self.data_[uri] = datum

        self.file_labels_ = {k: sorted(file_labels[k]) for k in file_labels}
        self.segment_labels_ = sorted(segment_labels)


        for uri in list(self.data_):
            current_file = self.data_[uri]["current_file"]
            y = self.initialize_y(current_file, spk_list)
            self.data_[uri]["y"] = y
            if self.mask is not None:
                mask = current_file[self.mask]
                current_file[self.mask] = mask.align(y)

        return sum(datum["duration"] for datum in self.data_.values())

    @property
    def specifications(self):
        """Task & sample specifications

        Returns
        -------
        specs : `dict`
            ['task'] (`pyannote.audio.train.Task`) : task
            ['X']['dimension'] (`int`) : features dimension
            ['y']['classes'] (`list`) : list of classes
        """

        specs = {
            "task": self.task,
            "X": {"dimension": self.feature_extraction.dimension},
        }

        if not self.local_labels:
            specs["y"] = {"classes": self.segment_labels_}

        return specs

    def samples(self):
        if self.exhaustive:
            return self._sliding_samples()
        else:
            return self._random_samples()

    def _random_samples(self):
        """Random samples

        Returns
        -------
        samples : generator
            Generator that yields {'X': ..., 'y': ...} samples indefinitely.
        """
        
        uris = list(self.data_)
        durations = np.array([self.data_[uri]["duration"] for uri in uris])
        probabilities = durations / np.sum(durations)
        
        while True:
            # choose file at random with probability
            # proportional to its (annotated) duration
            uri = uris[np.random.choice(len(uris), p=probabilities)]

            datum = self.data_[uri]
            current_file = datum["current_file"]

            # choose one segment at random with probability
            # proportional to its duration
            segment = next(random_segment(datum["segments"], weighted=True))
            
            # choose fixed-duration subsegment at random
            subsegment = next(random_subsegment(segment, self.duration))
            X = self.feature_extraction.crop(
                current_file, subsegment, mode="center", fixed=self.duration
            )
            y = self.crop_y(datum["y"], subsegment)
            sample = {"X": X, "y": y}
            
            if self.mask is not None:
                mask = self.crop_y(current_file[self.mask], subsegment)
                sample["mask"] = mask

            for key, classes in self.file_labels_.items():
                sample[key] = classes.index(current_file[key])
          
            yield sample

    def _sliding_samples(self):

        uris = list(self.data_)
        durations = np.array([self.data_[uri]["duration"] for uri in uris])
        probabilities = durations / np.sum(durations)
        sliding_segments = SlidingWindow(
            duration=self.duration, step=self.step * self.duration
        )

        while True:

            np.random.shuffle(uris)

            # loop on all files
            for uri in uris:

                datum = self.data_[uri]

                # make a copy of current file
                current_file = dict(datum["current_file"])

                # compute features for the whole file
                features = self.feature_extraction(current_file)

                # randomly shift 'annotated' segments start time so that
                # we avoid generating exactly the same subsequence twice
                annotated = Timeline()
                for segment in get_annotated(current_file):
                    shifted_segment = Segment(
                        segment.start + np.random.random() * self.duration, segment.end
                    )
                    if shifted_segment:
                        annotated.add(shifted_segment)

                samples = []
                for sequence in sliding_segments(annotated):

                    X = features.crop(sequence, mode="center", fixed=self.duration)
                    y = self.crop_y(datum["y"], sequence)
                    sample = {"X": X, "y": y}

                    if self.mask is not None:

                        # extract mask for current sub-segment
                        mask = current_file[self.mask].crop(
                            sequence, mode="center", fixed=self.duration
                        )

                        # it might happen that "mask" and "y" use different
                        # sliding windows. therefore, we simply resample "mask"
                        # to match "y"
                        if len(mask) != len(y):
                            mask = scipy.signal.resample(mask, len(y), axis=0)
                        sample["mask"] = mask

                    for key, classes in self.file_labels_.items():
                        sample[key] = classes.index(current_file[key])

                    samples.append(sample)

                np.random.shuffle(samples)
                for sample in samples:
                    yield sample

    @property
    def batches_per_epoch(self):
        """Number of batches needed to complete an epoch"""
        duration_per_epoch = self.per_epoch * SECONDS_IN_A_DAY
        duration_per_batch = self.duration * self.batch_size
        return int(np.ceil(duration_per_epoch / duration_per_batch))


class LabelingTask(Trainer):
    """Base class for various labeling tasks

    This class should be inherited from: it should not be used directy

    Parameters
    ----------
    duration : float, optional
        Duration of audio chunks. Defaults to 2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Force total audio duration per epoch, in days.
        Defaults to total duration of protocol subset.
    exhaustive : bool, optional
        Ensure training files are covered exhaustively (useful in case of
        non-uniform label distribution).
    step : `float`, optional
        Ratio of audio chunk duration used as step between two consecutive
        audio chunks. Defaults to 0.1. Has not effect when exhaustive is False.
    """

    def __init__(
        self,
        duration: float = 2.0,
        batch_size: int = 32,
        per_epoch: float = None,
        exhaustive: bool = False,
        step: float = 0.1,
        collar: float = 0.1,
        non_speech: bool = True,
        #num_loss: int = 1,
    ):
        super(LabelingTask, self).__init__()
        self.duration = duration
        self.batch_size = batch_size
        self.per_epoch = per_epoch
        self.exhaustive = exhaustive
        self.step = step
        self.collar = collar


    def get_batch_generator(
        self,
        feature_extraction: Wrappable,
        protocol: Protocol,
        subset: Subset = "train",
        resolution: Optional[Resolution] = None,
        alignment: Optional[Alignment] = None,
        down_rate = 1,
        spk_list = [],
        collar = 0.1,
    ) -> LabelingTaskGenerator:
        """This method should be overriden by subclass

        Parameters
        ----------
        feature_extraction : Wrappable
            Describes how features should be obtained.
            See pyannote.audio.features.wrapper.Wrapper documentation for details.
        protocol : Protocol
        subset : {'train', 'development'}, optional
            Defaults to 'train'.
        resolution : `pyannote.core.SlidingWindow`, optional
            Override `feature_extraction.sliding_window`. This is useful for
            models that include the feature extraction step (e.g. SincNet) and
            therefore output a lower sample rate than that of the input.
        alignment : {'center', 'loose', 'strict'}, optional
            Which mode to use when cropping labels. This is useful for models
            that include the feature extraction step (e.g. SincNet) and
            therefore use a different cropping mode. Defaults to 'center'.

        Returns
        -------
        batch_generator : `LabelingTaskGenerator`
        """
        #self.collar = collar
        #self.resolution = resolution
        return LabelingTaskGenerator(
            self.task,
            feature_extraction,
            protocol,
            subset=subset,
            resolution=resolution,
            alignment=alignment,
            duration=self.duration,
            per_epoch=self.per_epoch,
            batch_size=self.batch_size,
            exhaustive=self.exhaustive,
            step=self.step,
            down_rate=down_rate,
            spk_list=spk_list,
            collar=collar,
        )

    @property
    def weight(self):
        """Class/task weights

        Returns
        -------
        weight : None or `torch.Tensor`
        """
        return None




    def get_acc(self, prob, label, mask=None):
        non_sil = (label.sum(-1) > 0).int()
        index = prob.argmax(-1)
        pred = torch.zeros_like(label).scatter_(2,index.unsqueeze(-1),1)

        if non_sil.sum() == 0:
            acc = torch.tensor(0)
        else:
            if mask is None:
                correct = torch.sum((pred + label) > 1)
            else:
                correct = torch.sum(((pred + label) > 1)*mask)
                non_sil = non_sil * mask[:,:,0]

            acc = correct/non_sil.sum()
        #acc = torch.tensor(0)
        return acc

    def get_ser(self, prob, label, mask, not_padding_after_cif):
        prob = prob.detach().to("cpu").numpy()
        label = label.detach().to("cpu").numpy()
        not_padding_after_cif = not_padding_after_cif[:,:,0].detach().to("cpu").numpy()
        mask = mask[:,:,0].detach().to("cpu").numpy()
        num = 0
        total_ref_len = 0.0
        total_error_num = 0.0
        total_sub_num = 0.0
        total_del_num = 0.0
        total_ins_num = 0.0

        preds = prob.argmax(-1)
        label = label.argmax(-1)
        for i in range(len(preds)):
            ref_len = sum(mask[i])
            hyp_len = sum(not_padding_after_cif[i])

            spk_target_char_list = [c for c in label[i][:int(ref_len)]]
            spk_prediction_char_list = [c for c in preds[i][:int(hyp_len)]]
            error_num, _, sub_num, del_num, ins_num, _ = wer.distance(spk_target_char_list, spk_prediction_char_list)

            total_ref_len += ref_len
            total_error_num += error_num
            total_sub_num += sub_num
            total_del_num += del_num
            total_ins_num += ins_num
            num += 1
        #logger.info('\n' + 'total_ref_len, sub, del, ins:' + str(total_ref_len)+ ','+str(total_sub_num)+','+str(total_del_num)+','+str(total_ins_num)+'\n')

        ser = total_error_num / total_ref_len
        return ser

    def on_train_start(self):
        """Set loss function (with support for class weights)

        loss_func_ = Function f(input, target, weight=None) -> loss value
        """

        self.task_ = self.model_.task
        
        def count_loss_func(mask, sum_a, rm_flag, count_loss_type, train_single_spk=False):
            grep_mask = (mask[:,:,0].sum(1) == 1).float()
            count = torch.sum(mask[:,:,0], dim=-1) - 1
            #return torch.mean(rm_flag * torch.abs(count-sum_a))
            if count_loss_type == 'abs/c':
                return torch.sum(torch.abs(count-sum_a)) / torch.sum(count+1)
            elif count_loss_type == 'square/c':
                return torch.sum(torch.square(count-sum_a)) / torch.sum(count+1)
            elif count_loss_type == 'abs':
                if train_single_spk:
                    return torch.mean(grep_mask*torch.abs(count-sum_a))
                else:
                    if rm_flag != None:
                        return torch.mean(rm_flag * torch.abs(count-sum_a))
                    else:
                        return torch.mean(torch.abs(count-sum_a))
            elif count_loss_type == 'square':
                return torch.mean(torch.square(count-sum_a)) 
            else:
                raise ValueError(f'wrong count loss type')

        def multi_label2multi_class(multi_label_target):
            batch_size = multi_label_target.size(0)
            length = multi_label_target.size(1)
            #pdb.set_trace()
    
            num_spk = multi_label_target.sum(-1)

            ovlp = num_spk + 135

            one_spk = multi_label_target.argmax(-1)

            multi_class_target = torch.where(num_spk > 1,
                                    ovlp.long(),
                                    one_spk)
            multi_class_target = torch.where(num_spk == 0,
                                    torch.full([batch_size, length], 136).cuda(),
                                    multi_class_target)
            return multi_class_target
            
        def focal_loss(inputs, target, alpha=0.25, gama=2):
            zeros = torch.zeros_like(inputs).cuda()
            pos_p_sub = torch.where(target > zeros, target - inputs, zeros)
            neg_p_sub = torch.where(target > zeros, zeros, inputs)
            per_cross_entry = -alpha*(pos_p_sub**gama)*torch.log(torch.clamp(inputs, 1e-8, 1.0))-(1-alpha)*(neg_p_sub**gama)*torch.log(torch.clamp(1-inputs,1e-8,1.0))
            return per_cross_entry

        def loss_func(loss_cfg, input, prob, target, frame_target, change_p, weight_keep, fired_flag, speaker_embedding, sum_a, weight=None, mask=None, not_padding_after_cif=None, log_var=None, aux_change=None):
            loss_dict = dict()
            if loss_cfg["using_bce_weight"]:
                target_weight = target
                frame_target_weight = frame_target
            
            else:
                target_weight = None
                frame_target_weight = None

            if loss_cfg["train_single_spk"]:
                grep_mask = (mask[:,:,0].sum(1) == 1).float().unsqueeze(-1).unsqueeze(-1).repeat(1, mask.size(1), mask.size(2))
                mask = mask * grep_mask                


            # cfg = dict()
            # if config_yml.exists():
            # with open(config_yml, "r") as fp:
            #     cfg = yaml.load(fp, Loader=yaml.SafeLoader)
            #DPCL Loss
            if loss_cfg["using_DPCL"]:
                speaker_embedding_ = speaker_embedding.transpose(1,2)
                frame_target_ = frame_target.transpose(1,2)
                mask_sil = torch.ones(frame_target.size(0), frame_target.size(1), frame_target.size(1)).cuda()
                mask_sil = torch.where(torch.sum(frame_target, -1).unsqueeze(-1) == 0.0,
                        torch.zeros(frame_target.size(0), frame_target.size(1), frame_target.size(1)).cuda(),
                        mask_sil)
                mask_sil = mask_sil.transpose(1,2)
                mask_sil = torch.where(torch.sum(frame_target, -1).unsqueeze(-1) == 0.0,
                        torch.zeros(frame_target.size(0), frame_target.size(1), frame_target.size(1)).cuda(),
                        mask_sil)
                mask_sil = mask_sil.transpose(1,2)
                DPCL = (torch.matmul(speaker_embedding, speaker_embedding_)-torch.matmul(frame_target, frame_target_))*mask_sil
                if mask_sil.sum() == 0:
                    DPCL = torch.tensor(0).cuda()
                else:
                    DPCL = torch.norm(DPCL)**2 / mask_sil.sum()

                loss_dict['reg_dpcl_loss'] = DPCL
            else:
                DPCL = torch.tensor(0).cuda()

            if loss_cfg["using_frame_spk"]:
                frame_acc = self.get_acc(prob, frame_target)
                frame_spk_loss = F.binary_cross_entropy(
                    prob, frame_target, weight=frame_target_weight, reduction="mean"
                )
                loss_dict['cls_frame_spk_loss'] = frame_spk_loss
            else:
                frame_spk_loss = torch.tensor(0)
                frame_acc = torch.tensor(0)
            # 这里的权重weight是调节输出结点中每一个节点在loss计算中的重要程度的

            if loss_cfg["using_change_p"]:
                if loss_cfg["change_loss_method"] == 'bce':
                    change_p_loss = F.binary_cross_entropy(
                        weight_keep, change_p , weight=None, reduction="mean"
                    )

                elif loss_cfg["change_loss_method"] == 'edu':
                    change_p_loss = torch.mean((weight_keep-change_p)**2)
                else:
                    raise ValueError('wrong setting of change_loss_method')
                loss_dict['cls_change_p_loss'] = change_p_loss
            else:
                change_p_loss = torch.tensor(0)


            if aux_change != None:
                #pdb.set_trace()
                aux_change = aux_change.view(-1, aux_change.size(-1))
                change_p_for_ce = change_p.view(-1).long()
                aux_change_loss = F.cross_entropy(
                        aux_change, change_p_for_ce , weight=None, reduction="mean"
                )
                loss_dict['cls_aux_change_loss'] = aux_change_loss
            else:
                loss_cfg["aux_change_weight"] = 0 
                aux_change_loss = torch.tensor(0)

            if loss_cfg["using_fired_loss"]:
                fired_flag = fired_flag[:,:,-1].unsqueeze(-1)
                fired_flag.requires_grad_(False)
                fired_loss = ((weight_keep-fired_flag)**2).sum()/(weight_keep.size(0)*weight_keep.size(1))
                loss_dict['reg_fired_loss'] = fired_loss
            else:
                fired_loss = torch.tensor(0)

            # compute fired number
            total_fired = fired_flag[:,:,-1].sum()
            max_fired = torch.max(fired_flag[:,:,-1].sum(-1))
            min_fired = torch.min(fired_flag[:,:,-1].sum(-1))

            def filter_len1(mask, keep_rate):
                l = mask.size(1)
                hidden_size = mask.size(2)
                mask_ = mask[:,:,0]
                change_sample = (mask_.sum(-1) != 1).int()
                no_change_sample = (mask_.sum(-1) == 1).int()

                rand = torch.rand(change_sample.size()).cuda()
                rand = (rand < keep_rate).int()
                rm_flag = change_sample + no_change_sample * rand
                return mask*rm_flag.unsqueeze(-1).unsqueeze(-1).repeat(1, l, hidden_size), rm_flag


            if loss_cfg.get("filter_no_change_sample", False):
                mask, rm_flag = filter_len1(mask, loss_cfg.get("no_change_sample_keep_rate", 0.0))
            else:
                rm_flag = None
            
            if loss_cfg["using_count_loss"]:
                count_loss = count_loss_func(mask, sum_a, rm_flag, loss_cfg["count_loss_type"], loss_cfg["train_single_spk"])
                loss_dict['reg_count_loss'] = count_loss
            else:
                count_loss = torch.tensor(0)
            if loss_cfg["spk_loss_type"] == 'bce':
                if mask is None:
                    spk_loss = F.binary_cross_entropy(
                        input, target, weight=target_weight, reduction="mean"
                    )
                else:
                    spk_loss = torch.sum(
                        mask
                        * F.binary_cross_entropy(
                            input, target, weight=target_weight, reduction="none"
                        )
                    ) / (mask.sum() + 1e-8)

                acc = self.get_acc(input, target, mask)
                ser = self.get_ser(input, target, mask, not_padding_after_cif)

            elif loss_cfg["spk_loss_type"] == 'focal_bce':
                if mask is None:
                    spk_loss = focal_loss(input, target)
                    spk_loss = spk_loss.mean()
                else:
                    spk_loss = focal_loss(input, target)
                    spk_loss = torch.sum(mask * spk_loss) / (mask.sum() + 1e-8)

                acc = self.get_acc(input, target, mask)
                ser = self.get_ser(input, target, mask, not_padding_after_cif)

            elif loss_cfg["spk_loss_type"] == 'softmax':
                target = multi_label2multi_class(target)
                if mask is None:
                    spk_loss = F.nll_loss(input.view((-1, input.size(-1))), target.contiguous().view((-1,)), weight=weight, reduction="mean")
                else:
                    spk_loss = torch.sum(
                        mask[:,:,0].contiguous().view((-1,))
                        * F.nll_loss(input.view((-1, input.size(-1))), \
                        target.contiguous().view((-1,)),\
                         weight=weight, reduction="none" \
                         ) 
                    ) / (mask[:,:,0].sum() + 1e-8)
                
                pred = input.argmax(-1)
                correct = ((pred == target)*mask[:,:,0]).sum()
                acc = correct / (mask[:,:,0].sum() + 1e-8)
            else:
                raise ValueError(f'wrong spk loss type')

            loss_dict['cls_spk_loss'] = spk_loss

            if not loss_cfg['uncertainty_weight']:
                weighted_loss = dict()
                weighted_loss['spk_loss'] = loss_cfg["spk_loss_weight"] * spk_loss
                weighted_loss['count_loss'] = loss_cfg["count_weight"] * count_loss
                weighted_loss['DPCL'] = loss_cfg["DPCL_weight"] * DPCL
                weighted_loss['frame_spk_loss'] = loss_cfg["frame_spk_weight"] * frame_spk_loss
                weighted_loss['change_p_loss'] = loss_cfg["change_p_weight"] * change_p_loss
                weighted_loss['fired_loss'] = loss_cfg["fired_loss_weight"] * fired_loss
                weighted_loss['aux_change_loss'] = loss_cfg["aux_change_weight"] * aux_change_loss

                loss = weighted_loss['spk_loss'] + weighted_loss['count_loss'] + weighted_loss['DPCL'] + \
                        weighted_loss['frame_spk_loss'] + weighted_loss['change_p_loss'] + \
                        weighted_loss['fired_loss'] + weighted_loss['aux_change_loss']
                print_var = weighted_loss
                print_var["total_fired"] = total_fired
                print_var["max_fired"] = max_fired
                print_var["min_fired"] = min_fired
                return loss, print_var, acc, frame_acc, ser
            else:
                var = torch.exp(-log_var)
                weighted_loss = dict()
                print_var = dict()
                assert len(loss_dict) == var.size(0)
                loss_sum = 0
                for i, loss in enumerate(loss_dict.keys()):
                    if loss.startswith('cls_'):
                        weighted_loss[loss] = 2 * var[i] * loss_dict[loss]
                        var_name = loss + '_weight'
                        print_var[var_name] = 2 * var[i]
                    elif loss.startswith('reg_'):
                        weighted_loss[loss] = var[i] * loss_dict[loss]
                        var_name = loss + '_weight'
                        print_var[var_name] = var[i]
                    else:
                        ValueError(f'wrong loss name')
                #pdb.set_trace()
                loss = sum(weighted_loss.values()) + log_var.sum()
                print_var.update(weighted_loss)
                return loss, print_var, acc, frame_acc, ser

        self.loss_func_ = loss_func

    def process_sil_rep(self, y):
        batch_size = y.shape[0]
        length = y.shape[1]
        num_class = y.shape[2]
        y_ = []
        #change_p_ = []
        max_len = 0
        #pdb.set_trace()
        for i in range(batch_size):
            cur_y = y[i]
            mask = np.sum(np.abs(cur_y), -1) != 0
            cur_y_ = cur_y[mask]
            # all sil
            #change_p = np.sum(np.abs(np.diff(cur_y, axis=0)), axis=1, keepdims=True)
            #change_p = np.vstack(([[False]], change_p > 0)).astype(float)
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
            #change_p_.append(np.expand_dims(change_p,0))

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
        #change_p_ = np.concatenate(change_p_, axis=0)
        mask = np.expand_dims(mask, -1).repeat(y_o.shape[-1], -1)

        return y_o, mask


    def process_rep(self, y):
        batch_size = y.shape[0]
        length = y.shape[1]
        num_class = y.shape[2]
        y_ = []
        #change_p_ = []
        max_len = 0
        #pdb.set_trace()
        for i in range(batch_size):
            cur_y = y[i]

            index = np.sum(np.abs(np.diff(cur_y, axis=0)), axis=1, keepdims=True)

            #change_p = np.vstack(([[False]], index > 0)).astype(float)

            index = np.vstack(([[True]], index > 0)).squeeze(-1)
            index = np.where(index == True)
            cur_y = cur_y[index]
            max_len = max([max_len, len(cur_y)])
            y_.append(cur_y)
            #change_p_.append(np.expand_dims(change_p,0))

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
        
        #change_p_ = np.concatenate(change_p_, axis=0)
        mask = np.expand_dims(mask, -1).repeat(y_o.shape[-1], -1)

        return y_o, mask


    def process_y(self, y):

        batch_size = y.shape[0]
        length = y.shape[1]
        num_class = y.shape[2] - 1
        y_ = []
        change_p_ = []
        max_len = 0
        change_p = y[:,:,-1]
        y = y[:,:,:-1]
        #pdb.set_trace()
        for i in range(batch_size):
            cur_y = y[i]
            cur_change_p = change_p[i]
            cur_change_p_shift = np.append(cur_change_p[1:], 0)

            cur_y_ = cur_y[cur_change_p_shift==1]

            # all sil
            if cur_y_.shape[0] == 0:
                cur_y =np.expand_dims(cur_y[0], 0)
            else:
                cur_y = cur_y_

            max_len = max([max_len, len(cur_y)])
            y_.append(cur_y)

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
        
        mask = np.expand_dims(mask, -1).repeat(y_o.shape[-1], -1)

        return y_o, mask


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

    
    def index2chunk(self, avg_index):
        chunk_list = []
        max_flag = -1
        l = avg_index.size(0)
        for i in range(l):
            tmp_index = []
            if avg_index[i] <= max_flag:
                continue
            else:
                max_flag = avg_index[i]
                start = avg_index[i]
                while(i < l-1 and avg_index[i+1] == avg_index[i] + 1):
                    i += 1
                    max_flag = avg_index[i]
                end = max_flag
                chunk_list.append([start, end])
        return chunk_list

    def dynamic_fix_fX(self, fX, not_padding_after_cif, y, mask):
        batch_size = fX.shape[0]
        dim = fX.size(2)
        output_len = not_padding_after_cif.sum(-1)
        label_len = mask[:,:,0].sum(-1)
        avg_times = (output_len - label_len).int()
        fX1 = fX[:,:-1,:]
        fX2 = fX[:,1:,:]
        log_fX1 = torch.log(fX1)
        log_fX2 = torch.log(fX2)
        cross = - (fX1 * log_fX2 + fX2 * log_fX1) / 2
        #pdb.set_trace()
        cross = cross.sum(-1) * not_padding_after_cif[:,1:]
        new_fX = torch.zeros([0, y.size(1), dim]).cuda()
        #pdb.set_trace()
        
        for i in range(batch_size):
            fXi = fX[i]
            if avg_times[i] > 0:
                new_fXi = torch.zeros([0, dim]).cuda()
                #pdb.set_trace()
                _, avg_index = torch.topk(cross[i], avg_times[i].int().item())
                avg_index, _ = torch.sort(avg_index, dim=-1) 
                chunk_list = self.index2chunk(avg_index)
                #avg_flag = torch.ones([fXi.size(0)])
                new_fXi = torch.cat([new_fXi, fXi[:chunk_list[0][0]]], 0)
                for j in range(len(chunk_list)):
                    chunk = fXi[chunk_list[j][0]:chunk_list[j][1]+2]
                    avg = torch.mean(chunk, 0).unsqueeze(0)
                    new_fXi = torch.cat([new_fXi, avg], 0)
                    if j < len(chunk_list)-1:
                        gap = fXi[chunk_list[j][1]+2:chunk_list[j+1][0]]
                    else:
                        gap = fXi[chunk_list[j][1]+2:output_len[i]]

                    new_fXi = torch.cat([new_fXi, gap], 0)
                
                fXi = new_fXi
                    # # avg j-th avg time step
                    # #fXi_tmp = fXi.detach()
                    # tmp = (fXi[avg_index[j]] + avg_flag[avg_index[j]+1] * fXi[avg_index[j]+1]) / (avg_flag[avg_index[j]+1] + 1)
                    # fXi[avg_index[j]] = tmp
                    # avg_flag[avg_index[j]] += avg_flag[avg_index[j]+1]
                    # # dicard (j+1)-th time step 
                    # fXi = fXi[torch.arange(fXi.size(0)).cuda()!=avg_index[j]+1]

                if fXi.size(0) < y.size(1):
                    pad_len = y.size(1) - fXi.size(0)
                    pad = torch.zeros([pad_len, dim]).cuda()
                    fXi = torch.cat([fXi, pad], 0)
                else:
                    fXi = fXi[:y.size(1)]
                fXi = fXi.unsqueeze(0)
                       
            else:
                pad_len = y.size(1) - output_len[i]
                pad = torch.zeros([1, pad_len, dim]).cuda()
                fXi = torch.cat([fXi[:output_len[i]].unsqueeze(0), pad], 1)
                
            new_fX = torch.cat([new_fX, fXi], 0)
        return new_fX, mask


    def batch_loss(self, batch):
        """Compute loss for current `batch`

        Parameters
        ----------
        batch : `dict`
            ['X'] (`numpy.ndarray`)
            ['y'] (`numpy.ndarray`)
            ['mask'] (`numpy.ndarray`, optional)

        Returns
        -------
        batch_loss : `dict`
            ['loss'] (`torch.Tensor`) : Loss
        """

        # forward pass
        loss_cfg = self.model_.loss_cfg
        down_rate = self.model_.down_rate


        if not loss_cfg["pretraining"]:
            #elif self.task_.is_multilabel_classification or self.task_.is_regression:
            if loss_cfg["generate_target_y"] == 'rmsil_rmrep':
                #自己写函数去除静音，去除重复标签
                frame_target = batch["y"][:,:,:-1]
                target, mask = self.process_sil_rep(frame_target)
            elif loss_cfg["generate_target_y"] == 'rmrep':
                #只出去重复的标签，保留静音的部分，相当于把静音当作一个独立的说话人
                frame_target = batch["y"][:,:,:-1]
                target, mask = self.process_rep(frame_target)
            elif loss_cfg["generate_target_y"] == 'origin':
                # 复用原本代码中找change点的函数，只保留静音间隔比较短的说话人change点
                target, mask = self.process_y(batch["y"])
                
            else:
                msg = (
                    f"error value of generate_target_y"
                )
                raise ValueError(msg)

            change_p =  batch["y"][:,:,-1]
            collar_ = 50 * self.collar / down_rate + 1
            reset = change_p[:,0]==1
            for i in range(0, int(collar_)):
                p = change_p[:,i]
                reset = reset & (p == 1)
                if reset.sum()==0:
                    break 
                else:
                    change_p[:,i] = change_p[:,i] - p * reset.astype(change_p.dtype)


            change_p = np.expand_dims(change_p, -1)           

            target = torch.tensor(target, dtype=torch.float32, device=self.device_)
            change_p = torch.tensor(change_p, dtype=torch.float32, device=self.device_)
            frame_target = torch.tensor(frame_target, dtype=torch.float32, device=self.device_)
            X = torch.tensor(batch["X"], dtype=torch.float32, device=self.device_)
            mask = torch.tensor(mask, dtype=torch.float32, device=self.device_)

            # if loss_cfg["grep_single_spk"]:
            #     X, target, frame_target, change_p, mask = self.grep_single_spk_func(X, target, frame_target, change_p, mask)

            # if self.collar < 0.3:
            #     _, fX, not_padding_after_cif, sum_a, weight_keep, fired_flag, speaker_embedding, loss_cfg, prob, _, log_var = self.model_(X)
            # else:
            _, fX, not_padding_after_cif, sum_a, weight_keep, fired_flag, speaker_embedding, loss_cfg, prob, change_p, log_var, aux_change = self.model_(X, change_p, mask)
            
            #if loss_cfg["cif_type"] != 'cif3':
            #pdb.set_trace()
            if loss_cfg["dynamic_fix"] == True:
                fX, not_padding_after_cif = self.dynamic_fix_fX(fX, not_padding_after_cif, target, mask)
            if loss_cfg["force_al"]:
                fX, not_padding_after_cif = self.fix_fX(fX, not_padding_after_cif, target)

            not_padding_after_cif = not_padding_after_cif.unsqueeze(-1).repeat(1, 1, mask.size(-1))
            
            if "mask" in batch:
                mask = torch.tensor(
                    batch["mask"], dtype=torch.float32, device=self.device_
                )

            weight = self.weight
            if weight is not None:
                weight = weight.to(device=self.device_)


            loss, print_var, acc, frame_acc, ser = self.loss_func_(
                    loss_cfg, fX, prob, target, \
                    frame_target, change_p, \
                    weight_keep, fired_flag, \
                    speaker_embedding=speaker_embedding, 
                    sum_a=sum_a, 
                    weight=weight, 
                    mask=mask, 
                    not_padding_after_cif=not_padding_after_cif, 
                    log_var=log_var,
                    aux_change=aux_change,
                    )
            #ret = dict()
            ret = print_var
            ret['loss'] = loss
            ret['acc'] = acc
            ret['frame_acc'] = frame_acc
            ret['ser'] = ser
            return ret

        else:
            X = torch.tensor(batch["X"], dtype=torch.float32, device=self.device_)
            prob = self.model_(X)
            frame_target = batch["y"][:,:,:-1]
            frame_target = torch.tensor(frame_target, dtype=torch.float32, device=self.device_)
            frame_acc = self.get_acc(prob, frame_target)
            frame_spk_loss = F.binary_cross_entropy(
                prob, frame_target, weight=None, reduction="mean"
            )
            loss = frame_spk_loss
            return {
                "loss": loss,
                "frame_acc": frame_acc
            }
 

    @property
    def task(self):
        return Task(
            type=TaskType.MULTI_CLASS_CLASSIFICATION, output=TaskOutput.SEQUENCE
        )
