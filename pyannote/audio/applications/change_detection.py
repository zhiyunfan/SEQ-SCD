#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2019 CNRS

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
# Ruiqing YIN - yin@limsi.fr
# Hervé BREDIN - http://herve.niderb.fr

from functools import partial
import scipy.optimize
from base_labeling import BaseLabeling
from pyannote.database import get_annotated
import scipy.signal

from pyannote.audio.features import Pretrained
from pyannote.audio.pipeline.speaker_change_detection import (
    SpeakerChangeDetection as SpeakerChangeDetectionPipeline,
)
from pyannote.core.utils.numpy import one_hot_encoding
import copy
import numpy as np

from pyannote.core import *
from pyannote.core.notebook import notebook
import matplotlib.pyplot as plt
import os
import pdb
def validate_helper_func(current_file, oracle_vad, plot_flag, pipeline=None, metric=None):
    save_dir = '/opt/tiger/fanzhiyun/code/pyannote-audio/view'
    reference = current_file["annotation"]
    uem = get_annotated(current_file)
    hypothesis, fired_flag = pipeline(current_file)
    scores = current_file["scores"]
    # plot ref and hyp
    uri = current_file['uri']
    save_dir = save_dir + '/' + uri

    plot_flag = False
    if plot_flag:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        dur = 40
        total_dur = int(current_file['duration'])
        num_fig = total_dur // dur
        for i in range(num_fig):
            start = i*dur
            end = start + dur
            plt.figure()
            notebook.crop = Segment(start, end) 
            plt.subplot(311)
            notebook.plot_annotation(reference)
            plt.subplot(312)
            notebook.plot_timeline(hypothesis.get_timeline())
            plt.subplot(313)
            notebook.plot_feature(scores)
            plt.savefig(save_dir + '/' + str(start) + '_' + str(end) + '.eps', format='eps')

    return metric(reference, hypothesis, oracle_vad, uem=uem)


def vos(current_file, theta_vad, theta_overdet, spk_list):
    save_dir = '/opt/tiger/fanzhiyun/code/SEQ-SCD-book/view'
    reference = current_file["annotation"]
    seg_score = current_file["seg_score"]
    uri = current_file["uri"]
    save_dir = save_dir + '/' + uri
    hyp = Annotation()
    for item in seg_score:
        seg = item[0]
        score = item[1]
        score_sorted = copy.copy(score)
        score_sorted.sort()
        max_score = score_sorted[-1]
        second_score = score_sorted[-2]
        if max_score < theta_vad:
            continue
        else:
            if second_score > theta_overdet:
                hyp[seg] = 'over'
            else:
                spk_index = score.argmax()
                hyp[seg] = spk_list[spk_index]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    hyp = hyp.support(0.1)

    dur = 40
    total_dur = int(current_file['duration'])
    num_fig = total_dur // dur
    for i in range(num_fig):
        start = i*dur
        end = start + dur
        plt.figure()
        notebook.crop = Segment(start, end) 
        plt.subplot(311)
        notebook.plot_annotation(reference)
        plt.subplot(313)
        notebook.plot_annotation(hyp)

        plt.savefig(save_dir + '/' + str(start) + '_' + str(end) + '.eps', format='eps')



class SpeakerChangeDetection(BaseLabeling):

    Pipeline = SpeakerChangeDetectionPipeline

    def validation_criterion(self, protocol, diarization=False, **kwargs):
        if diarization:
            return "diarization_fscore"
        else:
            return "segmentation_fscore"

    def postprocess_y(self, Y: np.ndarray, resolution, collar) -> np.ndarray:
        """Generate labels for speaker change detection

        Parameters
        ----------
        Y : (n_samples, n_speakers) numpy.ndarray
            Discretized annotation returned by `pyannote.core.utils.numpy.one_hot_encoding`.

        Returns
        -------
        y : (n_samples, 1) numpy.ndarray

        See also
        --------
        `pyannote.core.utils.numpy.one_hot_encoding`
        """
        collar_ = resolution.duration_to_samples(collar)
        # window
        window_ = scipy.signal.triang(collar_)[:, np.newaxis]
        # replace NaNs by 0s
        Y = np.nan_to_num(Y)
        n_samples, n_speakers = Y.shape
        # True = change. False = no change
        y = np.sum(np.abs(np.diff(Y, axis=0)), axis=1, keepdims=True)
        y = np.vstack(([[0]], y > 0))


        # mark change points neighborhood as positive


        ##mark half collar follows change points as positive
        wid = window_.shape[0] // 2
        if wid != 0:
            window_[-wid:] = 0
        l = y.shape[0]
        y = np.minimum(1, scipy.signal.convolve(y, window_)[:l])
        # HACK for some reason, y rarely equals zero
        if not False:
            y = 1 * (y > 1e-10)

        # at this point, all segment boundaries are marked as change
        # (including non-speech/speaker changesà

        # remove non-speech/speaker change
        if not self.non_speech:

            # append (half collar) empty samples at the beginning/end
            expanded_Y = np.vstack(
                [
                    np.zeros(((collar_ + 1) // 2, n_speakers), dtype=Y.dtype),
                    Y,
                    np.zeros(((collar_ + 1) // 2, n_speakers), dtype=Y.dtype),
                ]
            )

            # stride trick. data[i] is now a sliding window of collar length
            # centered at time step i.
            data = np.lib.stride_tricks.as_strided(
                expanded_Y,
                shape=(n_samples, n_speakers, collar_),
                strides=(Y.strides[0], Y.strides[1], Y.strides[0]),
            )

            # y[i] = 1 if more than one speaker are speaking in the
            # corresponding window. 0 otherwise
            x_speakers = 1 * (np.sum(np.sum(data, axis=2) > 0, axis=1) > 1)
            x_speakers = x_speakers.reshape(-1, 1)

            y *= x_speakers
        return y


    def initialize_y(self, current_file, resolution, down_rate, collar, spk_list):
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

        labels = current_file["annotation"].labels()
        resolution_ = copy.deepcopy(resolution)
        if down_rate != 1:
            resolution_._SlidingWindow__duration = resolution_._SlidingWindow__duration + (down_rate - 1) * resolution_._SlidingWindow__step
            resolution_._SlidingWindow__step = resolution_._SlidingWindow__step * down_rate
        
        y = one_hot_encoding(
            current_file["annotation"],
            get_annotated(current_file),
            resolution_,
            labels=spk_list,
            mode="center",
            is_training=False,
        )

        y_change = self.postprocess_y(y.data, resolution_, collar)
        y.data = np.concatenate([y.data, y_change], -1)

        return y

    def validate_epoch(
        self,
        epoch,
        validation_data,
        device=None,
        batch_size=32,
        diarization=False,
        n_jobs=1,
        duration=None,
        step=0.25,
        oracle_vad=False,
        down_rate=1,
        spk_list=[],
        collar=0.1,
        writer=None,
        up_bound=1.0,
        best_threshold=None,
        **kwargs
    ):
        # compute (and store) SCD scores
        pretrained = Pretrained(
            validate_dir=self.validate_dir_,
            epoch=epoch,
            duration=duration,
            step=step,
            batch_size=batch_size,
            device=device,
            down_rate=down_rate,
            writer=writer,
        )
        for current_file in validation_data:
            resolution = pretrained.get_resolution()
            current_file["y"] = self.initialize_y(current_file, resolution, down_rate, collar, spk_list)
            scores_, seg_score = pretrained(current_file)
            scores = copy.deepcopy(scores_)
            if down_rate != 1:
                scores.sliding_window._SlidingWindow__duration = scores.sliding_window._SlidingWindow__duration + (down_rate - 1) * scores.sliding_window._SlidingWindow__step
                scores.sliding_window._SlidingWindow__step = scores.sliding_window._SlidingWindow__step * down_rate

            current_file["scores"] = scores
            current_file["seg_score"] = seg_score


        # pipeline
        pipeline = self.Pipeline(scores="@scores", fscore=True, diarization=diarization)

        def compare_timeline(hyp, ref):
            ref_gap = ref.gaps()
            hyp_gap = hyp.gaps()

            hyp_ref_union = hyp | ref
            hyp_ref_gap_union = hyp | ref_gap
            hyp_gap_ref_union = hyp_gap | ref
            #pdb.set_trace()
            hit = hyp.duration() + ref.duration() - hyp_ref_union.duration()
            false = hyp.duration() + ref_gap.duration() - hyp_ref_gap_union.duration()
            miss = hyp_gap.duration() + ref.duration() - hyp_gap_ref_union.duration()

            return hit, false, miss
            
        def fun(threshold):
            
            pipeline.instantiate({"alpha": threshold, "min_duration": 0.100})
            metric = pipeline.get_metric(parallel=True)
            validate = partial(validate_helper_func, pipeline=pipeline, metric=metric)
            if n_jobs > 1:
                _ = self.pool_.map(validate, validation_data, oracle_vad, self.plot_flag)
            else:
                for file in validation_data:
                    _ = validate(file, oracle_vad, self.plot_flag)

            return 1.0 - abs(metric)

        #### adjust the vad threshold
        def fun_vad(threshold):
            #print('ssss', threshold)
            total_hit = 0
            total_false = 0
            total_miss = 0
            for current_file in validation_data:
                #pdb.set_trace()
                reference = current_file["annotation"]
                seg_score = current_file["seg_score"]
                T = current_file["annotated"][0].end
                #pdb.set_trace()
                speech = [Segment(0, 0.001)]

                for item in seg_score:
                    seg = item[0]
                    score = item[1]
                    max_score = max(score)
                    if max_score > threshold:
                        speech.append(seg)

                speech.append(Segment(T-0.001, T))
                speech = Timeline(speech).support()

                ref = reference.get_timeline()
                hit_speech, false_speech, miss_speech = compare_timeline(speech, ref)

                total_hit = total_hit + hit_speech
                total_false = total_false + false_speech
                total_miss = total_miss + miss_speech

            return -(total_hit - total_false - total_miss)

        vad_res = scipy.optimize.minimize_scalar(
                fun_vad, bounds=(0.0, 1.0), method="bounded", options={"maxiter": 10}
            )
        theta_vad = vad_res.x.item()
        hit_speech = -vad_res.fun

        ### adjust the overlap detection threshold
        def fun_overdet(threshold):
            total_hit = 0
            total_false = 0
            total_miss = 0 
            for current_file in validation_data:
                reference = current_file["annotation"]
                seg_score = current_file["seg_score"]
                T = current_file["annotated"][0].end
                hyp = [Segment(0, 0.001)]            
                for item in seg_score:
                    
                    seg = item[0]
                    score = copy.copy(item[1])
                    score.sort()
                    second_score = score[-2]

                    if second_score > 0.5:
                       hyp.append(seg)
                #pdb.set_trace()
                hyp.append(Segment(T-0.001, T))
                hyp = Timeline(hyp).support()

                overlap = []
                labels = reference.labels()
                #pdb.set_trace()
                for label_i in labels:
                    for label_j in labels:
                        if label_j == label_i:
                            continue 
                        timeline_i = reference.label_support(label_i)
                        timeline_j = reference.label_support(label_j)
                        for segment_i, segment_j in timeline_i.co_iter(timeline_j):
                            segment_and = segment_i & segment_j
                            overlap.append(segment_and)
                #pdb.set_trace()
                overlap = Timeline(overlap).support()
                hit_overlap, false_overlap, miss_overlap = compare_timeline(hyp, overlap)
                total_hit = total_hit + hit_overlap
                total_false = total_false + false_overlap
                total_miss = total_miss + miss_overlap

            return -(total_hit - total_false - total_miss)

        overlap_res = scipy.optimize.minimize_scalar(
                fun_overdet, bounds=(0.0, 0.5), method="bounded", options={"maxiter": 10}
            )

        theta_overdet = overlap_res.x.item()
        hit_overlap = -overlap_res.fun


        for file in validation_data:
            vos(file, theta_vad, 0.5, spk_list)

        self.plot_flag = False
        if best_threshold == None:
            res = scipy.optimize.minimize_scalar(
                fun, bounds=(0.0, up_bound), method="bounded", options={"maxiter": 10}
            )
            self.plot_flag = True
            threshold = res.x.item()
            value = fun(threshold)
            return {
                "metric": self.validation_criterion(None, diarization=diarization),
                "minimize": False,
                "value": float(1.0 - res.fun),
                "pipeline": pipeline.instantiate(
                    {"alpha": threshold, "min_duration": 0.100}
                ),
            }
        else:
            value = fun(float(best_threshold))
            return {
                "metric": self.validation_criterion(None, diarization=diarization),
                "minimize": False,
                "value": float(1.0 - value),
                "pipeline": pipeline.instantiate(
                    {"alpha": best_threshold, "min_duration": 0.100}
                ),
            }
