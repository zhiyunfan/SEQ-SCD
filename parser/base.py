#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2012-2015 CNRS

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

from __future__ import unicode_literals

import six
from abc import ABCMeta, abstractmethod


class Parser(object):

    __metaclass__ = ABCMeta

    @classmethod
    @abstractmethod
    def file_extensions(cls):
        pass

    @abstractmethod
    def read(self, path, **kwargs):
        pass

    @abstractmethod
    def empty(self, uri=None, modality=None, **kwargs):
        pass

    def __get_uris(self):
        return sorted(set([v for (v, m) in self._loaded]))
    uris = property(fget=__get_uris)
    """"""

    def __get_modalities(self):
        return sorted(set([m for (v, m) in self._loaded]))
    modalities = property(fget=__get_modalities)
    """"""

    def __call__(self, uri=None, modality=None, **kwargs):

        match = dict(self._loaded)

        # filter out all annotations
        # but the ones for the requested resource
        if uri is not None:
            match = {(v, m): ann for (v, m), ann in six.iteritems(match)
                     if v == uri}

        # filter out all remaining annotations
        # but the ones for the requested modality
        if modality is not None:
            match = {(v, m): ann for (v, m), ann in six.iteritems(match)
                     if m == modality}

        if len(match) == 0:
            A = self.empty(uri=uri, modality=modality, **kwargs)

        elif len(match) == 1:
            A = list(match.values())[0]

        else:
            msg = 'Found more than one matching annotation: %s'
            raise ValueError(msg % match.keys())

        return A
