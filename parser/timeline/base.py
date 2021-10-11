#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2017 CNRS

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

from abc import abstractmethod
from pyannote.parser.base import Parser

from pyannote.core import Timeline
from pyannote.core import PYANNOTE_URI, PYANNOTE_SEGMENT

import pandas


class TimelineParser(Parser):

    @abstractmethod
    def fields(self):
        pass

    @abstractmethod
    def get_segment(self, row):
        pass

    def converters(self):
        return None

    def comment(self):
        return None

    def read(self, path, uri=None, **kwargs):

        # load whole file
        df = pandas.read_table(path,
                               delim_whitespace=True,
                               header=None, names=self.fields(),
                               comment=self.comment(),
                               converters=self.converters(),
                               keep_default_na=False, na_values=[],
                               dtype={PYANNOTE_URI: object})

        # remove comment lines
        # (i.e. lines for which all fields are either None or NaN)
        keep = [not all(pandas.isnull(item) for item in row[1:])
                for row in df.itertuples()]
        df = df[keep]

        # add 'segment' column build from start time & duration
        df[PYANNOTE_SEGMENT] = [self.get_segment(row)
                                for row in df.itertuples()]

        # add uri column in case it does not exist
        if PYANNOTE_URI not in df:
            if uri is None:
                raise ValueError('missing uri -- use uri=')
            df[PYANNOTE_URI] = uri

        # obtain list of resources
        uris = list(df[PYANNOTE_URI].unique())

        self._loaded = {}

        # loop on resources
        for uri in uris:

            # filter based on resource
            df_ = df[df[PYANNOTE_URI] == uri]

            t = Timeline.from_df(df_, uri=uri)
            self._loaded[uri, None] = t

        return self

    def empty(self, uri=None, **kwargs):
        return Timeline(uri=uri)

    def write(self, timeline, f, uri=None, **kwargs):
        """

        Parameters
        ----------
        timeline : `Timeline`
            Timeline
        f : file handle
            Default is stdout.
        uri, modality : str, optional
            Override `timeline` attributes

        """

        if uri is None:
            uri = timeline.uri

        self._append(timeline, f, uri)

    def _append(self, timeline, f, uri):
        raise NotImplementedError('')
