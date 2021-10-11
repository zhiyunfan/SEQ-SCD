#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2015 CNRS

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

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import os
import sys
from pkg_resources import iter_entry_points


__all__ = []

ParserPlugins = {}

# iterate over parser plugins
for o in iter_entry_points(group='pyannote.parser.plugin', name=None):

    # obtain parser class name (e.g. "MDTMParser")
    parser_name = o.name

    # obtain parser class (e.g. MDTMParser)
    parser_class = o.load()

    # iterate over supported file extensions
    # (e.g. ['mdtm'] for MDTMParser)
    for extension in parser_class.file_extensions():

        # if extension is already registered by another parser plugin
        # raise an exception
        if extension in ParserPlugins:

            msg = 'Extension {e} is registered by both {p1} and {p2}'
            raise ValueError(
                msg.format(
                    e=extension,
                    p1=parser_class.__name__,
                    p2=ParserPlugins[extension].__name__))

        # otherwise, register the extension with the parser class
        ParserPlugins[extension] = parser_class

    # import parser class at package root
    # (e.g. pyannote.parser.MDTMParser)
    setattr(sys.modules[__name__], parser_name, parser_class)

    # make sure parser class is imported with
    # >>> from pyannote.parser import *
    __all__.append(parser_name)


class MagicParser(object):
    """Magic parser chooses which parser to use based on file extension

    Notes
    -----
    kwargs are passed to selected parser

    """

    @staticmethod
    def get_parser(extension):
        try:
            Parser = ParserPlugins[extension]
        except Exception:
            msg = 'Extension "{e}" is not supported.'
            raise NotImplementedError(msg.format(e=extension))
        return Parser

    @staticmethod
    def guess_parser(path):

        # obtain file extension (without leading .)
        _, extension = os.path.splitext(path)
        extension = extension[1:]

        return MagicParser.get_parser(extension)

    def __init__(self, **kwargs):
        super(MagicParser, self).__init__()
        self.init_kwargs = kwargs

    def read(self, path, **kwargs):

        # obtain file extension (without leading .)
        _, extension = os.path.splitext(path)
        extension = extension[1:]

        # obtain parser based on the extension
        Parser = self.get_parser(extension)

        # initialize parser
        parser = Parser(**self.init_kwargs)

        # read file
        parser.read(path, **kwargs)

        # return parser with file loaded internally
        return parser


__all__.append(str('MagicParser'))
