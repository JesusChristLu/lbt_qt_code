# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage
#
# Copyright (c) 2017-2030 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/10/18
# __author:       Lang Zhu
# __corporation:  OriginQuantum
"""
Parser module.
"""

from pyqcat_visage.md.parser.parser import Parser
from pyqcat_visage.md.parser.parser_experiment import ExperimentParser
from pyqcat_visage.md.parser.parser_dag import DagParser

parser_dict = {
    "dag": DagParser,
    "exp": ExperimentParser,
}

__all__ = ["Parser", "ExperimentParser", "DagParser", "parser_dict"]