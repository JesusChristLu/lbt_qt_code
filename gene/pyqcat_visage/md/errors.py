# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage
#
# Copyright (c) 2017-2030 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/10/12
# __author:       Lang Zhu
# __corporation:  OriginQuantum
"""
Exception module.
"""


class MdErr(Exception):
    reason = "Md Exception"

    def __repr__(self):
        return f"{self.__class__.__name__}\nreason:{self.reason}"


class IMGTypeErr(MdErr):
    reason = "image type is not support to converter."
