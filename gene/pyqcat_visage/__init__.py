# -*- coding: utf-8 -*-

# This code is part of pyqcat-legend.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/30
# __author:       XuYao

VERSION = (0x00, 0x04, 0x0a)


def get_version():
    """Return the VERSION as a string.

    For example, if `VERSION == (1, 11, 1)`, return '1.11.1'.
    """
    return '.'.join(map(str, VERSION))


__version__ = get_version()
