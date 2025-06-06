# -*- coding: utf-8 -*-

# This code is part of pyQCat Visage.
#
# Copyright (c) 2022-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/09/16
# __author:       SS Fang

"""
Define some common Options class.
"""

from pyQCat.structures import Options


class RunOptions:
    """RunOptions class"""

    _run_options = Options()

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default options values: meth:`run` method."""
        return Options()

    @property
    def run_options(self) -> Options:
        """Return options values: meth:`run` method."""
        return self._run_options

    def set_run_options(self, **fields):
        """Set options values for: meth:`run` method.

        Args:
            fields: The fields to update the options
        """
        for field in fields:
            if field not in self._run_options:
                raise AttributeError(
                    f"Options field {field} is not valid for {type(self).__name__}"
                )
        self._run_options.update(**fields)
