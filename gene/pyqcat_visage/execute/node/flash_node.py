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
Node class.
"""

from copy import deepcopy
from typing import List, Dict

from pyqcat_visage.options import Options, RunOptions

from ..experiment.flash_exp import ExpExecutor
from ..log import logger


class Node(RunOptions):
    """Node class."""

    status_dict = {
        -1: "execute error.",
        0: "no execute",
        1: "success",
        2: "failed"
    }

    def __init__(self,
                 _id: str = None,
                 name: str = None,
                 exp_name: str = None,
                 exp_params: Dict = None,
                 adjust_params: Dict = None,
                 location: List = None):
        """Initial `Node` object."""
        self._id = _id
        self._name = name if isinstance(name, str) else exp_name
        self._exp_name = exp_name
        self._exp_params = exp_params if isinstance(exp_params, dict) else {}
        self._adjust_params = adjust_params if isinstance(adjust_params, dict) else {}
        self._location = location if isinstance(location, list) else []

        self._status = 0
        self._result = {
            "exp_id": None,
            # "exp_name": None,
            "exp_params": {},
            "status": 0,
            "result": {
                # "analysis_results": None,
                # "analysis_quality": None
            },
            "change_record": {
                # "before_execute": {},
                # "after_execute": {},
            }
        }

        # Run node middle operate options
        self._run_options = self._default_run_options()
        self.ctx_update_record = None
        self.exp_obj = None

    def __repr__(self) -> str:
        """Return description."""
        return f"<{self.__class__.__name__} {self._name} {self._status}>"

    @classmethod
    def _default_run_options(cls) -> Options:
        """Set run node operate options.

        Options:
            context: An object of context...

        """
        options = super()._default_run_options()
        options.context = None
        options.simulator = False
        options.simulator_base_path = None

        return options

    @classmethod
    def from_dict(cls, data: Dict) -> "Node":
        """From dict data create a Node object.

        Args:
            data (dict): Create node dict data.

        Returns:
            BaseDag: The dag object.
        """
        kwargs = {
            "_id": data.get("_id"),
            "name": data.get("exp_name"),
            "exp_name": data.get("exp_name"),
            "exp_params": data.get("exp_params"),
            "adjust_params": data.get("adjust_params"),
            "location": data.get("location")
        }
        exp_obj = cls(**kwargs)
        return exp_obj

    def to_dict(self) -> Dict:
        """To dict data.

        Returns:
            dict: The node object some main information to dict.
        """
        data = {
            "_id": self._id,
            "name": self._name,
            "exp_name": self._exp_name,
            "exp_params": self._exp_params,
            "adjust_params": self._adjust_params,
            "location": self._location
        }
        return data

    @property
    def id(self) -> str:
        """Return node id."""
        return self._id

    @property
    def name(self) -> str:
        """Return node name."""
        return self._name

    @property
    def status(self) -> int:
        """Return node status.

        0 not execute
        1 execute success
        2 execute failed
        """
        return self._status

    @property
    def result(self) -> Dict:
        """Return node result."""
        return self._result

    def clear(self):
        """Clear node."""
        self._result.clear()
        self._run_options = self._default_run_options()

    def adjust_exp_params(self):
        """Adjust experiment parameters."""
        new_exp_params = deepcopy(self._exp_params)
        # todo, use self._adjust_params update exp_params
        # dag dispatch
        self._exp_params = new_exp_params

    def run(self):
        """Run node."""
        exp_data = {
            "exp_name": self._exp_name,
            "exp_params": self._exp_params
        }

        exp_obj = ExpExecutor.from_dict(exp_data)
        self.exp_obj = exp_obj
        exp_obj.set_run_options(context=self.run_options.context, belong="dag", simulator=self.run_options.simulator,
                                simulator_base_path=self.run_options.simulator_base_path)
        exp_obj.run()
        self.ctx_update_record = deepcopy(exp_obj.update_params) or None

        quality = exp_obj.quality
        self._status = exp_obj.status

        results = {}
        if exp_obj.quality:
            if isinstance(quality, dict):
                for key, quality in exp_obj.quality.items():
                    results.update(
                        {
                            f"{key}_quality": f"{getattr(quality, 'descriptor', None)} -- {getattr(quality, 'value', None)}"}
                    )
            else:
                results = {
                    "analysis_quality": f"{exp_obj.quality.descriptor} -- {exp_obj.quality.value}"
                }

        if exp_obj.results:
            for key, value in exp_obj.results.items():
                if value.unit is not None:
                    result = f"{value.value} ({value.unit})"
                else:
                    result = str(value.value)
                results.update({key: result})

        change_record = deepcopy(exp_obj.update_params)
        record_move_old_value(change_record)

        self._result.update({
            "exp_id": exp_obj.id,
            "exp_name": self._exp_name,
            "exp_params": self._exp_params,
            "status": self._status,
            "result": results,
            # "change_record": {
            #     "before_execute": before_execute,
            #     "after_execute": after_execute,
            # }
            "change_record": change_record
        })


def record_move_old_value(record_dict):
    if isinstance(record_dict, dict):
        for key, value in record_dict.items():
            if isinstance(value, dict):
                record_move_old_value(value)
            elif isinstance(value, list) and len(value) == 2:
                record_dict[key] = value[1]
