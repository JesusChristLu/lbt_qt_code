# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/02
# __author:       HanQing Shi
"""visage async task lib."""
from typing import Union, List
from pyQCat.invoker import AsyncDataCenter
from pyqcat_visage.backend.component import VisageComponent
from functools import cmp_to_key

ASYNC_TASK_MAP = {}


def async_task_register(func_name=None):
    def wrapper(func):
        real_func_name = func_name or func.__name__
        if real_func_name not in ASYNC_TASK_MAP:
            ASYNC_TASK_MAP.update({real_func_name: func})
        return func

    return wrapper


@async_task_register("query_all")
async def query_component(backend, qid: str = None,
                          component_names: Union[str, List] = None,
                          user: str = None,
                          point_label: str = None,
                          sample: str = None,
                          env_name: str = None):
    def sort_component(c1, c2):
        # c1: VisageComponent = c1[1]
        # c2: VisageComponent = c2[1]

        if c1.style == c2.style:
            if c1.style == "qubit":
                return -1 if int(c1.name[1:]) < int(c2.name[1:]) else 1
            elif c1.style == "coupler":
                return (
                    -1
                    if int(c1.name.split("-")[0][1:])
                       < int(c2.name.split("-")[0][1:])
                    else 1
                )
            return 0
        else:
            return -1 if c1.sort_level < c2.sort_level else 1

    user = user or backend.login_user.get("username")
    sample = sample or backend.config.system.sample
    point_label = point_label or backend.config.system.point_label
    env_name = env_name or backend.config.system.env_name

    ret_data = await AsyncDataCenter().query_chip_all(qid=qid,
                                                      name=component_names,
                                                      username=user,
                                                      sample=sample,
                                                      point_label=point_label,
                                                      env_name=env_name, )
    if ret_data.get("code") == 200:
        backend._components.clear()
        for key, component_data in ret_data.get("data").items():
            name = component_data.get("name")

            if (
                    "." not in name
                    and name not in backend.view_channels.keys()
                    and component_data.get("bit_type") != "QubitPair"
            ):
                continue

            if (
                    name.endswith("dat")
                    and name.split(".")[0].split("_")[-1]
                    not in backend.view_channels.keys()
            ):
                continue

            backend._components.append(
                VisageComponent.from_dict(
                    component_data, len(ret_data.get("data").items())
                )
            )

        backend._components = sorted(backend._components, key=cmp_to_key(sort_component))
        backend.context_builder.refresh_chip_data(ret_data.get("data"))

    return ret_data
