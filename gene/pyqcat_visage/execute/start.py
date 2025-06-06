# -*- coding: utf-8 -*-
# This code is part of pyQCat Visage.
#
# Copyright (c) 2022-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# __date:         2022/11/18
# __author:       Lang Zhu
import asyncio
from pyqcat_visage.execute.network_manager import ExecuteScheduler
from pyQCat.invoker import Invoker


def async_process():
    Invoker.load_account()
    execute = ExecuteScheduler()
    asyncio.run(execute.execute())


if __name__ == "__main__":
    async_process()
