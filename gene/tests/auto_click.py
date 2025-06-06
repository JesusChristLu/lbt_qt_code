# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/1/20
# __author:       YangChao Zhao

"""
Auto Test
"""

import random
import time

import pyautogui as ui
import win32api
import win32con
from pymouse import PyMouse

ui.FAILSAFE = False

exp_list = [
    'CavityFreqSpectrum',
    'QubitSpectrum',
    'RabiScanWidth',
    'RabiScanAmp',
    'ReadoutFreqCalibrate',
    'ReadoutPowerCalibrate',
    'SingleShot',
    'QubitFreqCalibration',
    'DetuneCalibration',
    'AmpOptimize',
    'T1',
]


def position_catch():
    m = PyMouse()
    a = m.position()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ",当前鼠标位置坐标:", a)


def create_context():
    m = PyMouse()
    m.click(464, 207, 1)
    time.sleep(0.5)
    m.click(475, 242, 1)
    time.sleep(0.5)
    m.click(1057, 386, 1)
    time.sleep(0.5)
    c = 400 + int(130 * random.random())
    m.click(947, c, 1)
    time.sleep(0.5)
    m.click(1147, 475, 1)


def click_system_save():
    m = PyMouse()
    m.click(1143, 819, 1)
    time.sleep(0.5)
    m.click(826, 529, 1)
    time.sleep(1.5)
    m.click(968, 530, 1)


def simulator_keyboard(m, name: str = 'A'):
    m.click(342, 170, 1)
    for n in name:
        asc = ord(n.upper())
        win32api.keybd_event(asc, 0, 0, 0)
        win32api.keybd_event(86, 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.2)
    simulator_enter()
    simulator_enter()
    time.sleep(0.5)
    m.click(119, 238, 1)
    time.sleep(0.5)
    m.click(33, 82, 1)
    time.sleep(60)


def simulator_enter():
    win32api.keybd_event(13, 0, 0, 0)
    win32api.keybd_event(86, 0, win32con.KEYEVENTF_KEYUP, 0)


def simulator_single_qubit_calibration():
    m = PyMouse()
    m.click(159, 168, 1)
    for exp in exp_list:
        time.sleep(0.5)
        simulator_keyboard(m, exp)


if __name__ == '__main__':

    while True:
        # position_catch()
        # create_context()
        # click_system_save()
        simulator_single_qubit_calibration()
        time.sleep(2)
