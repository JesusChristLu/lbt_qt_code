# -*- coding: utf-8 -*-

# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/12/15
# __author:       YangChao Zhao
from typing import Any

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon, QAction, QActionGroup
from PySide6.QtWidgets import QMenu


def bind_action(menu: QMenu, action_name: Any, icon_name: str = "") -> QAction:
    """ Bind action for QMenu.

    Args:
        menu: QMenu object.
        action_name: Action name.
        icon_name: Resource icon name.

    Returns: New action.
    """
    action = menu.addAction(action_name)
    if icon_name:
        icon = QIcon()
        icon.addFile(icon_name, QSize(), QIcon.Normal, QIcon.Off)
        action.setIcon(icon)
    return action


def bind_check_group_action(menu: QMenu, action_name: Any, group: QActionGroup, checked: bool = False):

    action = menu.addAction(action_name)
    action.setCheckable(True)
    if checked:
        action.setChecked(True)
    menu.addAction(action)
    group.addAction(action)
    return action


def bind_menu_action(menu: QMenu, action_menu: QMenu, icon_name: str = ""):
    if icon_name:
        icon = QIcon()
        icon.addFile(icon_name, QSize(), QIcon.Normal, QIcon.Off)
        action_menu.setIcon(icon)
    menu.addAction(action_menu.menuAction())
    return action_menu
