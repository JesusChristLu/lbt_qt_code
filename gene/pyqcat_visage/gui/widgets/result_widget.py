# -*- coding: utf-8 -*-
# This code is part of pyqcat-monster.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2023/01/08
# __author:       YangChao Zhao

import os
import pickle
import shutil
from pickle import PickleError
from threading import Thread
from typing import TYPE_CHECKING, Union
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

import numpy as np
from PySide6.QtCore import QDir, QModelIndex, Slot, QByteArray, Qt, QSize
from PySide6.QtGui import (
    QColorSpace,
    QGuiApplication,
    QImage,
    QAction,
    QImageReader,
    QPalette,
    QPixmap,
    QIcon,
)
from PySide6.QtWidgets import QSizePolicy, QMessageBox, QLabel, QFileDialog
from minio import Minio
from minio.error import MinioException

from pyQCat.log import pyqlog
from pyQCat.structures import QDict
from pyqcat_visage.gui.file_system_ui import Ui_MainWindow
from pyqcat_visage.gui.widgets.result.file_list_model import (
    QFileListModel,
    FolderFileNode,
    DocumentFileNode,
    FileNode,
)
from pyqcat_visage.gui.widgets.result.table_model_dat import QTableModelDat
from pyqcat_visage.gui.widgets.title_window import TitleWindow

if TYPE_CHECKING:
    from ..main_window import VisageGUI


class FileSysMainWindow(TitleWindow):
    def __init__(self, gui: "VisageGUI", parent=None):
        super().__init__(parent)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.widget_2.hide()

        self.gui = gui

        self.file_tree = FolderFileNode()

        self.dat_data = {}
        self.save_type: str = ""
        self.bucket: str = ""
        self.root_path: str = ""

        self._build_model_view()
        self._build_image_area()
        self._setup_status_bar()
        self._init_context_menu()

        # set default splitter layout
        self.ui.splitter_3.setStretchFactor(0, 1)
        self.ui.splitter_3.setStretchFactor(1, 3)

        # s3 client
        self.client = None

        # history download path
        self._his_download_dir = None

    def init_file_system(self):
        if self.system_config:
            self.save_type = self.system_config.system.save_type
            self.bucket = self.system_config.system.sample
            self.root_path = self.system_config.system.local_root
            if not self.root_path.endswith("\\"):
                self.root_path += "\\"
            self.ui.type_combox.setCurrentText(self.save_type)
            self.switch(self.save_type)

    @property
    def system_config(self):
        return self.gui.backend.config

    def _init_context_menu(self):
        self.ui.listView.setContextMenuPolicy(Qt.ActionsContextMenu)

        def add_action(name: str, icon_name: str):
            action = QAction(self.ui.listView)
            action.setText(name)
            if icon_name:
                icon = QIcon()
                icon.addFile(icon_name, QSize(), QIcon.Normal, QIcon.Off)
                action.setIcon(icon)
            return action

        # menu items
        delete_option = add_action("Delete", ":/delete.png")
        delete_option.triggered.connect(self._delete_file)

        download_option = add_action("Download", ":/database-download.png")
        download_option.triggered.connect(self._download_file)

        upload_option = add_action("Upload", ":/upload.png")
        upload_option.triggered.connect(self._upload_file)

        # TODO-Lucas We don't have `copy_path.png` in this project yet. Find a `copy` icon whose style is consistent with the existing ones.
        copy_path_option = add_action("Copy path", ":/copy_path.png")
        copy_path_option.triggered.connect(self._copy_path)

        # TODO-Lucas We don't have `open_with_default_app.png` in this project yet. Find a `open` icon whose style is consistent with the existing ones.
        open_with_default_app_option = add_action("Open with default app", ":/open_with_default_app.png")
        open_with_default_app_option.triggered.connect(self._open_with_default_app)

        self.ui.listView.addAction(open_with_default_app_option)
        self.ui.listView.addAction(copy_path_option)
        self.ui.listView.addAction(download_option)
        self.ui.listView.addAction(upload_option)
        self.ui.listView.addAction(delete_option)

    def _delete_file(self):
        if self.save_type == "s3":
            return self.handler_ret_data(
                QDict(code=600, msg="Visage doesn't support deleting files on S3 currently.")
            )

        indexes = self.ui.listView.selectedIndexes()
        nodes = [self.file_model.node_from_index(index) for index in indexes]
        for node in nodes:
            file_name = self.root_path + os.path.join(*node.path)
            try:
                if isinstance(node, DocumentFileNode):
                    os.remove(file_name)
                else:
                    shutil.rmtree(file_name)
            except Exception as e:
                pyqlog.error(f"Delete {file_name} error! Because {e}")

        self.refresh_dirs()

    def _download_file(self):
        """Download s3 data to local"""

        if self.save_type != "s3":
            return self.handler_ret_data(
                QDict(code=600, msg="Visage only supports downloading files from S3.")
            )

        # get download dirname
        dirs = QFileDialog.getExistingDirectory(
            self,
            "Save As",
            self._his_download_dir or self.system_config.system.config_path,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        self._his_download_dir = dirs

        # get selected index
        indexes = self.ui.listView.selectedIndexes()
        nodes = [self.file_model.node_from_index(index) for index in indexes]

        def download_process(client, bucket, _dirs, _nodes):
            """s3 download process"""

            def is_folder(f_name: str):
                """Check file is a folder."""
                return not (
                        f_name.endswith("dat")
                        or f_name.endswith("png")
                        or f_name.endswith("txt")
                        or f_name.endswith("log")
                        or f_name.endswith("json")
                )

            def dg_load(f_name: str):
                """recurrence download file to local."""
                if is_folder(f_name):
                    if not f_name.endswith("/"):
                        f_name += "/"
                    objs = client.list_objects(bucket, f_name)
                    for obj in objs:
                        obj_name = obj.object_name
                        dg_load(obj_name)
                else:
                    local_path = dirs + "/" + f_name
                    client.fget_object(bucket, f_name, local_path)
                    pyqlog.info(f"Download {local_path} Success!")

            for node in _nodes:
                file_name = "/".join(node.path[1:])
                # Get a full object and prints the original object stat information.
                try:
                    dg_load(file_name)
                except MinioException as err:
                    pyqlog.error(err)

        thread = Thread(
            target=download_process,
            args=(self.client, self.bucket, dirs, nodes),
        )

        try:
            thread.setDaemon(True)
            thread.start()
        except Exception as e:
            return self.handler_ret_data(QDict(code=600, msg=e))

    def _upload_file(self):

        if not self._link_s3():
            return

        if self.save_type != "local":
            return self.handler_ret_data(
                QDict(code=600, msg="Visage only supports uploading local files to S3.")
            )

        dirs, ok = self.ask_input("Upload to S3", "Please input dirname:")

        # get selected index
        indexes = self.ui.listView.selectedIndexes()
        nodes = [self.file_model.node_from_index(index) for index in indexes]

        def upload_process(client, bucket, _nodes, describe: str = ""):
            """s3 download process"""

            def is_folder(f_name: str):
                """Check file is a folder."""
                return not (
                        f_name.endswith("dat")
                        or f_name.endswith("png")
                        or f_name.endswith("txt")
                        or f_name.endswith("log")
                        or f_name.endswith("json")
                )

            def dg_upload(f_name: str):
                """recurrence download file to local."""
                if is_folder(f_name):
                    for root, _, files in os.walk(f_name):
                        for file in files:
                            path = os.path.join(root, file)
                            dg_upload(path)
                else:
                    object_name = describe + "/" + f_name.split(self.root_path)[1]
                    object_name = object_name.replace("\\", "/")
                    try:
                        client.fput_object(bucket, object_name, f_name)
                        pyqlog.info(f"Upload {f_name} to {object_name} success!")
                    except MinioException as e:
                        pyqlog.error(f"Upload {f_name} to {object_name} failed! {e}")

            for node in _nodes:
                file_name = self.root_path + os.path.join(*node.path)
                # Get a full object and prints the original object stat information.
                try:
                    dg_upload(file_name)
                except MinioException as err:
                    pyqlog.error(err)

        thread = Thread(
            target=upload_process,
            args=(self.client, self.bucket, nodes, dirs),
        )

        if ok:
            thread.setDaemon(True)
            thread.start()

    def _copy_path(self):
        """Copy path to clipboard."""

        def list_to_string(lst):
            if len(lst) == 0:
                return None
            elif len(lst) == 1:
                return lst[0]
            else:
                return "\n".join(lst)

        if self.save_type != "local":
            return self.handler_ret_data(
                QDict(code=600, msg="Visage only supports copying file paths of files on your local computer.")
            )

        indexes = self.ui.listView.selectedIndexes()
        nodes = [self.file_model.node_from_index(index) for index in indexes]

        paths = []
        for node in nodes:
            path = self.root_path + os.path.join(*node.path)
            paths.append(f'"{path}"')

        text = list_to_string(paths)
        if text is not None:
            clipboard = QGuiApplication.clipboard()
            clipboard.setText(f"{text}")

    def _open_with_default_app(self):
        """Open the selected item(s) in default app(s).

        If the selected item is a folder, the default file explorer app of the current OS is used to open this
        folder.
        If the OS is Windows, then the default file explorer app is often `File Explorer`.

        The user is allowed to select **multiple** items and open them with default app(s).
        """
        if self.save_type != "local":
            return self.handler_ret_data(
                QDict(code=600, msg="Visage only supports open **local** file(s)/folder(s) with default app(s).")
            )

        indexes = self.ui.listView.selectedIndexes()
        nodes = [self.file_model.node_from_index(index) for index in indexes]

        for node in nodes:
            path = self.root_path + os.path.join(*node.path)
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _build_model_view(self):
        self.file_model = QFileListModel(self)
        self.ui.listView.setModel(self.file_model)
        self.ui.listView.doubleClicked.connect(self.change_root)

        self.data_model = QTableModelDat(self)
        self.ui.tableView.setModel(self.data_model)

    def _build_image_area(self):
        self._image_label = QLabel()
        self._image_label.setBackgroundRole(QPalette.Base)
        self._image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._image_label.setScaledContents(True)

        self.ui.scrollArea.setBackgroundRole(QPalette.Dark)
        self.ui.scrollArea.setVisible(True)
        self.ui.scrollArea.setWidget(self._image_label)

    def _setup_status_bar(self):
        self.path_label = QLabel("")
        self.ui.statusbar.addWidget(self.path_label)

    @Slot()
    def refresh_dirs(self):
        self._refresh_model()

    @Slot()
    def last_dirs(self):
        if self.gui.backend.current_dirs:
            self._find_branch(self.gui.backend.current_dirs)

    @Slot(QModelIndex)
    def change_root(self, index: QModelIndex):
        node = self.file_model.node_from_index(index)
        self._refresh_model(node)

    @Slot()
    def pre_page(self):
        node = self.file_model.root.parent
        self._refresh_model(node)

    @Slot(str)
    def switch(self, save_type: str):
        if save_type == "s3":
            if not self._link_s3():
                return
        self.save_type = save_type
        self.file_tree.children.clear()
        self.file_model.root = self.file_tree
        self._refresh_model()

    def _link_s3(self):
        try:
            if self.client is None:
                self.client = Minio(
                    endpoint=self.system_config.minio.s3_root,
                    access_key=str(self.system_config.minio.s3_access_key),
                    secret_key=str(self.system_config.minio.s3_secret_key),
                    secure=False,
                )
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
            return True
        except Exception as err:
            pyqlog.error(f"S3 error, because {err}!")
            self.ui.type_combox.setCurrentText("local")
            return False

    @Slot()
    def find_path(self):
        self._find_branch(self.ui.input_edit.text())

    def _find_branch(self, path: str):
        if self.save_type == "s3" and path.startswith(self.bucket):
            if path.endswith("/"):
                path = path[:-1]
            self.file_tree.children.clear()
            self.file_model.root = self.file_tree
            self._refresh_model()
            path_list = path.split(self.bucket)[1].split("/")[1:]
            for p in path_list:
                for child in self.file_model.root.children:
                    if child.name == p:
                        self._refresh_model(child)
                        break
                else:
                    pyqlog.error(f"Path: ({path}) is not existed!")
        elif self.save_type == "local" and path.startswith(self.root_path):
            if os.path.exists(path):
                self.file_tree.children.clear()
                self.file_model.root = self.file_tree
                self._refresh_model()
                path_list = path.split(self.root_path)[1].split("\\")
                for p in path_list:
                    for child in self.file_model.root.children:
                        if child.name == p:
                            self._refresh_model(child)
                            break
            else:
                pyqlog.error(f"Path: ({path}) is not existed!")
        else:
            if self.save_type == "s3":
                pyqlog.error(f"Path: ({path}) is not start bucket ({self.bucket})!")
            else:
                pyqlog.error(f"Path: ({path}) is not start root path ({self.root_path})!")

    def _refresh_model(self, node: Union[FolderFileNode, DocumentFileNode] = None):

        self._refresh_ui(node)

        if node is None or isinstance(node, FolderFileNode):
            self._grow_tree(node)
        else:
            self._display(node)

        self._check_pre_page()

    def _check_pre_page(self):
        if self.file_model.root.has_parent():
            self.ui.actionPre.setEnabled(True)
        else:
            self.ui.actionPre.setEnabled(False)

    def _grow_tree(self, root: FolderFileNode = None):

        if root is None:
            root = self.file_model.root
        root.children.clear()

        if self.save_type == "s3":
            prefix = "/".join(root.path[1:])
            if prefix != "":
                prefix += "/"
            self._refresh_statue(prefix)
            objs = self.client.list_objects(self.bucket, prefix)
            filenames = [obj.object_name for obj in objs]
            for fn in filenames:
                fn = fn.split("/")
                if fn[-1] == "":
                    child = FolderFileNode(fn[-2], root)
                else:
                    child = DocumentFileNode(fn[-1], root)
                root.insert_child(child)
        else:
            prefix = self.root_path + os.path.join(*root.path)
            os.makedirs(prefix, exist_ok=True)
            self._refresh_statue(prefix)
            filenames = os.listdir(prefix)
            filenames.sort(key=lambda x: os.path.getmtime(os.path.join(prefix, x)))
            for fn in filenames:
                loc = os.path.join(prefix, fn)
                if os.path.isdir(loc):
                    child = FolderFileNode(fn, root)
                else:
                    child = DocumentFileNode(fn, root)
                root.insert_child(child)

        self.file_model.root = root
        self.file_model.refresh()

    def _display(self, node: DocumentFileNode):
        if self.save_type == "s3":
            file_name = "/".join(node.path[1:])
            self._refresh_statue(file_name)
            response = self.client.get_object(self.bucket, file_name)
            if response:
                obj = response.read()
                if node.name.endswith(".png"):
                    new_image = QImage()
                    new_image.loadFromData(QByteArray(obj))
                    if not new_image.isNull():
                        self._set_image(new_image)
                elif node.name.endswith(".dat"):
                    try:
                        data = pickle.loads(obj)
                    except PickleError:
                        obj = obj.decode()
                        data = [
                            [eval(d) for d in s.split("\t")]
                            for s in str(obj).split("\r\n")[:-1]
                        ]
                        data = np.array(data).T
                    if len(data.shape) != 2:
                        pyqlog.error(f"data must be 2 shape! {data.shape}")
                    self.data_model.input_data = data.T
                    self.data_model.refresh()
                elif node.name.endswith(".txt") or node.name.endswith(".log") or node.name.endswith(".json"):
                    self.ui.textEdit.setText(str(obj, "utf-8"))
                else:
                    pyqlog.error(f"Can not display {file_name}")
        else:
            file_name = self.root_path + os.path.join(*node.path)
            self._refresh_statue(file_name)
            if file_name.endswith(".dat"):
                data = np.loadtxt(file_name)
                self.data_model.input_data = data
                self.data_model.refresh()
            elif file_name.endswith(".png"):
                self._load_imag(file_name)
            elif (
                    file_name.endswith(".txt")
                    or file_name.endswith(".log")
                    or file_name.endswith(".json")
            ):
                with open(file_name, "r", encoding="utf-8") as f:
                    text = f.read()
                    self.ui.textEdit.setText(text)
            else:
                pyqlog.error(f"Can not display {file_name}")

    def _load_imag(self, filename: str):
        reader = QImageReader(filename)
        reader.setAutoTransform(True)
        new_image = reader.read()
        native_filename = QDir.toNativeSeparators(filename)
        if new_image.isNull():
            error = reader.errorString()
            QMessageBox.information(
                self,
                QGuiApplication.applicationDisplayName(),
                f"Cannot load {native_filename}: {error}",
            )
            return False

        self._set_image(new_image)

    def _set_image(self, new_image):
        self._image = new_image
        if self._image.colorSpace().isValid():
            self._image.convertToColorSpace(QColorSpace.SRgb)
        self._image_label.setPixmap(QPixmap.fromImage(self._image))
        self.ui.scrollArea.setWidgetResizable(True)

    def _refresh_ui(self, node: FileNode):
        if node is None or isinstance(node, FolderFileNode):
            self.ui.widget_2.hide()
        else:
            if (
                    node.name.endswith("txt")
                    or node.name.endswith("log")
                    or node.name.endswith("json")
            ):
                self.ui.widget_2.show()
                self.ui.textEdit.show()
                self.ui.tableView.hide()
                self.ui.scrollArea.hide()
            elif node.name.endswith("png"):
                self.ui.widget_2.show()
                self.ui.textEdit.hide()
                self.ui.tableView.hide()
                self.ui.scrollArea.show()
            elif node.name.endswith("dat"):
                self.ui.widget_2.show()
                self.ui.textEdit.hide()
                self.ui.tableView.show()
                self.ui.scrollArea.hide()

    def _refresh_statue(self, path: str):
        if self.save_type == "s3":
            path = self.bucket + "/" + path
        self.path_label.setText(path)
        self.ui.input_edit.setText(path)
