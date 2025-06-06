# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'workspace_sync_note_ui.ui'
##
## Created by: Qt User Interface Compiler version 6.4.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLayout, QLineEdit,
    QMainWindow, QPushButton, QSizePolicy, QSpacerItem,
    QSpinBox, QSplitter, QStatusBar, QTextEdit,
    QVBoxLayout, QWidget)

from .widgets.chip_manage_files.table_view_workspace_note import QTableViewWorkSpaceNoteWidget
from .widgets.chip_manage_files.tree_view_workspace_note import QTreeViewSpaceNoteWidget
from .widgets.combox_custom.combox_search import SearchComboBox


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1161, 700)
        icon = QIcon()
        icon.addFile(u":/context-edit.png", QSize(), QIcon.Normal, QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.splitter_2 = QSplitter(self.centralwidget)
        self.splitter_2.setObjectName(u"splitter_2")
        self.splitter_2.setOrientation(Qt.Horizontal)
        self.widget_2 = QWidget(self.splitter_2)
        self.widget_2.setObjectName(u"widget_2")
        self.verticalLayout_4 = QVBoxLayout(self.widget_2)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.widget = QWidget(self.widget_2)
        self.widget.setObjectName(u"widget")
        self.horizontalLayout_2 = QHBoxLayout(self.widget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, -1, -1, -1)
        self.widget_12 = QWidget(self.widget)
        self.widget_12.setObjectName(u"widget_12")
        self.verticalLayout_2 = QVBoxLayout(self.widget_12)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, -1, -1, -1)
        self.widget_6 = QWidget(self.widget_12)
        self.widget_6.setObjectName(u"widget_6")
        self.horizontalLayout_6 = QHBoxLayout(self.widget_6)
        self.horizontalLayout_6.setSpacing(6)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(-1, 0, -1, 0)
        self.label_4 = QLabel(self.widget_6)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_6.addWidget(self.label_4)

        self.UserContent = SearchComboBox(self.widget_6)
        self.UserContent.setObjectName(u"UserContent")

        self.horizontalLayout_6.addWidget(self.UserContent)

        self.horizontalSpacer_4 = QSpacerItem(91, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_4)

        self.horizontalLayout_6.setStretch(0, 2)
        self.horizontalLayout_6.setStretch(1, 6)
        self.horizontalLayout_6.setStretch(2, 3)

        self.verticalLayout_2.addWidget(self.widget_6)

        self.widget_7 = QWidget(self.widget_12)
        self.widget_7.setObjectName(u"widget_7")
        self.horizontalLayout_7 = QHBoxLayout(self.widget_7)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.label_5 = QLabel(self.widget_7)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_7.addWidget(self.label_5)

        self.SampleContent = SearchComboBox(self.widget_7)
        self.SampleContent.setObjectName(u"SampleContent")
        self.SampleContent.setEnabled(True)

        self.horizontalLayout_7.addWidget(self.SampleContent)

        self.horizontalSpacer_5 = QSpacerItem(91, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_7.addItem(self.horizontalSpacer_5)

        self.horizontalLayout_7.setStretch(0, 2)
        self.horizontalLayout_7.setStretch(1, 6)
        self.horizontalLayout_7.setStretch(2, 3)

        self.verticalLayout_2.addWidget(self.widget_7)

        self.widget_13 = QWidget(self.widget_12)
        self.widget_13.setObjectName(u"widget_13")
        self.widget_13.setEnabled(True)
        self.horizontalLayout_8 = QHBoxLayout(self.widget_13)
        self.horizontalLayout_8.setSpacing(5)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(-1, 0, -1, 0)
        self.label_6 = QLabel(self.widget_13)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_8.addWidget(self.label_6)

        self.EnvContent = SearchComboBox(self.widget_13)
        self.EnvContent.setObjectName(u"EnvContent")

        self.horizontalLayout_8.addWidget(self.EnvContent)

        self.horizontalSpacer_6 = QSpacerItem(38, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_8.addItem(self.horizontalSpacer_6)

        self.horizontalLayout_8.setStretch(0, 2)
        self.horizontalLayout_8.setStretch(1, 6)
        self.horizontalLayout_8.setStretch(2, 3)

        self.verticalLayout_2.addWidget(self.widget_13)

        self.widget_3 = QWidget(self.widget_12)
        self.widget_3.setObjectName(u"widget_3")
        self.horizontalLayout = QHBoxLayout(self.widget_3)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(-1, 0, -1, 0)
        self.label = QLabel(self.widget_3)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.nameContent = QLineEdit(self.widget_3)
        self.nameContent.setObjectName(u"nameContent")

        self.horizontalLayout.addWidget(self.nameContent)

        self.horizontalSpacer = QSpacerItem(103, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.horizontalLayout.setStretch(0, 2)
        self.horizontalLayout.setStretch(1, 6)
        self.horizontalLayout.setStretch(2, 3)

        self.verticalLayout_2.addWidget(self.widget_3)


        self.horizontalLayout_2.addWidget(self.widget_12)

        self.widget_14 = QWidget(self.widget)
        self.widget_14.setObjectName(u"widget_14")
        self.verticalLayout_5 = QVBoxLayout(self.widget_14)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalSpacer_3 = QSpacerItem(20, 15, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_5.addItem(self.verticalSpacer_3)

        self.queryButton = QPushButton(self.widget_14)
        self.queryButton.setObjectName(u"queryButton")

        self.verticalLayout_5.addWidget(self.queryButton)

        self.verticalSpacer_4 = QSpacerItem(20, 15, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_5.addItem(self.verticalSpacer_4)


        self.horizontalLayout_2.addWidget(self.widget_14)


        self.verticalLayout_4.addWidget(self.widget)

        self.groupBox_2 = QGroupBox(self.widget_2)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.verticalLayout_3 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.page_layout = QHBoxLayout()
        self.page_layout.setObjectName(u"page_layout")
        self.label_3 = QLabel(self.groupBox_2)
        self.label_3.setObjectName(u"label_3")

        self.page_layout.addWidget(self.label_3)

        self.page_spinBox = QSpinBox(self.groupBox_2)
        self.page_spinBox.setObjectName(u"page_spinBox")
        self.page_spinBox.setEnabled(True)
        self.page_spinBox.setMinimum(1)

        self.page_layout.addWidget(self.page_spinBox)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.page_layout.addItem(self.horizontalSpacer_7)

        self.label_7 = QLabel(self.groupBox_2)
        self.label_7.setObjectName(u"label_7")

        self.page_layout.addWidget(self.label_7)

        self.volume_spinBox = QSpinBox(self.groupBox_2)
        self.volume_spinBox.setObjectName(u"volume_spinBox")
        self.volume_spinBox.setEnabled(True)
        self.volume_spinBox.setMinimum(1)
        self.volume_spinBox.setMaximum(10000000)
        self.volume_spinBox.setValue(10)
        self.volume_spinBox.setDisplayIntegerBase(10)

        self.page_layout.addWidget(self.volume_spinBox)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.page_layout.addItem(self.horizontalSpacer_3)

        self.page_layout.setStretch(0, 1)
        self.page_layout.setStretch(1, 1)
        self.page_layout.setStretch(2, 1)
        self.page_layout.setStretch(3, 1)
        self.page_layout.setStretch(4, 1)
        self.page_layout.setStretch(5, 1)

        self.verticalLayout_3.addLayout(self.page_layout)

        self.table_view_context = QTableViewWorkSpaceNoteWidget(self.groupBox_2)
        self.table_view_context.setObjectName(u"table_view_context")

        self.verticalLayout_3.addWidget(self.table_view_context)


        self.verticalLayout_4.addWidget(self.groupBox_2)

        self.verticalLayout_4.setStretch(0, 2)
        self.verticalLayout_4.setStretch(1, 8)
        self.splitter_2.addWidget(self.widget_2)
        self.groupBox_3 = QGroupBox(self.splitter_2)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.verticalLayout = QVBoxLayout(self.groupBox_3)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.splitter = QSplitter(self.groupBox_3)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Vertical)
        self.tree_view_context = QTreeViewSpaceNoteWidget(self.splitter)
        self.tree_view_context.setObjectName(u"tree_view_context")
        self.splitter.addWidget(self.tree_view_context)
        self.textEdit = QTextEdit(self.splitter)
        self.textEdit.setObjectName(u"textEdit")
        self.splitter.addWidget(self.textEdit)

        self.verticalLayout.addWidget(self.splitter)

        self.splitter_2.addWidget(self.groupBox_3)

        self.gridLayout.addWidget(self.splitter_2, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.queryButton.clicked.connect(MainWindow.query_workspace_his)
        self.page_spinBox.valueChanged.connect(MainWindow.change_page)
        self.volume_spinBox.valueChanged.connect(MainWindow.change_volume)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"WorkSpace Sync History", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"user", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"sample", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"env_name", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"name", None))
        self.queryButton.setText(QCoreApplication.translate("MainWindow", u"query", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"note list", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"        Page", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"      Volume", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Change", None))
    # retranslateUi

