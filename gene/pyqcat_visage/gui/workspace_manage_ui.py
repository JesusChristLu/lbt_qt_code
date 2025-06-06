# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'workspace_manage_ui.ui'
##
## Created by: Qt User Interface Compiler version 6.4.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLayout, QMainWindow,
    QPushButton, QSizePolicy, QSpacerItem, QStatusBar,
    QToolBar, QVBoxLayout, QWidget)

from .widgets.chip_manage_files.table_view_workspace import QTableViewWorkSpaceWidget
from .widgets.combox_custom.combox_search import SearchComboBox

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1086, 625)
        self.actionCreatSpace = QAction(MainWindow)
        self.actionCreatSpace.setObjectName(u"actionCreatSpace")
        self.actionRefresh = QAction(MainWindow)
        self.actionRefresh.setObjectName(u"actionRefresh")
        self.actionCopySpace = QAction(MainWindow)
        self.actionCopySpace.setObjectName(u"actionCopySpace")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(-1, -1, -1, 0)
        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.gridLayout_3 = QGridLayout(self.groupBox_2)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.tableWorkSpaceView = QTableViewWorkSpaceWidget(self.groupBox_2)
        self.tableWorkSpaceView.setObjectName(u"tableWorkSpaceView")

        self.gridLayout_3.addWidget(self.tableWorkSpaceView, 1, 0, 1, 1)

        self.widget_5 = QWidget(self.groupBox_2)
        self.widget_5.setObjectName(u"widget_5")
        self.horizontalLayout_5 = QHBoxLayout(self.widget_5)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.widget_12 = QWidget(self.widget_5)
        self.widget_12.setObjectName(u"widget_12")
        self.verticalLayout_4 = QVBoxLayout(self.widget_12)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.widget_6 = QWidget(self.widget_12)
        self.widget_6.setObjectName(u"widget_6")
        self.horizontalLayout_6 = QHBoxLayout(self.widget_6)
        self.horizontalLayout_6.setSpacing(6)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(-1, 0, -1, 0)
        self.label_4 = QLabel(self.widget_6)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_6.addWidget(self.label_4)

        self.workUserContent = SearchComboBox(self.widget_6)
        self.workUserContent.setObjectName(u"workUserContent")

        self.horizontalLayout_6.addWidget(self.workUserContent)

        self.horizontalSpacer_4 = QSpacerItem(91, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_4)

        self.horizontalLayout_6.setStretch(0, 2)
        self.horizontalLayout_6.setStretch(1, 10)
        self.horizontalLayout_6.setStretch(2, 5)

        self.verticalLayout_4.addWidget(self.widget_6)

        self.widget_7 = QWidget(self.widget_12)
        self.widget_7.setObjectName(u"widget_7")
        self.horizontalLayout_7 = QHBoxLayout(self.widget_7)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.horizontalLayout_7.setContentsMargins(-1, 0, -1, 0)
        self.label_5 = QLabel(self.widget_7)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_7.addWidget(self.label_5)

        self.workSampleContent = SearchComboBox(self.widget_7)
        self.workSampleContent.setObjectName(u"workSampleContent")
        self.workSampleContent.setEnabled(True)

        self.horizontalLayout_7.addWidget(self.workSampleContent)

        self.horizontalSpacer_5 = QSpacerItem(91, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_7.addItem(self.horizontalSpacer_5)

        self.horizontalLayout_7.setStretch(0, 2)
        self.horizontalLayout_7.setStretch(1, 10)
        self.horizontalLayout_7.setStretch(2, 5)

        self.verticalLayout_4.addWidget(self.widget_7)

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

        self.workEnvContent = SearchComboBox(self.widget_13)
        self.workEnvContent.setObjectName(u"workEnvContent")

        self.horizontalLayout_8.addWidget(self.workEnvContent)

        self.horizontalSpacer_6 = QSpacerItem(38, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_8.addItem(self.horizontalSpacer_6)

        self.horizontalLayout_8.setStretch(0, 2)
        self.horizontalLayout_8.setStretch(1, 10)
        self.horizontalLayout_8.setStretch(2, 5)

        self.verticalLayout_4.addWidget(self.widget_13)


        self.horizontalLayout_5.addWidget(self.widget_12)

        self.widget_14 = QWidget(self.widget_5)
        self.widget_14.setObjectName(u"widget_14")
        self.verticalLayout_5 = QVBoxLayout(self.widget_14)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalSpacer_3 = QSpacerItem(20, 15, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_5.addItem(self.verticalSpacer_3)

        self.queryWorkSpace = QPushButton(self.widget_14)
        self.queryWorkSpace.setObjectName(u"queryWorkSpace")

        self.verticalLayout_5.addWidget(self.queryWorkSpace)

        self.verticalSpacer_4 = QSpacerItem(20, 15, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_5.addItem(self.verticalSpacer_4)


        self.horizontalLayout_5.addWidget(self.widget_14)

        self.horizontalLayout_5.setStretch(0, 5)
        self.horizontalLayout_5.setStretch(1, 1)

        self.gridLayout_3.addWidget(self.widget_5, 0, 0, 1, 1)

        self.gridLayout_3.setRowStretch(0, 2)
        self.gridLayout_3.setRowStretch(1, 10)

        self.gridLayout.addWidget(self.groupBox_2, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QToolBar(MainWindow)
        self.toolBar.setObjectName(u"toolBar")
        MainWindow.addToolBar(Qt.TopToolBarArea, self.toolBar)

        self.toolBar.addAction(self.actionCreatSpace)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionCopySpace)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionRefresh)

        self.retranslateUi(MainWindow)
        self.queryWorkSpace.clicked.connect(MainWindow.query_workspace)
        self.actionCreatSpace.triggered.connect(MainWindow.creat_work_space)
        self.actionRefresh.triggered.connect(MainWindow.refresh)
        self.workSampleContent.currentTextChanged.connect(MainWindow.space_sample_change)
        self.actionCopySpace.triggered.connect(MainWindow.copy_space)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"WorkSpace Manage", None))
        self.actionCreatSpace.setText(QCoreApplication.translate("MainWindow", u"Creat Space", None))
        self.actionRefresh.setText(QCoreApplication.translate("MainWindow", u"refresh", None))
        self.actionCopySpace.setText(QCoreApplication.translate("MainWindow", u"Copy Space", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"WorkSpace", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"user", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"sample", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"env_name", None))
        self.queryWorkSpace.setText(QCoreApplication.translate("MainWindow", u"query", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar", None))
    # retranslateUi

