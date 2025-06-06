# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'chimera_manage_ui.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QMainWindow, QPushButton,
    QSizePolicy, QSpacerItem, QStatusBar, QToolBar,
    QVBoxLayout, QWidget)

from .widgets.chip_manage_files.table_view_chimera import QTableViewChimeraWidget
from .widgets.chip_manage_files.tree_view_chimera import QTreeViewChimeraWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1145, 643)
        self.actionCreateChip = QAction(MainWindow)
        self.actionCreateChip.setObjectName(u"actionCreateChip")
        self.actionRefresh = QAction(MainWindow)
        self.actionRefresh.setObjectName(u"actionRefresh")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(-1, -1, -1, 0)
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.horizontalLayout_2 = QHBoxLayout(self.groupBox)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.widget_2 = QWidget(self.groupBox)
        self.widget_2.setObjectName(u"widget_2")
        self.gridLayout_2 = QGridLayout(self.widget_2)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.widget = QWidget(self.widget_2)
        self.widget.setObjectName(u"widget")
        self.horizontalLayout_4 = QHBoxLayout(self.widget)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.widget_11 = QWidget(self.widget)
        self.widget_11.setObjectName(u"widget_11")
        self.horizontalLayout_3 = QHBoxLayout(self.widget_11)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.widget_3 = QWidget(self.widget_11)
        self.widget_3.setObjectName(u"widget_3")
        self.horizontalLayout = QHBoxLayout(self.widget_3)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.checkBoxShow = QCheckBox(self.widget_3)
        self.checkBoxShow.setObjectName(u"checkBoxShow")

        self.horizontalLayout.addWidget(self.checkBoxShow)

        self.horizontalSpacer = QSpacerItem(480, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)


        self.horizontalLayout_3.addWidget(self.widget_3)


        self.horizontalLayout_4.addWidget(self.widget_11)

        self.widget_10 = QWidget(self.widget)
        self.widget_10.setObjectName(u"widget_10")
        self.verticalLayout_3 = QVBoxLayout(self.widget_10)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalSpacer = QSpacerItem(20, 18, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer)

        self.queryChip = QPushButton(self.widget_10)
        self.queryChip.setObjectName(u"queryChip")

        self.verticalLayout_3.addWidget(self.queryChip)

        self.verticalSpacer_2 = QSpacerItem(20, 18, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer_2)


        self.horizontalLayout_4.addWidget(self.widget_10)

        self.horizontalLayout_4.setStretch(0, 4)
        self.horizontalLayout_4.setStretch(1, 1)

        self.gridLayout_2.addWidget(self.widget, 0, 0, 1, 1)

        self.tableChipView = QTableViewChimeraWidget(self.widget_2)
        self.tableChipView.setObjectName(u"tableChipView")

        self.gridLayout_2.addWidget(self.tableChipView, 1, 0, 1, 1)

        self.gridLayout_2.setRowStretch(0, 2)
        self.gridLayout_2.setRowStretch(1, 10)

        self.horizontalLayout_2.addWidget(self.widget_2)

        self.widget_5 = QWidget(self.groupBox)
        self.widget_5.setObjectName(u"widget_5")
        self.verticalLayout = QVBoxLayout(self.widget_5)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.TreeChimeraView = QTreeViewChimeraWidget(self.widget_5)
        self.TreeChimeraView.setObjectName(u"TreeChimeraView")

        self.verticalLayout.addWidget(self.TreeChimeraView)


        self.horizontalLayout_2.addWidget(self.widget_5)

        self.horizontalLayout_2.setStretch(0, 10)
        self.horizontalLayout_2.setStretch(1, 4)

        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QToolBar(MainWindow)
        self.toolBar.setObjectName(u"toolBar")
        MainWindow.addToolBar(Qt.TopToolBarArea, self.toolBar)

        self.toolBar.addAction(self.actionCreateChip)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionRefresh)

        self.retranslateUi(MainWindow)
        self.actionCreateChip.triggered.connect(MainWindow.create_chip)
        self.queryChip.clicked.connect(MainWindow.query_chip)
        self.actionRefresh.triggered.connect(MainWindow.refresh)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Chimera Manage", None))
        self.actionCreateChip.setText(QCoreApplication.translate("MainWindow", u"Create Chimera", None))
        self.actionRefresh.setText(QCoreApplication.translate("MainWindow", u"refresh", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Chimera", None))
        self.checkBoxShow.setText(QCoreApplication.translate("MainWindow", u"show all", None))
        self.queryChip.setText(QCoreApplication.translate("MainWindow", u"query", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar", None))
    # retranslateUi

