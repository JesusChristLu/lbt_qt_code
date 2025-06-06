# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'storm_manage_ui.ui'
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
    QHeaderView, QLabel, QMainWindow, QPushButton,
    QSizePolicy, QSpacerItem, QStatusBar, QToolBar,
    QVBoxLayout, QWidget)

from .widgets.chip_manage_files.table_view_storm import QTableViewStormWidget
from .widgets.combox_custom.combox_search import SearchComboBox

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(910, 682)
        self.actionCreateStom = QAction(MainWindow)
        self.actionCreateStom.setObjectName(u"actionCreateStom")
        self.actionRefresh = QAction(MainWindow)
        self.actionRefresh.setObjectName(u"actionRefresh")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(-1, -1, -1, 0)
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout_2 = QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.widget = QWidget(self.groupBox)
        self.widget.setObjectName(u"widget")
        self.horizontalLayout_2 = QHBoxLayout(self.widget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.widget_11 = QWidget(self.widget)
        self.widget_11.setObjectName(u"widget_11")
        self.gridLayout_3 = QGridLayout(self.widget_11)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.widget_3 = QWidget(self.widget_11)
        self.widget_3.setObjectName(u"widget_3")
        self.horizontalLayout_3 = QHBoxLayout(self.widget_3)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(-1, 0, -1, 0)
        self.label = QLabel(self.widget_3)
        self.label.setObjectName(u"label")

        self.horizontalLayout_3.addWidget(self.label)

        self.SampleContent = SearchComboBox(self.widget_3)
        self.SampleContent.setObjectName(u"SampleContent")

        self.horizontalLayout_3.addWidget(self.SampleContent)

        self.horizontalSpacer = QSpacerItem(95, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer)

        self.horizontalLayout_3.setStretch(0, 2)
        self.horizontalLayout_3.setStretch(1, 10)
        self.horizontalLayout_3.setStretch(2, 3)

        self.gridLayout_3.addWidget(self.widget_3, 0, 0, 1, 1)

        self.widget_4 = QWidget(self.widget_11)
        self.widget_4.setObjectName(u"widget_4")
        self.horizontalLayout = QHBoxLayout(self.widget_4)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(-1, 0, -1, 0)
        self.label_2 = QLabel(self.widget_4)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout.addWidget(self.label_2)

        self.EnvContent = SearchComboBox(self.widget_4)
        self.EnvContent.setObjectName(u"EnvContent")

        self.horizontalLayout.addWidget(self.EnvContent)

        self.horizontalSpacer_2 = QSpacerItem(95, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)

        self.horizontalLayout.setStretch(0, 2)
        self.horizontalLayout.setStretch(1, 10)
        self.horizontalLayout.setStretch(2, 3)

        self.gridLayout_3.addWidget(self.widget_4, 1, 0, 1, 1)


        self.horizontalLayout_2.addWidget(self.widget_11)

        self.widget_10 = QWidget(self.widget)
        self.widget_10.setObjectName(u"widget_10")
        self.verticalLayout_3 = QVBoxLayout(self.widget_10)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalSpacer = QSpacerItem(20, 18, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer)

        self.queryButton = QPushButton(self.widget_10)
        self.queryButton.setObjectName(u"queryButton")

        self.verticalLayout_3.addWidget(self.queryButton)

        self.verticalSpacer_2 = QSpacerItem(20, 18, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer_2)


        self.horizontalLayout_2.addWidget(self.widget_10)

        self.horizontalLayout_2.setStretch(0, 5)
        self.horizontalLayout_2.setStretch(1, 1)

        self.gridLayout_2.addWidget(self.widget, 0, 0, 1, 1)

        self.tableStormView = QTableViewStormWidget(self.groupBox)
        self.tableStormView.setObjectName(u"tableStormView")

        self.gridLayout_2.addWidget(self.tableStormView, 1, 0, 1, 1)

        self.gridLayout_2.setRowStretch(0, 1)
        self.gridLayout_2.setRowStretch(1, 10)

        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QToolBar(MainWindow)
        self.toolBar.setObjectName(u"toolBar")
        MainWindow.addToolBar(Qt.TopToolBarArea, self.toolBar)

        self.toolBar.addAction(self.actionCreateStom)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionRefresh)

        self.retranslateUi(MainWindow)
        self.actionCreateStom.triggered.connect(MainWindow.create_storm)
        self.queryButton.clicked.connect(MainWindow.query_storm)
        self.actionRefresh.triggered.connect(MainWindow.refresh)
        self.SampleContent.currentTextChanged.connect(MainWindow.sample_change)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Storm Manage", None))
        self.actionCreateStom.setText(QCoreApplication.translate("MainWindow", u"Create Storm", None))
        self.actionRefresh.setText(QCoreApplication.translate("MainWindow", u"refresh", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Storm", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"sample", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"env_name", None))
        self.queryButton.setText(QCoreApplication.translate("MainWindow", u"query", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar", None))
    # retranslateUi

