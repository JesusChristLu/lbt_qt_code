# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'revert_bit_ui.ui'
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
from PySide6.QtWidgets import (QApplication, QDateTimeEdit, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QSpacerItem, QStatusBar,
    QVBoxLayout, QWidget)

from .widgets.component.table_view_revert import QTableViewRevertWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(572, 625)
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
        self.gridLayout_2 = QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
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
        self.horizontalLayout = QHBoxLayout(self.widget_6)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label_4 = QLabel(self.widget_6)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout.addWidget(self.label_4)

        self.horizontalSpacer_6 = QSpacerItem(58, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_6)

        self.TimeNodeText = QDateTimeEdit(self.widget_6)
        self.TimeNodeText.setObjectName(u"TimeNodeText")
        self.TimeNodeText.setMinimumDateTime(QDateTime(QDate(2020, 1, 1), QTime(0, 0, 0)))
        self.TimeNodeText.setCurrentSection(QDateTimeEdit.YearSection)

        self.horizontalLayout.addWidget(self.TimeNodeText)

        self.horizontalSpacer_4 = QSpacerItem(91, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_4)

        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(2, 3)
        self.horizontalLayout.setStretch(3, 3)

        self.verticalLayout_4.addWidget(self.widget_6)


        self.horizontalLayout_5.addWidget(self.widget_12)

        self.widget_14 = QWidget(self.widget_5)
        self.widget_14.setObjectName(u"widget_14")
        self.verticalLayout_5 = QVBoxLayout(self.widget_14)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalSpacer_3 = QSpacerItem(20, 15, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_5.addItem(self.verticalSpacer_3)

        self.queryRevertButton = QPushButton(self.widget_14)
        self.queryRevertButton.setObjectName(u"queryRevertButton")

        self.verticalLayout_5.addWidget(self.queryRevertButton)

        self.verticalSpacer_4 = QSpacerItem(20, 15, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_5.addItem(self.verticalSpacer_4)


        self.horizontalLayout_5.addWidget(self.widget_14)

        self.horizontalLayout_5.setStretch(0, 5)

        self.gridLayout_2.addWidget(self.widget_5, 0, 0, 1, 1)

        self.tableRevertView = QTableViewRevertWidget(self.groupBox_2)
        self.tableRevertView.setObjectName(u"tableRevertView")

        self.gridLayout_2.addWidget(self.tableRevertView, 1, 0, 1, 1)

        self.widget = QWidget(self.groupBox_2)
        self.widget.setObjectName(u"widget")
        self.horizontalLayout_2 = QHBoxLayout(self.widget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalSpacer = QSpacerItem(451, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.RevetButton = QPushButton(self.widget)
        self.RevetButton.setObjectName(u"RevetButton")

        self.horizontalLayout_2.addWidget(self.RevetButton)

        self.horizontalSpacer_2 = QSpacerItem(450, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)


        self.gridLayout_2.addWidget(self.widget, 2, 0, 1, 1)

        self.gridLayout_2.setRowStretch(0, 1)
        self.gridLayout_2.setRowStretch(1, 5)
        self.gridLayout_2.setRowStretch(2, 1)
        self.gridLayout_2.setRowMinimumHeight(0, 1)
        self.gridLayout_2.setRowMinimumHeight(1, 5)
        self.gridLayout_2.setRowMinimumHeight(2, 1)

        self.gridLayout.addWidget(self.groupBox_2, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.queryRevertButton.clicked.connect(MainWindow.query_revert_bits)
        self.RevetButton.clicked.connect(MainWindow.revert_bits)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Revert Bits", None))
        self.actionCreatSpace.setText(QCoreApplication.translate("MainWindow", u"Creat Space", None))
        self.actionRefresh.setText(QCoreApplication.translate("MainWindow", u"refresh", None))
        self.actionCopySpace.setText(QCoreApplication.translate("MainWindow", u"Copy Space", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Revet Qubit/Coupler", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"time node", None))
        self.TimeNodeText.setDisplayFormat(QCoreApplication.translate("MainWindow", u"yyyy-M-d HH:mm:ss", None))
        self.queryRevertButton.setText(QCoreApplication.translate("MainWindow", u"query", None))
        self.RevetButton.setText(QCoreApplication.translate("MainWindow", u"revert to this time", None))
    # retranslateUi

