# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'work_sapce_ui.ui'
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
    QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSizePolicy, QSpacerItem, QToolBar, QVBoxLayout,
    QWidget)

from .widgets.combox_custom.combox_multi import QMultiComboBox

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(647, 443)
        self.ActionRefresh = QAction(MainWindow)
        self.ActionRefresh.setObjectName(u"ActionRefresh")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_2 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.widget_5 = QWidget(self.groupBox)
        self.widget_5.setObjectName(u"widget_5")
        self.horizontalLayout_7 = QHBoxLayout(self.widget_5)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_4 = QLabel(self.widget_5)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_7.addWidget(self.label_4)

        self.QubitBox = QMultiComboBox(self.widget_5)
        self.QubitBox.setObjectName(u"QubitBox")

        self.horizontalLayout_7.addWidget(self.QubitBox)

        self.horizontalLayout_7.setStretch(0, 1)
        self.horizontalLayout_7.setStretch(1, 7)

        self.gridLayout.addWidget(self.widget_5, 0, 0, 1, 1)

        self.widget_3 = QWidget(self.groupBox)
        self.widget_3.setObjectName(u"widget_3")
        self.horizontalLayout_2 = QHBoxLayout(self.widget_3)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_2 = QLabel(self.widget_3)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_2.addWidget(self.label_2)

        self.ConfigBox = QMultiComboBox(self.widget_3)
        self.ConfigBox.setObjectName(u"ConfigBox")

        self.horizontalLayout_2.addWidget(self.ConfigBox)

        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 7)

        self.gridLayout.addWidget(self.widget_3, 1, 0, 1, 1)

        self.widget_4 = QWidget(self.groupBox)
        self.widget_4.setObjectName(u"widget_4")
        self.horizontalLayout_5 = QHBoxLayout(self.widget_4)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_3 = QLabel(self.widget_4)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_5.addWidget(self.label_3)

        self.PointBox = QMultiComboBox(self.widget_4)
        self.PointBox.setObjectName(u"PointBox")

        self.horizontalLayout_5.addWidget(self.PointBox)

        self.horizontalLayout_5.setStretch(0, 1)
        self.horizontalLayout_5.setStretch(1, 7)

        self.gridLayout.addWidget(self.widget_4, 2, 0, 1, 1)

        self.widget_6 = QWidget(self.groupBox)
        self.widget_6.setObjectName(u"widget_6")
        self.horizontalLayout_8 = QHBoxLayout(self.widget_6)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.label_5 = QLabel(self.widget_6)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_8.addWidget(self.label_5)

        self.AttrBox = QMultiComboBox(self.widget_6)
        self.AttrBox.setObjectName(u"AttrBox")

        self.horizontalLayout_8.addWidget(self.AttrBox)

        self.horizontalLayout_8.setStretch(0, 1)
        self.horizontalLayout_8.setStretch(1, 7)

        self.gridLayout.addWidget(self.widget_6, 3, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 4, 0, 1, 1)

        self.widget_2 = QWidget(self.groupBox)
        self.widget_2.setObjectName(u"widget_2")
        self.horizontalLayout_6 = QHBoxLayout(self.widget_2)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalSpacer_3 = QSpacerItem(249, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_3)

        self.SaveConfButton = QPushButton(self.widget_2)
        self.SaveConfButton.setObjectName(u"SaveConfButton")

        self.horizontalLayout_6.addWidget(self.SaveConfButton)

        self.horizontalSpacer_4 = QSpacerItem(249, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_4)


        self.gridLayout.addWidget(self.widget_2, 5, 0, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer_2, 6, 0, 1, 1)


        self.verticalLayout_2.addWidget(self.groupBox)

        self.groupBox_3 = QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.horizontalLayout_3 = QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalSpacer_9 = QSpacerItem(74, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_9)

        self.autoPullCheck = QCheckBox(self.groupBox_3)
        self.autoPullCheck.setObjectName(u"autoPullCheck")

        self.horizontalLayout_3.addWidget(self.autoPullCheck)

        self.horizontalSpacer_10 = QSpacerItem(74, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_10)

        self.autoPushCheck = QCheckBox(self.groupBox_3)
        self.autoPushCheck.setObjectName(u"autoPushCheck")

        self.horizontalLayout_3.addWidget(self.autoPushCheck)

        self.horizontalSpacer_11 = QSpacerItem(74, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_11)

        self.autoPullPushSave = QPushButton(self.groupBox_3)
        self.autoPullPushSave.setObjectName(u"autoPullPushSave")

        self.horizontalLayout_3.addWidget(self.autoPullPushSave)


        self.verticalLayout_2.addWidget(self.groupBox_3)

        self.groupBox_4 = QGroupBox(self.centralwidget)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.horizontalLayout_4 = QHBoxLayout(self.groupBox_4)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalSpacer_12 = QSpacerItem(80, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_12)

        self.pullButton = QPushButton(self.groupBox_4)
        self.pullButton.setObjectName(u"pullButton")
        font = QFont()
        font.setBold(True)
        self.pullButton.setFont(font)
        self.pullButton.setAutoDefault(False)

        self.horizontalLayout_4.addWidget(self.pullButton)

        self.horizontalSpacer_13 = QSpacerItem(80, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_13)

        self.pushButton = QPushButton(self.groupBox_4)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setFont(font)

        self.horizontalLayout_4.addWidget(self.pushButton)

        self.horizontalSpacer_14 = QSpacerItem(80, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_14)


        self.verticalLayout_2.addWidget(self.groupBox_4)

        MainWindow.setCentralWidget(self.centralwidget)
        self.toolBar = QToolBar(MainWindow)
        self.toolBar.setObjectName(u"toolBar")
        MainWindow.addToolBar(Qt.TopToolBarArea, self.toolBar)

        self.toolBar.addAction(self.ActionRefresh)

        self.retranslateUi(MainWindow)
        self.autoPullPushSave.clicked.connect(MainWindow.change_auto_option)
        self.pullButton.clicked.connect(MainWindow.pull_data)
        self.pushButton.clicked.connect(MainWindow.push_data)
        self.ActionRefresh.triggered.connect(MainWindow.refresh_query_data)
        self.SaveConfButton.clicked.connect(MainWindow.save_space_conf)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"WorkSpace", None))
        self.ActionRefresh.setText(QCoreApplication.translate("MainWindow", u"refresh", None))
#if QT_CONFIG(tooltip)
        self.ActionRefresh.setToolTip(QCoreApplication.translate("MainWindow", u"Refresh the options in the drop-down menu below", None))
#endif // QT_CONFIG(tooltip)
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"User WorkSpace Configuration", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"qubit", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"config", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"point_label", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"bit_attr", None))
        self.SaveConfButton.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Setting", None))
        self.autoPullCheck.setText(QCoreApplication.translate("MainWindow", u"Auto PULL", None))
        self.autoPushCheck.setText(QCoreApplication.translate("MainWindow", u"Auto PUSH", None))
        self.autoPullPushSave.setText(QCoreApplication.translate("MainWindow", u"save", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("MainWindow", u"Manual-Sync", None))
        self.pullButton.setText(QCoreApplication.translate("MainWindow", u"PULL", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"PUSH", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar", None))
    # retranslateUi

