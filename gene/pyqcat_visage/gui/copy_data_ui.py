# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'copy_data_ui.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QSizePolicy, QSpacerItem,
    QStatusBar, QVBoxLayout, QWidget)

from .widgets.combox_custom.combox_multi import QMultiComboBox
from .widgets.combox_custom.combox_search import SearchComboBox

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1048, 601)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        self.horizontalLayout_2 = QHBoxLayout(self.widget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.groupBox_2 = QGroupBox(self.widget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.verticalLayout_3 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.widget_2 = QWidget(self.groupBox_2)
        self.widget_2.setObjectName(u"widget_2")
        self.gridLayout = QGridLayout(self.widget_2)
        self.gridLayout.setObjectName(u"gridLayout")
        self.CopyTips = QLabel(self.widget_2)
        self.CopyTips.setObjectName(u"CopyTips")
        self.CopyTips.setWordWrap(True)

        self.gridLayout.addWidget(self.CopyTips, 0, 0, 1, 1)


        self.verticalLayout_3.addWidget(self.widget_2)

        self.widget_5 = QWidget(self.groupBox_2)
        self.widget_5.setObjectName(u"widget_5")
        self.horizontalLayout_9 = QHBoxLayout(self.widget_5)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalSpacer_8 = QSpacerItem(164, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer_8)

        self.label_2 = QLabel(self.widget_5)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_9.addWidget(self.label_2)

        self.horizontalSpacer_9 = QSpacerItem(163, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer_9)

        self.label_11 = QLabel(self.widget_5)
        self.label_11.setObjectName(u"label_11")

        self.horizontalLayout_9.addWidget(self.label_11)

        self.horizontalSpacer_10 = QSpacerItem(92, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer_10)


        self.verticalLayout_3.addWidget(self.widget_5)

        self.widget_4 = QWidget(self.groupBox_2)
        self.widget_4.setObjectName(u"widget_4")
        self.horizontalLayout_7 = QHBoxLayout(self.widget_4)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label = QLabel(self.widget_4)
        self.label.setObjectName(u"label")

        self.horizontalLayout_7.addWidget(self.label)

        self.OtherNameText = SearchComboBox(self.widget_4)
        self.OtherNameText.setObjectName(u"OtherNameText")
        self.OtherNameText.setEditable(True)

        self.horizontalLayout_7.addWidget(self.OtherNameText)

        self.label_10 = QLabel(self.widget_4)
        self.label_10.setObjectName(u"label_10")

        self.horizontalLayout_7.addWidget(self.label_10)

        self.OtherNameText_2 = QLineEdit(self.widget_4)
        self.OtherNameText_2.setObjectName(u"OtherNameText_2")

        self.horizontalLayout_7.addWidget(self.OtherNameText_2)

        self.horizontalSpacer = QSpacerItem(116, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_7.addItem(self.horizontalSpacer)

        self.horizontalLayout_7.setStretch(0, 6)
        self.horizontalLayout_7.setStretch(1, 20)
        self.horizontalLayout_7.setStretch(2, 1)
        self.horizontalLayout_7.setStretch(3, 20)
        self.horizontalLayout_7.setStretch(4, 1)

        self.verticalLayout_3.addWidget(self.widget_4)

        self.widget_7 = QWidget(self.groupBox_2)
        self.widget_7.setObjectName(u"widget_7")
        self.horizontalLayout_6 = QHBoxLayout(self.widget_7)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_4 = QLabel(self.widget_7)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_6.addWidget(self.label_4)

        self.SampleText = QComboBox(self.widget_7)
        self.SampleText.setObjectName(u"SampleText")

        self.horizontalLayout_6.addWidget(self.SampleText)

        self.label_9 = QLabel(self.widget_7)
        self.label_9.setObjectName(u"label_9")
        font = QFont()
        font.setBold(True)
        self.label_9.setFont(font)

        self.horizontalLayout_6.addWidget(self.label_9)

        self.SampleText_2 = QLineEdit(self.widget_7)
        self.SampleText_2.setObjectName(u"SampleText_2")

        self.horizontalLayout_6.addWidget(self.SampleText_2)

        self.horizontalSpacer_4 = QSpacerItem(116, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_4)

        self.horizontalLayout_6.setStretch(0, 6)
        self.horizontalLayout_6.setStretch(1, 20)
        self.horizontalLayout_6.setStretch(2, 1)
        self.horizontalLayout_6.setStretch(3, 20)
        self.horizontalLayout_6.setStretch(4, 1)

        self.verticalLayout_3.addWidget(self.widget_7)

        self.widget_8 = QWidget(self.groupBox_2)
        self.widget_8.setObjectName(u"widget_8")
        self.horizontalLayout_4 = QHBoxLayout(self.widget_8)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_5 = QLabel(self.widget_8)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_4.addWidget(self.label_5)

        self.EnvNameText = QComboBox(self.widget_8)
        self.EnvNameText.setObjectName(u"EnvNameText")

        self.horizontalLayout_4.addWidget(self.EnvNameText)

        self.label_8 = QLabel(self.widget_8)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setFont(font)

        self.horizontalLayout_4.addWidget(self.label_8)

        self.EnvNameText_2 = QLineEdit(self.widget_8)
        self.EnvNameText_2.setObjectName(u"EnvNameText_2")

        self.horizontalLayout_4.addWidget(self.EnvNameText_2)

        self.horizontalSpacer_5 = QSpacerItem(116, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_5)

        self.horizontalLayout_4.setStretch(0, 6)
        self.horizontalLayout_4.setStretch(1, 20)
        self.horizontalLayout_4.setStretch(2, 1)
        self.horizontalLayout_4.setStretch(3, 20)
        self.horizontalLayout_4.setStretch(4, 1)

        self.verticalLayout_3.addWidget(self.widget_8)

        self.widget_9 = QWidget(self.groupBox_2)
        self.widget_9.setObjectName(u"widget_9")
        self.horizontalLayout_3 = QHBoxLayout(self.widget_9)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_6 = QLabel(self.widget_9)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_3.addWidget(self.label_6)

        self.PointLabelText = QComboBox(self.widget_9)
        self.PointLabelText.setObjectName(u"PointLabelText")

        self.horizontalLayout_3.addWidget(self.PointLabelText)

        self.label_7 = QLabel(self.widget_9)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_3.addWidget(self.label_7)

        self.PointLabelText_2 = QLineEdit(self.widget_9)
        self.PointLabelText_2.setObjectName(u"PointLabelText_2")

        self.horizontalLayout_3.addWidget(self.PointLabelText_2)

        self.horizontalSpacer_6 = QSpacerItem(21, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_6)

        self.horizontalLayout_3.setStretch(0, 6)
        self.horizontalLayout_3.setStretch(1, 20)
        self.horizontalLayout_3.setStretch(2, 1)
        self.horizontalLayout_3.setStretch(3, 20)
        self.horizontalLayout_3.setStretch(4, 1)

        self.verticalLayout_3.addWidget(self.widget_9)

        self.widget_6 = QWidget(self.groupBox_2)
        self.widget_6.setObjectName(u"widget_6")
        self.horizontalLayout_8 = QHBoxLayout(self.widget_6)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.label_3 = QLabel(self.widget_6)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_8.addWidget(self.label_3)

        self.ElementNamesText = QMultiComboBox(self.widget_6)
        self.ElementNamesText.setObjectName(u"ElementNamesText")
        self.ElementNamesText.setEditable(True)

        self.horizontalLayout_8.addWidget(self.ElementNamesText)

        self.horizontalSpacer_3 = QSpacerItem(37, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_8.addItem(self.horizontalSpacer_3)

        self.horizontalLayout_8.setStretch(0, 6)
        self.horizontalLayout_8.setStretch(1, 41)
        self.horizontalLayout_8.setStretch(2, 1)

        self.verticalLayout_3.addWidget(self.widget_6)

        self.widget_10 = QWidget(self.groupBox_2)
        self.widget_10.setObjectName(u"widget_10")
        self.horizontalLayout_10 = QHBoxLayout(self.widget_10)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.label_12 = QLabel(self.widget_10)
        self.label_12.setObjectName(u"label_12")

        self.horizontalLayout_10.addWidget(self.label_12)

        self.ElementConfigsText = QMultiComboBox(self.widget_10)
        self.ElementConfigsText.setObjectName(u"ElementConfigsText")
        self.ElementConfigsText.setEditable(True)

        self.horizontalLayout_10.addWidget(self.ElementConfigsText)

        self.horizontalSpacer_11 = QSpacerItem(37, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_10.addItem(self.horizontalSpacer_11)

        self.horizontalLayout_10.setStretch(0, 6)
        self.horizontalLayout_10.setStretch(1, 27)
        self.horizontalLayout_10.setStretch(2, 16)

        self.verticalLayout_3.addWidget(self.widget_10)

        self.widget_11 = QWidget(self.groupBox_2)
        self.widget_11.setObjectName(u"widget_11")
        self.horizontalLayout_11 = QHBoxLayout(self.widget_11)
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.horizontalLayout_11.addItem(self.verticalSpacer_3)


        self.verticalLayout_3.addWidget(self.widget_11)

        self.widget_tool = QWidget(self.groupBox_2)
        self.widget_tool.setObjectName(u"widget_tool")
        self.widget_tool.setMinimumSize(QSize(0, 0))
        font1 = QFont()
        font1.setUnderline(False)
        self.widget_tool.setFont(font1)
        self.horizontalLayout_5 = QHBoxLayout(self.widget_tool)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.LocalBox = QCheckBox(self.widget_tool)
        self.LocalBox.setObjectName(u"LocalBox")
        font2 = QFont()
        font2.setBold(True)
        font2.setUnderline(False)
        self.LocalBox.setFont(font2)
        self.LocalBox.setChecked(True)

        self.horizontalLayout_5.addWidget(self.LocalBox)

        self.horizontalSpacer_12 = QSpacerItem(46, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_12)

        self.CopyQubitBox = QCheckBox(self.widget_tool)
        self.CopyQubitBox.setObjectName(u"CopyQubitBox")
        self.CopyQubitBox.setFont(font2)
        self.CopyQubitBox.setChecked(True)

        self.horizontalLayout_5.addWidget(self.CopyQubitBox)

        self.horizontalSpacer_2 = QSpacerItem(136, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_2)

        self.SyncButton = QPushButton(self.widget_tool)
        self.SyncButton.setObjectName(u"SyncButton")

        self.horizontalLayout_5.addWidget(self.SyncButton)

        self.horizontalSpacer_7 = QSpacerItem(135, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_7)

        self.ResetButton = QPushButton(self.widget_tool)
        self.ResetButton.setObjectName(u"ResetButton")

        self.horizontalLayout_5.addWidget(self.ResetButton)

        self.horizontalLayout_5.setStretch(0, 1)
        self.horizontalLayout_5.setStretch(1, 1)
        self.horizontalLayout_5.setStretch(2, 1)
        self.horizontalLayout_5.setStretch(3, 4)
        self.horizontalLayout_5.setStretch(4, 1)
        self.horizontalLayout_5.setStretch(5, 6)
        self.horizontalLayout_5.setStretch(6, 1)

        self.verticalLayout_3.addWidget(self.widget_tool)

        self.verticalLayout_3.setStretch(0, 2)
        self.verticalLayout_3.setStretch(1, 1)
        self.verticalLayout_3.setStretch(2, 1)
        self.verticalLayout_3.setStretch(3, 1)
        self.verticalLayout_3.setStretch(4, 1)
        self.verticalLayout_3.setStretch(5, 1)
        self.verticalLayout_3.setStretch(6, 1)
        self.verticalLayout_3.setStretch(7, 1)
        self.verticalLayout_3.setStretch(8, 1)
        self.verticalLayout_3.setStretch(9, 1)

        self.horizontalLayout.addWidget(self.groupBox_2)

        self.widget_3 = QWidget(self.widget)
        self.widget_3.setObjectName(u"widget_3")
        self.verticalLayout_2 = QVBoxLayout(self.widget_3)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalSpacer = QSpacerItem(20, 65, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer)

        self.CopyButton = QPushButton(self.widget_3)
        self.CopyButton.setObjectName(u"CopyButton")

        self.verticalLayout_2.addWidget(self.CopyButton)

        self.verticalSpacer_2 = QSpacerItem(20, 66, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer_2)


        self.horizontalLayout.addWidget(self.widget_3)

        self.horizontalLayout.setStretch(0, 10)
        self.horizontalLayout.setStretch(1, 1)

        self.horizontalLayout_2.addLayout(self.horizontalLayout)

        self.horizontalLayout_2.setStretch(0, 8)

        self.verticalLayout.addWidget(self.widget)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.CopyButton.clicked.connect(MainWindow.copy_other_data)
        self.OtherNameText.currentIndexChanged.connect(MainWindow.update_sample)
        self.SampleText.currentTextChanged.connect(MainWindow.update_env_name)
        self.EnvNameText.currentTextChanged.connect(MainWindow.update_point_label)
        self.PointLabelText.currentTextChanged.connect(MainWindow.update_element_names)
        self.LocalBox.stateChanged.connect(MainWindow.change_local)
        self.SyncButton.clicked.connect(MainWindow.sync_env_data)
        self.ResetButton.clicked.connect(MainWindow.reset_window)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Copy Data", None))
        self.CopyTips.setText(QCoreApplication.translate("MainWindow", u"Please wait for a moment, the data copy task is being executed in the background, and it will be automatically completed later. If the copy fails, a pop-up message will be displayed. If there is no other message, the copy is successful!", None))
        # self.CopyTips.setText(QCoreApplication.translate("MainWindow", u"watting...", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"From", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"To", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"user", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"-->", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"sample", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"-->", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"env_name", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"-->", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"point_label", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"-->", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"element_names", None))
        self.ElementNamesText.setPlaceholderText("")
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"element_configs", None))
        self.ElementConfigsText.setPlaceholderText("")
        self.LocalBox.setText(QCoreApplication.translate("MainWindow", u"Local", None))
        self.CopyQubitBox.setText(QCoreApplication.translate("MainWindow", u"copy qubit", None))
        self.SyncButton.setText(QCoreApplication.translate("MainWindow", u"sync", None))
        self.ResetButton.setText(QCoreApplication.translate("MainWindow", u"reset", None))
        self.CopyButton.setText(QCoreApplication.translate("MainWindow", u"Copy", None))
    # retranslateUi

