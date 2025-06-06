# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'copy_workspace_dialog_ui.ui'
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
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSpacerItem, QVBoxLayout, QWidget)

from ..combox_custom.combox_search import SearchComboBox

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(921, 252)
        self.horizontalLayout = QHBoxLayout(Dialog)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.groupBox = QGroupBox(Dialog)
        self.groupBox.setObjectName(u"groupBox")
        self.verticalLayout = QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.widget = QWidget(self.groupBox)
        self.widget.setObjectName(u"widget")
        self.horizontalLayout_2 = QHBoxLayout(self.widget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_7 = QLabel(self.widget)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_2.addWidget(self.label_7)

        self.horizontalSpacer_4 = QSpacerItem(18, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_4)

        self.UserText = SearchComboBox(self.widget)
        self.UserText.setObjectName(u"UserText")

        self.horizontalLayout_2.addWidget(self.UserText)

        self.label = QLabel(self.widget)
        self.label.setObjectName(u"label")

        self.horizontalLayout_2.addWidget(self.label)

        self.UserText_2 = SearchComboBox(self.widget)
        self.UserText_2.setObjectName(u"UserText_2")

        self.horizontalLayout_2.addWidget(self.UserText_2)

        self.horizontalSpacer_3 = QSpacerItem(17, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_3)

        self.horizontalLayout_2.setStretch(0, 3)
        self.horizontalLayout_2.setStretch(1, 1)
        self.horizontalLayout_2.setStretch(2, 15)
        self.horizontalLayout_2.setStretch(3, 1)
        self.horizontalLayout_2.setStretch(4, 15)
        self.horizontalLayout_2.setStretch(5, 1)

        self.verticalLayout.addWidget(self.widget)

        self.widget_2 = QWidget(self.groupBox)
        self.widget_2.setObjectName(u"widget_2")
        self.horizontalLayout_3 = QHBoxLayout(self.widget_2)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_4 = QLabel(self.widget_2)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_3.addWidget(self.label_4)

        self.horizontalSpacer_5 = QSpacerItem(18, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_5)

        self.SampleText = SearchComboBox(self.widget_2)
        self.SampleText.setObjectName(u"SampleText")

        self.horizontalLayout_3.addWidget(self.SampleText)

        self.label_2 = QLabel(self.widget_2)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_3.addWidget(self.label_2)

        self.SampleText_2 = SearchComboBox(self.widget_2)
        self.SampleText_2.setObjectName(u"SampleText_2")

        self.horizontalLayout_3.addWidget(self.SampleText_2)

        self.horizontalSpacer_8 = QSpacerItem(17, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_8)

        self.horizontalLayout_3.setStretch(0, 3)
        self.horizontalLayout_3.setStretch(1, 1)
        self.horizontalLayout_3.setStretch(2, 15)
        self.horizontalLayout_3.setStretch(3, 1)
        self.horizontalLayout_3.setStretch(4, 15)
        self.horizontalLayout_3.setStretch(5, 1)

        self.verticalLayout.addWidget(self.widget_2)

        self.widget_3 = QWidget(self.groupBox)
        self.widget_3.setObjectName(u"widget_3")
        self.horizontalLayout_4 = QHBoxLayout(self.widget_3)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_3 = QLabel(self.widget_3)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_4.addWidget(self.label_3)

        self.horizontalSpacer_9 = QSpacerItem(18, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_9)

        self.EnvText = SearchComboBox(self.widget_3)
        self.EnvText.setObjectName(u"EnvText")

        self.horizontalLayout_4.addWidget(self.EnvText)

        self.label_5 = QLabel(self.widget_3)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_4.addWidget(self.label_5)

        self.EnvText_2 = SearchComboBox(self.widget_3)
        self.EnvText_2.setObjectName(u"EnvText_2")

        self.horizontalLayout_4.addWidget(self.EnvText_2)

        self.horizontalSpacer_12 = QSpacerItem(17, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_12)

        self.horizontalLayout_4.setStretch(0, 3)
        self.horizontalLayout_4.setStretch(1, 1)
        self.horizontalLayout_4.setStretch(2, 15)
        self.horizontalLayout_4.setStretch(3, 1)
        self.horizontalLayout_4.setStretch(4, 15)
        self.horizontalLayout_4.setStretch(5, 1)

        self.verticalLayout.addWidget(self.widget_3)


        self.horizontalLayout.addWidget(self.groupBox)

        self.frame = QFrame(Dialog)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.gridLayout_2 = QGridLayout(self.frame)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.pushButton = QPushButton(self.frame)
        self.pushButton.setObjectName(u"pushButton")
        icon = QIcon()
        icon.addFile(u"../../_imgs/ok.png", QSize(), QIcon.Normal, QIcon.Off)
        self.pushButton.setIcon(icon)

        self.gridLayout_2.addWidget(self.pushButton, 0, 0, 1, 1)

        self.pushButton_2 = QPushButton(self.frame)
        self.pushButton_2.setObjectName(u"pushButton_2")
        icon1 = QIcon()
        icon1.addFile(u"../../_imgs/cancel.png", QSize(), QIcon.Normal, QIcon.Off)
        self.pushButton_2.setIcon(icon1)

        self.gridLayout_2.addWidget(self.pushButton_2, 1, 0, 1, 1)


        self.horizontalLayout.addWidget(self.frame)

        self.horizontalLayout.setStretch(0, 10)
        self.horizontalLayout.setStretch(1, 1)

        self.retranslateUi(Dialog)
        self.pushButton.clicked.connect(Dialog.accept)
        self.pushButton_2.clicked.connect(Dialog.reject)
        self.UserText.currentTextChanged.connect(Dialog.user_change)
        self.SampleText.currentTextChanged.connect(Dialog.sample_change)
        self.EnvText.currentTextChanged.connect(Dialog.env_change)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Copy WorkSapce", None))
        self.groupBox.setTitle(QCoreApplication.translate("Dialog", u"WorkSpace", None))
        self.label_7.setText(QCoreApplication.translate("Dialog", u"username", None))
        self.label.setText(QCoreApplication.translate("Dialog", u"-->", None))
        self.label_4.setText(QCoreApplication.translate("Dialog", u"sample", None))
        self.label_2.setText(QCoreApplication.translate("Dialog", u"-->", None))
        self.label_3.setText(QCoreApplication.translate("Dialog", u"env_name", None))
        self.label_5.setText(QCoreApplication.translate("Dialog", u"-->", None))
        self.pushButton.setText(QCoreApplication.translate("Dialog", u"Copy", None))
        self.pushButton_2.setText(QCoreApplication.translate("Dialog", u"Cancel", None))
    # retranslateUi

