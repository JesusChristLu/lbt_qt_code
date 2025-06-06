# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'create_chimera_dialog_ui.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QWidget)

from ..combox_custom.combox_search import SearchComboBox

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(567, 430)
        self.horizontalLayout = QHBoxLayout(Dialog)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.groupBox = QGroupBox(Dialog)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_4 = QLabel(self.groupBox)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 0, 0, 1, 1)

        self.SampleText = QLineEdit(self.groupBox)
        self.SampleText.setObjectName(u"SampleText")
        self.SampleText.setEchoMode(QLineEdit.Normal)

        self.gridLayout.addWidget(self.SampleText, 0, 1, 1, 1)

        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)

        self.EnvText = QLineEdit(self.groupBox)
        self.EnvText.setObjectName(u"EnvText")
        self.EnvText.setEchoMode(QLineEdit.Normal)

        self.gridLayout.addWidget(self.EnvText, 1, 1, 1, 1)

        self.label_5 = QLabel(self.groupBox)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 2, 0, 1, 1)

        self.InstIpText = QLineEdit(self.groupBox)
        self.InstIpText.setObjectName(u"InstIpText")
        self.InstIpText.setEchoMode(QLineEdit.Normal)

        self.gridLayout.addWidget(self.InstIpText, 2, 1, 1, 1)

        self.label_6 = QLabel(self.groupBox)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 3, 0, 1, 1)

        self.InstPortText = QLineEdit(self.groupBox)
        self.InstPortText.setObjectName(u"InstPortText")
        self.InstPortText.setEchoMode(QLineEdit.Normal)

        self.gridLayout.addWidget(self.InstPortText, 3, 1, 1, 1)

        self.label_7 = QLabel(self.groupBox)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout.addWidget(self.label_7, 4, 0, 1, 1)

        self.GroupText = SearchComboBox(self.groupBox)
        self.GroupText.setObjectName(u"GroupText")

        self.gridLayout.addWidget(self.GroupText, 4, 1, 1, 1)

        self.label_8 = QLabel(self.groupBox)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout.addWidget(self.label_8, 5, 0, 1, 1)

        self.CoreNumText = QComboBox(self.groupBox)
        self.CoreNumText.setObjectName(u"CoreNumText")

        self.gridLayout.addWidget(self.CoreNumText, 5, 1, 1, 1)

        self.label_10 = QLabel(self.groupBox)
        self.label_10.setObjectName(u"label_10")

        self.gridLayout.addWidget(self.label_10, 6, 0, 1, 1)

        self.WindowSizeText = QComboBox(self.groupBox)
        self.WindowSizeText.setObjectName(u"WindowSizeText")

        self.gridLayout.addWidget(self.WindowSizeText, 6, 1, 1, 1)

        self.label_11 = QLabel(self.groupBox)
        self.label_11.setObjectName(u"label_11")

        self.gridLayout.addWidget(self.label_11, 7, 0, 1, 1)

        self.AlertDisText = QComboBox(self.groupBox)
        self.AlertDisText.setObjectName(u"AlertDisText")

        self.gridLayout.addWidget(self.AlertDisText, 7, 1, 1, 1)

        self.label_12 = QLabel(self.groupBox)
        self.label_12.setObjectName(u"label_12")

        self.gridLayout.addWidget(self.label_12, 8, 0, 1, 1)

        self.SecureDisText = QComboBox(self.groupBox)
        self.SecureDisText.setObjectName(u"SecureDisText")

        self.gridLayout.addWidget(self.SecureDisText, 8, 1, 1, 1)

        self.label_9 = QLabel(self.groupBox)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout.addWidget(self.label_9, 9, 0, 1, 1)

        self.DebugText = QComboBox(self.groupBox)
        self.DebugText.setObjectName(u"DebugText")

        self.gridLayout.addWidget(self.DebugText, 9, 1, 1, 1)


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

        self.horizontalLayout.setStretch(0, 7)
        self.horizontalLayout.setStretch(1, 1)

        self.retranslateUi(Dialog)
        self.pushButton.clicked.connect(Dialog.accept)
        self.pushButton_2.clicked.connect(Dialog.reject)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Create Chimera", None))
        self.groupBox.setTitle(QCoreApplication.translate("Dialog", u"Chimera", None))
        self.label_4.setText(QCoreApplication.translate("Dialog", u"sample", None))
        self.label_3.setText(QCoreApplication.translate("Dialog", u"env_name", None))
        self.label_5.setText(QCoreApplication.translate("Dialog", u"inst_ip", None))
        self.label_6.setText(QCoreApplication.translate("Dialog", u"inst_port", None))
        self.label_7.setText(QCoreApplication.translate("Dialog", u"groups", None))
        self.label_8.setText(QCoreApplication.translate("Dialog", u"core_num", None))
        self.label_10.setText(QCoreApplication.translate("Dialog", u"window_size", None))
        self.label_11.setText(QCoreApplication.translate("Dialog", u"alert_dis", None))
        self.label_12.setText(QCoreApplication.translate("Dialog", u"secure_dis", None))
        self.label_9.setText(QCoreApplication.translate("Dialog", u"debug", None))
        self.pushButton.setText(QCoreApplication.translate("Dialog", u"OK", None))
        self.pushButton_2.setText(QCoreApplication.translate("Dialog", u"Cancel", None))
    # retranslateUi

