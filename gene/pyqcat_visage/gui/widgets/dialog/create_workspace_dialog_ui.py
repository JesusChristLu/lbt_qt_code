# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'create_workspace_dialog_ui.ui'
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
    QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QWidget)

from ..combox_custom.combox_multi import QMultiComboBox
from ..combox_custom.combox_search import SearchComboBox

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(602, 343)
        self.horizontalLayout = QHBoxLayout(Dialog)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.groupBox = QGroupBox(Dialog)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_7 = QLabel(self.groupBox)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout.addWidget(self.label_7, 0, 0, 1, 1)

        self.UserText = SearchComboBox(self.groupBox)
        self.UserText.setObjectName(u"UserText")

        self.gridLayout.addWidget(self.UserText, 0, 1, 1, 1)

        self.label_4 = QLabel(self.groupBox)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 1, 0, 1, 1)

        self.SampleText = SearchComboBox(self.groupBox)
        self.SampleText.setObjectName(u"SampleText")

        self.gridLayout.addWidget(self.SampleText, 1, 1, 1, 1)

        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)

        self.EnvText = SearchComboBox(self.groupBox)
        self.EnvText.setObjectName(u"EnvText")

        self.gridLayout.addWidget(self.EnvText, 2, 1, 1, 1)

        self.label_5 = QLabel(self.groupBox)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 3, 0, 1, 1)

        self.qubitText = QMultiComboBox(self.groupBox)
        self.qubitText.setObjectName(u"qubitText")

        self.gridLayout.addWidget(self.qubitText, 3, 1, 1, 1)

        self.label_6 = QLabel(self.groupBox)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 4, 0, 1, 1)

        self.configText = QMultiComboBox(self.groupBox)
        self.configText.setObjectName(u"configText")

        self.gridLayout.addWidget(self.configText, 4, 1, 1, 1)

        self.label_8 = QLabel(self.groupBox)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout.addWidget(self.label_8, 5, 0, 1, 1)

        self.extraBitText = QLineEdit(self.groupBox)
        self.extraBitText.setObjectName(u"extraBitText")
        self.extraBitText.setEchoMode(QLineEdit.Normal)

        self.gridLayout.addWidget(self.extraBitText, 5, 1, 1, 1)

        self.label_9 = QLabel(self.groupBox)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout.addWidget(self.label_9, 6, 0, 1, 1)

        self.extraConfigText = QLineEdit(self.groupBox)
        self.extraConfigText.setObjectName(u"extraConfigText")
        self.extraConfigText.setEchoMode(QLineEdit.Normal)

        self.gridLayout.addWidget(self.extraConfigText, 6, 1, 1, 1)


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

        self.horizontalLayout.setStretch(0, 5)
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
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Create WorkSapce", None))
        self.groupBox.setTitle(QCoreApplication.translate("Dialog", u"WorkSpace", None))
        self.label_7.setText(QCoreApplication.translate("Dialog", u"username", None))
        self.label_4.setText(QCoreApplication.translate("Dialog", u"sample", None))
        self.label_3.setText(QCoreApplication.translate("Dialog", u"env_name", None))
        self.label_5.setText(QCoreApplication.translate("Dialog", u"qubit_names", None))
        self.label_6.setText(QCoreApplication.translate("Dialog", u"config_names", None))
        self.label_8.setText(QCoreApplication.translate("Dialog", u"extra bit", None))
        self.label_9.setText(QCoreApplication.translate("Dialog", u"extra config", None))
        self.pushButton.setText(QCoreApplication.translate("Dialog", u"OK", None))
        self.pushButton_2.setText(QCoreApplication.translate("Dialog", u"Cancel", None))
    # retranslateUi

