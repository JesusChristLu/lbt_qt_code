# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'change_password_dialog_ui.ui'
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

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(561, 178)
        self.horizontalLayout = QHBoxLayout(Dialog)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.groupBox = QGroupBox(Dialog)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.PasswordText = QLineEdit(self.groupBox)
        self.PasswordText.setObjectName(u"PasswordText")
        self.PasswordText.setEchoMode(QLineEdit.Password)

        self.gridLayout.addWidget(self.PasswordText, 1, 1, 1, 1)

        self.PasswordAgainText = QLineEdit(self.groupBox)
        self.PasswordAgainText.setObjectName(u"PasswordAgainText")
        self.PasswordAgainText.setEchoMode(QLineEdit.Password)

        self.gridLayout.addWidget(self.PasswordAgainText, 2, 1, 1, 1)

        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)

        self.label_4 = QLabel(self.groupBox)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 1, 0, 1, 1)


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

        self.horizontalLayout.setStretch(0, 3)
        self.horizontalLayout.setStretch(1, 1)

        self.retranslateUi(Dialog)
        self.pushButton.clicked.connect(Dialog.accept)
        self.pushButton_2.clicked.connect(Dialog.reject)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Change Password", None))
        self.groupBox.setTitle(QCoreApplication.translate("Dialog", u"Password", None))
        self.label_3.setText(QCoreApplication.translate("Dialog", u"new password again", None))
        self.label_4.setText(QCoreApplication.translate("Dialog", u"new password", None))
        self.pushButton.setText(QCoreApplication.translate("Dialog", u"OK", None))
        self.pushButton_2.setText(QCoreApplication.translate("Dialog", u"Cancel", None))
    # retranslateUi

