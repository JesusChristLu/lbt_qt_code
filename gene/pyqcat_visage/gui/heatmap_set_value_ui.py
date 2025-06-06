# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'heatmap_set_value_ui.ui'
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QSizePolicy, QWidget)

class Ui_SetValueWindow(object):
    def setupUi(self, SetValueWindow):
        if not SetValueWindow.objectName():
            SetValueWindow.setObjectName(u"SetValueWindow")
        SetValueWindow.resize(460, 106)
        self.centralwidget = QWidget(SetValueWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_2 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.lineEdit = QLineEdit(self.centralwidget)
        self.lineEdit.setObjectName(u"lineEdit")

        self.horizontalLayout.addWidget(self.lineEdit)

        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setObjectName(u"pushButton")
        icon = QIcon()
        icon.addFile(u"_imgs/ok.png", QSize(), QIcon.Normal, QIcon.Off)
        self.pushButton.setIcon(icon)

        self.horizontalLayout.addWidget(self.pushButton)

        self.horizontalLayout.setStretch(0, 3)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(2, 1)

        self.horizontalLayout_2.addLayout(self.horizontalLayout)

        SetValueWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(SetValueWindow)
        self.pushButton.clicked.connect(SetValueWindow.change_value)

        QMetaObject.connectSlotsByName(SetValueWindow)
    # setupUi

    def retranslateUi(self, SetValueWindow):
        SetValueWindow.setWindowTitle(QCoreApplication.translate("SetValueWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("SetValueWindow", u"SetValue", None))
        self.pushButton.setText(QCoreApplication.translate("SetValueWindow", u"OK", None))
    # retranslateUi

