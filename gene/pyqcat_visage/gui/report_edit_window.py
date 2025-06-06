# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'report_edit_window.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QPushButton, QSizePolicy,
    QStatusBar, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(494, 304)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.frame_2 = QFrame(self.centralwidget)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.frame_2)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.groupBox = QGroupBox(self.frame_2)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.detail = QComboBox(self.groupBox)
        self.detail.addItem("")
        self.detail.addItem("")
        self.detail.setObjectName(u"detail")

        self.gridLayout.addWidget(self.detail, 4, 1, 1, 1)

        self.is_report = QCheckBox(self.groupBox)
        self.is_report.setObjectName(u"is_report")

        self.gridLayout.addWidget(self.is_report, 0, 0, 1, 1)

        self.language = QComboBox(self.groupBox)
        self.language.addItem("")
        self.language.addItem("")
        self.language.setObjectName(u"language")

        self.gridLayout.addWidget(self.language, 3, 1, 1, 1)

        self.save_type = QComboBox(self.groupBox)
        self.save_type.addItem("")
        self.save_type.addItem("")
        self.save_type.setObjectName(u"save_type")

        self.gridLayout.addWidget(self.save_type, 2, 1, 1, 1)

        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)

        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)

        self.theme = QComboBox(self.groupBox)
        self.theme.addItem("")
        self.theme.addItem("")
        self.theme.addItem("")
        self.theme.setObjectName(u"theme")

        self.gridLayout.addWidget(self.theme, 1, 1, 1, 1)

        self.label_5 = QLabel(self.groupBox)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 4, 0, 1, 1)

        self.label_4 = QLabel(self.groupBox)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)


        self.horizontalLayout_2.addWidget(self.groupBox)

        self.frame = QFrame(self.frame_2)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.gridLayout_2 = QGridLayout(self.frame)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.ok_button = QPushButton(self.frame)
        self.ok_button.setObjectName(u"ok_button")
        icon = QIcon()
        icon.addFile(u"_imgs/ok.png", QSize(), QIcon.Normal, QIcon.Off)
        self.ok_button.setIcon(icon)

        self.gridLayout_2.addWidget(self.ok_button, 0, 0, 1, 1)

        self.cancel_button = QPushButton(self.frame)
        self.cancel_button.setObjectName(u"cancel_button")
        icon1 = QIcon()
        icon1.addFile(u"_imgs/cancel.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cancel_button.setIcon(icon1)

        self.gridLayout_2.addWidget(self.cancel_button, 1, 0, 1, 1)


        self.horizontalLayout_2.addWidget(self.frame)

        self.horizontalLayout_2.setStretch(0, 3)
        self.horizontalLayout_2.setStretch(1, 1)

        self.verticalLayout.addWidget(self.frame_2)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label_6 = QLabel(self.centralwidget)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout.addWidget(self.label_6)

        self.lineEdit = QLineEdit(self.centralwidget)
        self.lineEdit.setObjectName(u"lineEdit")

        self.horizontalLayout.addWidget(self.lineEdit)

        self.choose_button = QPushButton(self.centralwidget)
        self.choose_button.setObjectName(u"choose_button")
        icon2 = QIcon()
        icon2.addFile(u"_imgs/import.png", QSize(), QIcon.Normal, QIcon.Off)
        self.choose_button.setIcon(icon2)

        self.horizontalLayout.addWidget(self.choose_button)


        self.verticalLayout.addLayout(self.horizontalLayout)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.ok_button.clicked.connect(MainWindow.update_report)
        self.cancel_button.clicked.connect(MainWindow.cancel)
        self.choose_button.clicked.connect(MainWindow.choose_path)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Report Setting", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Report Informations", None))
        self.detail.setItemText(0, QCoreApplication.translate("MainWindow", u"detailed", None))
        self.detail.setItemText(1, QCoreApplication.translate("MainWindow", u"simple", None))

        self.is_report.setText(QCoreApplication.translate("MainWindow", u"enable report", None))
        self.language.setItemText(0, QCoreApplication.translate("MainWindow", u"cn", None))
        self.language.setItemText(1, QCoreApplication.translate("MainWindow", u"en", None))

        self.save_type.setItemText(0, QCoreApplication.translate("MainWindow", u"pdf", None))
        self.save_type.setItemText(1, QCoreApplication.translate("MainWindow", u"html", None))

        self.label_2.setText(QCoreApplication.translate("MainWindow", u"theme", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"save type", None))
        self.theme.setItemText(0, QCoreApplication.translate("MainWindow", u"sync os", None))
        self.theme.setItemText(1, QCoreApplication.translate("MainWindow", u"dark", None))
        self.theme.setItemText(2, QCoreApplication.translate("MainWindow", u"white", None))

        self.label_5.setText(QCoreApplication.translate("MainWindow", u"detail", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"language", None))
        self.ok_button.setText(QCoreApplication.translate("MainWindow", u"OK", None))
        self.cancel_button.setText(QCoreApplication.translate("MainWindow", u"Cancel", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"file path", None))
        self.choose_button.setText(QCoreApplication.translate("MainWindow", u"Choose", None))
    # retranslateUi

