# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'save_library_dialog_ui.ui'
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
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QSizePolicy, QWidget)
import pyqcat_visage.gui._imgs_rc

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(448, 312)
        self.horizontalLayout = QHBoxLayout(Dialog)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.groupBox = QGroupBox(Dialog)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.library_widget = QListWidget(self.groupBox)
        self.library_widget.setObjectName(u"library_widget")
        self.library_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.gridLayout.addWidget(self.library_widget, 0, 0, 1, 3)

        self.save_type_label = QLabel(self.groupBox)
        self.save_type_label.setObjectName(u"save_type_label")

        self.gridLayout.addWidget(self.save_type_label, 1, 0, 1, 1)

        self.save_describe_label = QLabel(self.groupBox)
        self.save_describe_label.setObjectName(u"save_describe_label")

        self.gridLayout.addWidget(self.save_describe_label, 2, 0, 1, 2)

        self.describe_edit = QLineEdit(self.groupBox)
        self.describe_edit.setObjectName(u"describe_edit")

        self.gridLayout.addWidget(self.describe_edit, 2, 2, 1, 1)

        self.type_com_box = QComboBox(self.groupBox)
        self.type_com_box.addItem("")
        self.type_com_box.addItem("")
        self.type_com_box.setObjectName(u"type_com_box")

        self.gridLayout.addWidget(self.type_com_box, 1, 2, 1, 1)


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
        icon.addFile(u":/ok.png", QSize(), QIcon.Normal, QIcon.Off)
        self.pushButton.setIcon(icon)

        self.gridLayout_2.addWidget(self.pushButton, 0, 0, 1, 1)

        self.pushButton_2 = QPushButton(self.frame)
        self.pushButton_2.setObjectName(u"pushButton_2")
        icon1 = QIcon()
        icon1.addFile(u":/cancel.png", QSize(), QIcon.Normal, QIcon.Off)
        self.pushButton_2.setIcon(icon1)

        self.gridLayout_2.addWidget(self.pushButton_2, 1, 0, 1, 1)


        self.horizontalLayout.addWidget(self.frame)


        self.retranslateUi(Dialog)
        self.pushButton.clicked.connect(Dialog.accept)
        self.pushButton_2.clicked.connect(Dialog.reject)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Save Experiment Dialog", None))
        self.groupBox.setTitle(QCoreApplication.translate("Dialog", u"Library", None))
        self.save_type_label.setText(QCoreApplication.translate("Dialog", u"Save Type", None))
        self.save_describe_label.setText(QCoreApplication.translate("Dialog", u"Save Describe", None))
        self.type_com_box.setItemText(0, QCoreApplication.translate("Dialog", u"Database", None))
        self.type_com_box.setItemText(1, QCoreApplication.translate("Dialog", u"Local", None))

        self.pushButton.setText(QCoreApplication.translate("Dialog", u"OK", None))
        self.pushButton_2.setText(QCoreApplication.translate("Dialog", u"Cancel", None))
    # retranslateUi

