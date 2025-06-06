# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'context_build_dialog_ui.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSizePolicy,
    QWidget)

from ..combox_custom.combox_multi import QMultiComboBox

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(514, 435)
        self.horizontalLayout = QHBoxLayout(Dialog)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.groupBox = QGroupBox(Dialog)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_9 = QLabel(self.groupBox)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout.addWidget(self.label_9, 0, 0, 1, 1)

        self.context_com = QComboBox(self.groupBox)
        self.context_com.addItem("")
        self.context_com.addItem("")
        self.context_com.addItem("")
        self.context_com.addItem("")
        self.context_com.addItem("")
        self.context_com.addItem("")
        self.context_com.addItem("")
        self.context_com.setObjectName(u"context_com")

        self.gridLayout.addWidget(self.context_com, 0, 1, 1, 1)

        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)

        self.base_qubit_com = QMultiComboBox(self.groupBox)
        self.base_qubit_com.setObjectName(u"base_qubit_com")
        self.base_qubit_com.setEnabled(True)
        self.base_qubit_com.setEditable(True)

        self.gridLayout.addWidget(self.base_qubit_com, 1, 1, 1, 1)

        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)

        self.work_type_com = QComboBox(self.groupBox)
        self.work_type_com.addItem("")
        self.work_type_com.addItem("")
        self.work_type_com.addItem("")
        self.work_type_com.setObjectName(u"work_type_com")
        self.work_type_com.setEnabled(True)
        self.work_type_com.setEditable(True)

        self.gridLayout.addWidget(self.work_type_com, 2, 1, 1, 1)

        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 3, 0, 1, 1)

        self.divide_type_com = QComboBox(self.groupBox)
        self.divide_type_com.addItem("")
        self.divide_type_com.addItem("")
        self.divide_type_com.addItem("")
        self.divide_type_com.addItem("")
        self.divide_type_com.addItem("")
        self.divide_type_com.setObjectName(u"divide_type_com")
        self.divide_type_com.setEnabled(True)
        self.divide_type_com.setEditable(True)

        self.gridLayout.addWidget(self.divide_type_com, 3, 1, 1, 1)

        self.label_6 = QLabel(self.groupBox)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 4, 0, 1, 1)

        self.read_com = QComboBox(self.groupBox)
        self.read_com.setObjectName(u"read_com")
        self.read_com.setEnabled(True)
        self.read_com.setEditable(True)

        self.gridLayout.addWidget(self.read_com, 4, 1, 1, 1)

        self.label_8 = QLabel(self.groupBox)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout.addWidget(self.label_8, 5, 0, 1, 1)

        self.union_edit = QLineEdit(self.groupBox)
        self.union_edit.setObjectName(u"union_edit")
        self.union_edit.setEnabled(True)

        self.gridLayout.addWidget(self.union_edit, 5, 1, 1, 1)

        self.label_4 = QLabel(self.groupBox)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 6, 0, 1, 1)

        self.max_qubit_com = QMultiComboBox(self.groupBox)
        self.max_qubit_com.setObjectName(u"max_qubit_com")
        self.max_qubit_com.setEnabled(True)
        self.max_qubit_com.setEditable(True)

        self.gridLayout.addWidget(self.max_qubit_com, 6, 1, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.use_dcm_check = QCheckBox(self.groupBox)
        self.use_dcm_check.setObjectName(u"use_dcm_check")
        self.use_dcm_check.setEnabled(True)

        self.horizontalLayout_2.addWidget(self.use_dcm_check)

        self.ac_switch_check = QCheckBox(self.groupBox)
        self.ac_switch_check.setObjectName(u"ac_switch_check")
        self.ac_switch_check.setEnabled(True)

        self.horizontalLayout_2.addWidget(self.ac_switch_check)

        self.crosstalk_check = QCheckBox(self.groupBox)
        self.crosstalk_check.setObjectName(u"crosstalk_check")
        self.crosstalk_check.setEnabled(True)
        self.crosstalk_check.setChecked(True)

        self.horizontalLayout_2.addWidget(self.crosstalk_check)


        self.gridLayout.addLayout(self.horizontalLayout_2, 7, 0, 1, 2)


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
        self.context_com.currentTextChanged.connect(Dialog.format_dialog)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Save As", None))
        self.groupBox.setTitle(QCoreApplication.translate("Dialog", u"Context Build Informations", None))
        self.label_9.setText(QCoreApplication.translate("Dialog", u"standrad context", None))
        self.context_com.setItemText(0, QCoreApplication.translate("Dialog", u"qubit_calibration", None))
        self.context_com.setItemText(1, QCoreApplication.translate("Dialog", u"coupler_probe_calibration", None))
        self.context_com.setItemText(2, QCoreApplication.translate("Dialog", u"coupler_calibration", None))
        self.context_com.setItemText(3, QCoreApplication.translate("Dialog", u"cz_gate_calibration", None))
        self.context_com.setItemText(4, QCoreApplication.translate("Dialog", u"crosstalk_measure", None))
        self.context_com.setItemText(5, QCoreApplication.translate("Dialog", u"union_read_measure", None))
        self.context_com.setItemText(6, QCoreApplication.translate("Dialog", u"net_tunable", None))

        self.label.setText(QCoreApplication.translate("Dialog", u"physical unit", None))
        self.label_2.setText(QCoreApplication.translate("Dialog", u"working type", None))
        self.work_type_com.setItemText(0, QCoreApplication.translate("Dialog", u"awg_bias", None))
        self.work_type_com.setItemText(1, QCoreApplication.translate("Dialog", u"ac", None))
        self.work_type_com.setItemText(2, QCoreApplication.translate("Dialog", u"dc", None))

        self.label_3.setText(QCoreApplication.translate("Dialog", u"divide type", None))
        self.divide_type_com.setItemText(0, QCoreApplication.translate("Dialog", u"character_idle_point", None))
        self.divide_type_com.setItemText(1, QCoreApplication.translate("Dialog", u"calibrate_idle_point", None))
        self.divide_type_com.setItemText(2, QCoreApplication.translate("Dialog", u"character_point", None))
        self.divide_type_com.setItemText(3, QCoreApplication.translate("Dialog", u"sweet_point", None))
        self.divide_type_com.setItemText(4, QCoreApplication.translate("Dialog", u"calibration_point", None))

        self.label_6.setText(QCoreApplication.translate("Dialog", u"readout type", None))
        self.label_8.setText(QCoreApplication.translate("Dialog", u"union bits", None))
        self.label_4.setText(QCoreApplication.translate("Dialog", u"max point unit", None))
        self.use_dcm_check.setText(QCoreApplication.translate("Dialog", u"use dcm", None))
        self.ac_switch_check.setText(QCoreApplication.translate("Dialog", u"ac switch", None))
        self.crosstalk_check.setText(QCoreApplication.translate("Dialog", u"crosstalk", None))
        self.pushButton.setText(QCoreApplication.translate("Dialog", u"OK", None))
        self.pushButton_2.setText(QCoreApplication.translate("Dialog", u"Cancel", None))
    # retranslateUi

