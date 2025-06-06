# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'chip_create_ui.ui'
##
## Created by: Qt User Interface Compiler version 6.3.2
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
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QPushButton,
    QSizePolicy, QSpacerItem, QVBoxLayout, QWidget)

from pyqcat_visage.gui.widgets.context.table_view_chip_bits import QTableViewChip
import pyqcat_visage.gui._imgs_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(601, 526)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalSpacer = QSpacerItem(20, 48, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.layout_style = QComboBox(self.centralwidget)
        self.layout_style.addItem("")
        self.layout_style.addItem("")
        self.layout_style.setObjectName(u"layout_style")

        self.horizontalLayout.addWidget(self.layout_style)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_3)

        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 2)
        self.horizontalLayout.setStretch(2, 3)
        self.horizontalLayout.setStretch(3, 1)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_4)

        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_2.addWidget(self.label_2)

        self.row_edit = QLineEdit(self.centralwidget)
        self.row_edit.setObjectName(u"row_edit")

        self.horizontalLayout_2.addWidget(self.row_edit)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_6)

        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 2)
        self.horizontalLayout_2.setStretch(2, 3)
        self.horizontalLayout_2.setStretch(3, 1)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_7)

        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_3.addWidget(self.label_3)

        self.col_edit = QLineEdit(self.centralwidget)
        self.col_edit.setObjectName(u"col_edit")

        self.horizontalLayout_3.addWidget(self.col_edit)

        self.horizontalSpacer_9 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_9)

        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 2)
        self.horizontalLayout_3.setStretch(2, 3)
        self.horizontalLayout_3.setStretch(3, 1)

        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_8)

        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_5.addWidget(self.label_4)

        self.chipTypeCom = QComboBox(self.centralwidget)
        self.chipTypeCom.setObjectName(u"chipTypeCom")
        self.chipTypeCom.setEditable(False)

        self.horizontalLayout_5.addWidget(self.chipTypeCom)

        self.horizontalSpacer_11 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_11)

        self.horizontalLayout_5.setStretch(0, 1)
        self.horizontalLayout_5.setStretch(1, 2)
        self.horizontalLayout_5.setStretch(2, 3)
        self.horizontalLayout_5.setStretch(3, 1)

        self.verticalLayout.addLayout(self.horizontalLayout_5)

        self.verticalSpacer_3 = QSpacerItem(20, 49, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_3)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalSpacer_10 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_10)

        self.layout_button = QPushButton(self.centralwidget)
        self.layout_button.setObjectName(u"layout_button")
        icon = QIcon()
        icon.addFile(u"C:/Users/BY210156/_imgs/cpu.png", QSize(), QIcon.Normal, QIcon.Off)
        self.layout_button.setIcon(icon)

        self.horizontalLayout_4.addWidget(self.layout_button)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_5)

        self.ok_button = QPushButton(self.centralwidget)
        self.ok_button.setObjectName(u"ok_button")
        icon1 = QIcon()
        icon1.addFile(u"C:/Users/BY210156/_imgs/ok.png", QSize(), QIcon.Normal, QIcon.Off)
        self.ok_button.setIcon(icon1)

        self.horizontalLayout_4.addWidget(self.ok_button)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_2)

        self.cancel_button = QPushButton(self.centralwidget)
        self.cancel_button.setObjectName(u"cancel_button")
        icon2 = QIcon()
        icon2.addFile(u"C:/Users/BY210156/_imgs/cancel.png", QSize(), QIcon.Normal, QIcon.Off)
        self.cancel_button.setIcon(icon2)

        self.horizontalLayout_4.addWidget(self.cancel_button)

        self.horizontalSpacer_12 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_12)

        self.horizontalLayout_4.setStretch(0, 1)
        self.horizontalLayout_4.setStretch(1, 2)
        self.horizontalLayout_4.setStretch(2, 1)
        self.horizontalLayout_4.setStretch(3, 2)
        self.horizontalLayout_4.setStretch(4, 1)
        self.horizontalLayout_4.setStretch(5, 2)
        self.horizontalLayout_4.setStretch(6, 1)

        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.verticalSpacer_2 = QSpacerItem(20, 48, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_2)

        self.tableView = QTableViewChip(self.centralwidget)
        self.tableView.setObjectName(u"tableView")

        self.verticalLayout.addWidget(self.tableView)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.ok_button.clicked.connect(MainWindow.ok)
        self.cancel_button.clicked.connect(MainWindow.cancel)
        self.layout_button.clicked.connect(MainWindow.layout)
        self.chipTypeCom.currentIndexChanged.connect(MainWindow.choose_chip_type)

        self.layout_style.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Chip Layout", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Layout Style", None))
        self.layout_style.setItemText(0, QCoreApplication.translate("MainWindow", u"Linear", None))
        self.layout_style.setItemText(1, QCoreApplication.translate("MainWindow", u"Grid", None))

        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Row", None))
        self.row_edit.setPlaceholderText(QCoreApplication.translate("MainWindow", u"please input row number", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Col", None))
        self.col_edit.setPlaceholderText(QCoreApplication.translate("MainWindow", u"please input col number", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"ChipType", None))
        self.layout_button.setText(QCoreApplication.translate("MainWindow", u"Layout", None))
        self.ok_button.setText(QCoreApplication.translate("MainWindow", u"Ok", None))
        self.cancel_button.setText(QCoreApplication.translate("MainWindow", u"Canel", None))
    # retranslateUi

