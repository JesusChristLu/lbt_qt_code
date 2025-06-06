# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'run_setting_ui.ui'
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
    QSpinBox, QStatusBar, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(598, 372)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.verticalLayout = QVBoxLayout(self.groupBox_2)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_2 = QLabel(self.groupBox_2)
        self.label_2.setObjectName(u"label_2")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy1)

        self.horizontalLayout_2.addWidget(self.label_2)

        self.exp_save_mode = QComboBox(self.groupBox_2)
        self.exp_save_mode.addItem("")
        self.exp_save_mode.addItem("")
        self.exp_save_mode.setObjectName(u"exp_save_mode")

        self.horizontalLayout_2.addWidget(self.exp_save_mode)

        self.use_simulator = QCheckBox(self.groupBox_2)
        self.use_simulator.setObjectName(u"use_simulator")

        self.horizontalLayout_2.addWidget(self.use_simulator)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Preferred, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_4)

        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 2)
        self.horizontalLayout_2.setStretch(2, 2)
        self.horizontalLayout_2.setStretch(3, 2)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setSpacing(6)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_3 = QLabel(self.groupBox_2)
        self.label_3.setObjectName(u"label_3")
        sizePolicy1.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy1)

        self.horizontalLayout_3.addWidget(self.label_3)

        self.simulator_data_path = QLineEdit(self.groupBox_2)
        self.simulator_data_path.setObjectName(u"simulator_data_path")
        sizePolicy1.setHeightForWidth(self.simulator_data_path.sizePolicy().hasHeightForWidth())
        self.simulator_data_path.setSizePolicy(sizePolicy1)

        self.horizontalLayout_3.addWidget(self.simulator_data_path)

        self.import_button = QPushButton(self.groupBox_2)
        self.import_button.setObjectName(u"import_button")
        sizePolicy1.setHeightForWidth(self.import_button.sizePolicy().hasHeightForWidth())
        self.import_button.setSizePolicy(sizePolicy1)
        icon = QIcon()
        icon.addFile(u"_imgs/import.png", QSize(), QIcon.Normal, QIcon.Off)
        self.import_button.setIcon(icon)

        self.horizontalLayout_3.addWidget(self.import_button)

        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 5)
        self.horizontalLayout_3.setStretch(2, 1)

        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.label_9 = QLabel(self.groupBox_2)
        self.label_9.setObjectName(u"label_9")
        sizePolicy1.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy1)

        self.horizontalLayout_8.addWidget(self.label_9)

        self.simulator_delay = QLineEdit(self.groupBox_2)
        self.simulator_delay.setObjectName(u"simulator_delay")
        sizePolicy1.setHeightForWidth(self.simulator_delay.sizePolicy().hasHeightForWidth())
        self.simulator_delay.setSizePolicy(sizePolicy1)

        self.horizontalLayout_8.addWidget(self.simulator_delay)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_8.addItem(self.horizontalSpacer_7)


        self.verticalLayout.addLayout(self.horizontalLayout_8)


        self.gridLayout.addWidget(self.groupBox_2, 0, 0, 2, 1)

        self.groupBox_3 = QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.verticalLayout_2 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer_2)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalSpacer_2 = QSpacerItem(37, 17, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_2)

        self.label_4 = QLabel(self.groupBox_3)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_4.addWidget(self.label_4)

        self.horizontalSpacer = QSpacerItem(37, 17, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer)

        self.inst_host = QLineEdit(self.groupBox_3)
        self.inst_host.setObjectName(u"inst_host")
        self.inst_host.setMaxLength(15)

        self.horizontalLayout_4.addWidget(self.inst_host)

        self.horizontalSpacer_3 = QSpacerItem(37, 17, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_3)


        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_5 = QLabel(self.groupBox_3)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_5.addWidget(self.label_5)

        self.inst_mongo = QSpinBox(self.groupBox_3)
        self.inst_mongo.setObjectName(u"inst_mongo")
        self.inst_mongo.setMaximum(50000)
        self.inst_mongo.setValue(27017)

        self.horizontalLayout_5.addWidget(self.inst_mongo)

        self.label_6 = QLabel(self.groupBox_3)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_5.addWidget(self.label_6)

        self.inst_log = QSpinBox(self.groupBox_3)
        self.inst_log.setObjectName(u"inst_log")
        self.inst_log.setMaximum(50000)
        self.inst_log.setValue(27021)
        self.inst_log.setDisplayIntegerBase(10)

        self.horizontalLayout_5.addWidget(self.inst_log)


        self.verticalLayout_2.addLayout(self.horizontalLayout_5)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer_3)

        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 2)
        self.verticalLayout_2.setStretch(2, 1)
        self.verticalLayout_2.setStretch(3, 2)
        self.verticalLayout_2.setStretch(4, 1)

        self.gridLayout.addWidget(self.groupBox_3, 4, 0, 1, 1)

        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.horizontalLayout = QHBoxLayout(self.groupBox)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        sizePolicy1.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy1)

        self.horizontalLayout.addWidget(self.label)

        self.dag_save_mode = QComboBox(self.groupBox)
        self.dag_save_mode.addItem("")
        self.dag_save_mode.addItem("")
        self.dag_save_mode.addItem("")
        self.dag_save_mode.setObjectName(u"dag_save_mode")

        self.horizontalLayout.addWidget(self.dag_save_mode)

        self.use_backtrace = QCheckBox(self.groupBox)
        self.use_backtrace.setObjectName(u"use_backtrace")

        self.horizontalLayout.addWidget(self.use_backtrace)

        self.register_dag = QCheckBox(self.groupBox)
        self.register_dag.setObjectName(u"register_dag")

        self.horizontalLayout.addWidget(self.register_dag)

        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 2)
        self.horizontalLayout.setStretch(2, 2)
        self.horizontalLayout.setStretch(3, 2)

        self.gridLayout.addWidget(self.groupBox, 2, 0, 1, 1)

        self.default_button = QPushButton(self.centralwidget)
        self.default_button.setObjectName(u"default_button")
        icon1 = QIcon()
        icon1.addFile(u"_imgs/reset.png", QSize(), QIcon.Normal, QIcon.Off)
        self.default_button.setIcon(icon1)

        self.gridLayout.addWidget(self.default_button, 4, 1, 1, 1)

        self.ok_button = QPushButton(self.centralwidget)
        self.ok_button.setObjectName(u"ok_button")
        icon2 = QIcon()
        icon2.addFile(u"_imgs/ok.png", QSize(), QIcon.Normal, QIcon.Off)
        self.ok_button.setIcon(icon2)

        self.gridLayout.addWidget(self.ok_button, 1, 1, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.ok_button.clicked.connect(MainWindow.ok)
        self.default_button.clicked.connect(MainWindow.default)
        self.import_button.clicked.connect(MainWindow.import_sp)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Run Setting", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"EXP", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Result", None))
        self.exp_save_mode.setItemText(0, QCoreApplication.translate("MainWindow", u"Unsave", None))
        self.exp_save_mode.setItemText(1, QCoreApplication.translate("MainWindow", u"Save", None))

        self.use_simulator.setText(QCoreApplication.translate("MainWindow", u"simulator", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Path", None))
        self.simulator_data_path.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Please input simulator dat path...", None))
        self.import_button.setText(QCoreApplication.translate("MainWindow", u"Import", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"simulator delay", None))
        self.simulator_delay.setText("")
        self.simulator_delay.setPlaceholderText(QCoreApplication.translate("MainWindow", u"delay...", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"QAIO", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"qaio ip", None))
        self.inst_host.setText(QCoreApplication.translate("MainWindow", u"127.0.0.1", None))
        self.inst_host.setPlaceholderText("")
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"mongo port", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"log port", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"DAG", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Result", None))
        self.dag_save_mode.setItemText(0, QCoreApplication.translate("MainWindow", u"Unsave", None))
        self.dag_save_mode.setItemText(1, QCoreApplication.translate("MainWindow", u"Save In Process", None))
        self.dag_save_mode.setItemText(2, QCoreApplication.translate("MainWindow", u"Save In Final", None))

        self.use_backtrace.setText(QCoreApplication.translate("MainWindow", u"backtrace", None))
        self.register_dag.setText(QCoreApplication.translate("MainWindow", u"register", None))
        self.default_button.setText(QCoreApplication.translate("MainWindow", u"Default", None))
        self.ok_button.setText(QCoreApplication.translate("MainWindow", u"Ok", None))
    # retranslateUi

