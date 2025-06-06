# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'diff_thread_ui.ui'
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
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMenuBar,
    QPushButton, QSizePolicy, QSpacerItem, QSpinBox,
    QSplitter, QStatusBar, QTextEdit, QVBoxLayout,
    QWidget)

from .near_task_view import NearTaskView

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(883, 346)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.splitter = QSplitter(self.centralwidget)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.task_widget = QWidget(self.splitter)
        self.task_widget.setObjectName(u"task_widget")
        self.verticalLayout_3 = QVBoxLayout(self.task_widget)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.label = QLabel(self.task_widget)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.spinBoxpage = QSpinBox(self.task_widget)
        self.spinBoxpage.setObjectName(u"spinBoxpage")
        self.spinBoxpage.setMinimum(1)
        self.spinBoxpage.setMaximum(999)

        self.horizontalLayout.addWidget(self.spinBoxpage)

        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_8)

        self.label_2 = QLabel(self.task_widget)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout.addWidget(self.label_2)

        self.spinBoxVolume = QSpinBox(self.task_widget)
        self.spinBoxVolume.setObjectName(u"spinBoxVolume")
        self.spinBoxVolume.setMinimum(5)
        self.spinBoxVolume.setMaximum(40)
        self.spinBoxVolume.setValue(8)

        self.horizontalLayout.addWidget(self.spinBoxVolume)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_7)

        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(2, 3)
        self.horizontalLayout.setStretch(3, 1)
        self.horizontalLayout.setStretch(5, 3)
        self.horizontalLayout.setStretch(6, 1)

        self.verticalLayout_3.addLayout(self.horizontalLayout)

        self.tableView = NearTaskView(self.task_widget)
        self.tableView.setObjectName(u"tableView")

        self.verticalLayout_3.addWidget(self.tableView)

        self.splitter.addWidget(self.task_widget)
        self.diff_widget = QWidget(self.splitter)
        self.diff_widget.setObjectName(u"diff_widget")
        self.verticalLayout_2 = QVBoxLayout(self.diff_widget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.task2_input = QLineEdit(self.diff_widget)
        self.task2_input.setObjectName(u"task2_input")

        self.gridLayout_2.addWidget(self.task2_input, 1, 2, 1, 1)

        self.task1_input = QLineEdit(self.diff_widget)
        self.task1_input.setObjectName(u"task1_input")

        self.gridLayout_2.addWidget(self.task1_input, 0, 2, 1, 1)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer_3, 1, 1, 1, 1)

        self.lb_task1 = QLabel(self.diff_widget)
        self.lb_task1.setObjectName(u"lb_task1")

        self.gridLayout_2.addWidget(self.lb_task1, 0, 0, 1, 1)

        self.lb_task2 = QLabel(self.diff_widget)
        self.lb_task2.setObjectName(u"lb_task2")

        self.gridLayout_2.addWidget(self.lb_task2, 1, 0, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer_2, 0, 1, 1, 1)

        self.gridLayout_2.setColumnStretch(0, 2)
        self.gridLayout_2.setColumnStretch(1, 1)
        self.gridLayout_2.setColumnStretch(2, 10)

        self.verticalLayout_2.addLayout(self.gridLayout_2)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_6)

        self.diff_btn = QPushButton(self.diff_widget)
        self.diff_btn.setObjectName(u"diff_btn")

        self.horizontalLayout_2.addWidget(self.diff_btn)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_4)

        self.diff_status = QLabel(self.diff_widget)
        self.diff_status.setObjectName(u"diff_status")
        font = QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.diff_status.setFont(font)
        self.diff_status.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.diff_status)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_5)

        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 4)
        self.horizontalLayout_2.setStretch(2, 1)
        self.horizontalLayout_2.setStretch(3, 2)
        self.horizontalLayout_2.setStretch(4, 1)

        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.diff_text = QTextEdit(self.diff_widget)
        self.diff_text.setObjectName(u"diff_text")
        self.diff_text.setLineWrapMode(QTextEdit.WidgetWidth)
        self.diff_text.setReadOnly(True)

        self.verticalLayout_2.addWidget(self.diff_text)

        self.verticalLayout_2.setStretch(0, 2)
        self.verticalLayout_2.setStretch(1, 1)
        self.verticalLayout_2.setStretch(2, 4)
        self.splitter.addWidget(self.diff_widget)

        self.verticalLayout.addWidget(self.splitter)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 883, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.spinBoxpage.valueChanged.connect(MainWindow.query_task_list)
        self.spinBoxVolume.valueChanged.connect(MainWindow.query_task_list)
        self.diff_btn.clicked.connect(MainWindow.query_task_diff)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"page", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"volume", None))
        self.lb_task1.setText(QCoreApplication.translate("MainWindow", u"task1", None))
        self.lb_task2.setText(QCoreApplication.translate("MainWindow", u"task2", None))
        self.diff_btn.setText(QCoreApplication.translate("MainWindow", u"diff", None))
        self.diff_status.setText(QCoreApplication.translate("MainWindow", u"O", None))
    # retranslateUi

