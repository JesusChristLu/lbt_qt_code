# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'multi_thread_ui.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QHeaderView,
    QLCDNumber, QLabel, QMainWindow, QPushButton,
    QSizePolicy, QSpacerItem, QSplitter, QStatusBar,
    QTabWidget, QTableView, QTextEdit, QVBoxLayout,
    QWidget)

from .multi_topology_widget import MultiTopologyWidget
from .threads_view import MultiTableView

class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        if not mainWindow.objectName():
            mainWindow.setObjectName(u"mainWindow")
        mainWindow.resize(1433, 686)
        self.centralwidget = QWidget(mainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_8 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.splitter = QSplitter(self.centralwidget)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.widget = QWidget(self.splitter)
        self.widget.setObjectName(u"widget")
        self.verticalLayout = QVBoxLayout(self.widget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)

        self.sample_name = QLabel(self.widget)
        self.sample_name.setObjectName(u"sample_name")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.sample_name.setFont(font)
        self.sample_name.setFocusPolicy(Qt.NoFocus)

        self.horizontalLayout.addWidget(self.sample_name)

        self.thread_count_lcd = QLCDNumber(self.widget)
        self.thread_count_lcd.setObjectName(u"thread_count_lcd")

        self.horizontalLayout.addWidget(self.thread_count_lcd)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.control_table = QPushButton(self.widget)
        self.control_table.setObjectName(u"control_table")
        icon = QIcon()
        icon.addFile(u":/refresh.png", QSize(), QIcon.Normal, QIcon.Off)
        self.control_table.setIcon(icon)
        self.control_table.setCheckable(False)
        self.control_table.setChecked(False)

        self.horizontalLayout.addWidget(self.control_table)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_4)

        self.shot_btn = QPushButton(self.widget)
        self.shot_btn.setObjectName(u"shot_btn")
        icon1 = QIcon()
        icon1.addFile(u":/screenshot.png", QSize(), QIcon.Normal, QIcon.Off)
        self.shot_btn.setIcon(icon1)

        self.horizontalLayout.addWidget(self.shot_btn)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_5)

        self.show_type = QComboBox(self.widget)
        self.show_type.setObjectName(u"show_type")

        self.horizontalLayout.addWidget(self.show_type)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_3)

        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 10)
        self.horizontalLayout.setStretch(2, 1)
        self.horizontalLayout.setStretch(3, 1)
        self.horizontalLayout.setStretch(4, 2)
        self.horizontalLayout.setStretch(5, 1)
        self.horizontalLayout.setStretch(6, 2)
        self.horizontalLayout.setStretch(7, 1)
        self.horizontalLayout.setStretch(8, 3)
        self.horizontalLayout.setStretch(9, 1)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.chip_widget = MultiTopologyWidget(self.widget)
        self.chip_widget.setObjectName(u"chip_widget")
        self.chip_widget.setEnabled(True)

        self.verticalLayout.addWidget(self.chip_widget)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 10)
        self.splitter.addWidget(self.widget)
        self.widget_2 = QWidget(self.splitter)
        self.widget_2.setObjectName(u"widget_2")
        self.verticalLayout_7 = QVBoxLayout(self.widget_2)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_8)

        self.label = QLabel(self.widget_2)
        self.label.setObjectName(u"label")

        self.horizontalLayout_2.addWidget(self.label)

        self.lcd_scheduler = QLCDNumber(self.widget_2)
        self.lcd_scheduler.setObjectName(u"lcd_scheduler")

        self.horizontalLayout_2.addWidget(self.lcd_scheduler)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_6)

        self.label_2 = QLabel(self.widget_2)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_2.addWidget(self.label_2)

        self.lcd_wait = QLCDNumber(self.widget_2)
        self.lcd_wait.setObjectName(u"lcd_wait")

        self.horizontalLayout_2.addWidget(self.lcd_wait)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_7)

        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 4)
        self.horizontalLayout_2.setStretch(2, 2)
        self.horizontalLayout_2.setStretch(3, 3)
        self.horizontalLayout_2.setStretch(4, 4)
        self.horizontalLayout_2.setStretch(5, 2)
        self.horizontalLayout_2.setStretch(6, 1)

        self.verticalLayout_7.addLayout(self.horizontalLayout_2)

        self.verticalSpacer_2 = QSpacerItem(20, 54, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_7.addItem(self.verticalSpacer_2)

        self.splitter_table = QSplitter(self.widget_2)
        self.splitter_table.setObjectName(u"splitter_table")
        self.splitter_table.setOrientation(Qt.Vertical)
        self.tr_table_view = MultiTableView(self.splitter_table)
        self.tr_table_view.setObjectName(u"tr_table_view")
        self.tr_table_view.setEnabled(True)
        self.splitter_table.addWidget(self.tr_table_view)
        self.widget1 = QWidget(self.splitter_table)
        self.widget1.setObjectName(u"widget1")
        self.verticalLayout_2 = QVBoxLayout(self.widget1)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer)

        self.tabWidget = QTabWidget(self.widget1)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.verticalLayout_3 = QVBoxLayout(self.tab)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.task_list_view = QTableView(self.tab)
        self.task_list_view.setObjectName(u"task_list_view")

        self.verticalLayout_3.addWidget(self.task_list_view)

        self.tabWidget.addTab(self.tab, "")
        self.tab_5 = QWidget()
        self.tab_5.setObjectName(u"tab_5")
        self.verticalLayout_6 = QVBoxLayout(self.tab_5)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalSpacer_13 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_13)

        self.label_3 = QLabel(self.tab_5)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_3.addWidget(self.label_3)

        self.horizontalSpacer_9 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_9)

        self.lcd_len_normal = QLCDNumber(self.tab_5)
        self.lcd_len_normal.setObjectName(u"lcd_len_normal")

        self.horizontalLayout_3.addWidget(self.lcd_len_normal)

        self.horizontalLayout_3.setStretch(0, 10)
        self.horizontalLayout_3.setStretch(1, 3)
        self.horizontalLayout_3.setStretch(2, 1)

        self.verticalLayout_6.addLayout(self.horizontalLayout_3)

        self.view_list_normal = QTableView(self.tab_5)
        self.view_list_normal.setObjectName(u"view_list_normal")

        self.verticalLayout_6.addWidget(self.view_list_normal)

        self.tabWidget.addTab(self.tab_5, "")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.verticalLayout_5 = QVBoxLayout(self.tab_3)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalSpacer_12 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_12)

        self.label_5 = QLabel(self.tab_3)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_4.addWidget(self.label_5)

        self.horizontalSpacer_10 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_10)

        self.lcd_len_low_priority = QLCDNumber(self.tab_3)
        self.lcd_len_low_priority.setObjectName(u"lcd_len_low_priority")

        self.horizontalLayout_4.addWidget(self.lcd_len_low_priority)

        self.horizontalLayout_4.setStretch(0, 10)
        self.horizontalLayout_4.setStretch(1, 3)
        self.horizontalLayout_4.setStretch(2, 1)

        self.verticalLayout_5.addLayout(self.horizontalLayout_4)

        self.view_list_low = QTableView(self.tab_3)
        self.view_list_low.setObjectName(u"view_list_low")

        self.verticalLayout_5.addWidget(self.view_list_low)

        self.tabWidget.addTab(self.tab_3, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.verticalLayout_4 = QVBoxLayout(self.tab_2)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.detail_text = QTextEdit(self.tab_2)
        self.detail_text.setObjectName(u"detail_text")
        self.detail_text.setEnabled(True)
        self.detail_text.setReadOnly(True)

        self.verticalLayout_4.addWidget(self.detail_text)

        self.tabWidget.addTab(self.tab_2, "")

        self.verticalLayout_2.addWidget(self.tabWidget)

        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 20)
        self.splitter_table.addWidget(self.widget1)

        self.verticalLayout_7.addWidget(self.splitter_table)

        self.verticalLayout_7.setStretch(0, 2)
        self.verticalLayout_7.setStretch(1, 1)
        self.verticalLayout_7.setStretch(2, 30)
        self.splitter.addWidget(self.widget_2)

        self.verticalLayout_8.addWidget(self.splitter)

        mainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(mainWindow)
        self.statusbar.setObjectName(u"statusbar")
        mainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(mainWindow)
        self.control_table.clicked.connect(mainWindow.change_tabel_hidden_status)
        self.show_type.currentTextChanged.connect(mainWindow.change_show_table)
        self.shot_btn.clicked.connect(mainWindow.screen_shot)

        self.tabWidget.setCurrentIndex(2)


        QMetaObject.connectSlotsByName(mainWindow)
    # setupUi

    def retranslateUi(self, mainWindow):
        mainWindow.setWindowTitle(QCoreApplication.translate("mainWindow", u"MainWindow", None))
        self.sample_name.setText(QCoreApplication.translate("mainWindow", u"sample", None))
        self.control_table.setText(QCoreApplication.translate("mainWindow", u"hidden tabel", None))
        self.shot_btn.setText(QCoreApplication.translate("mainWindow", u"shot", None))
        self.label.setText(QCoreApplication.translate("mainWindow", u"Scheduler Queue", None))
        self.label_2.setText(QCoreApplication.translate("mainWindow", u"Wait Queue", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("mainWindow", u"Scheduler_cache", None))
        self.label_3.setText(QCoreApplication.translate("mainWindow", u"normal_task_list", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), QCoreApplication.translate("mainWindow", u"Nomal_task_list", None))
        self.label_5.setText(QCoreApplication.translate("mainWindow", u"low_priority_list", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), QCoreApplication.translate("mainWindow", u"Low_priority_list", None))
        self.detail_text.setHtml(QCoreApplication.translate("mainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Microsoft YaHei UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">task details</p></body></html>", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("mainWindow", u"Details", None))
    # retranslateUi

