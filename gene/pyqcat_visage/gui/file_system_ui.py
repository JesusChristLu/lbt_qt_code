# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'file_system_ui.ui'
##
## Created by: Qt User Interface Compiler version 6.4.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QHBoxLayout,
    QHeaderView, QLineEdit, QListView, QMainWindow,
    QScrollArea, QSizePolicy, QSplitter, QStatusBar,
    QTextEdit, QToolBar, QVBoxLayout, QWidget)

from pyqcat_visage.gui.widgets.result.table_view_dat import QTableViewDat
import pyqcat_visage.gui._imgs_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1121, 700)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QSize(0, 0))
        self.actionRefresh = QAction(MainWindow)
        self.actionRefresh.setObjectName(u"actionRefresh")
        icon = QIcon()
        icon.addFile(u":/reset.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionRefresh.setIcon(icon)
        self.actionRefresh.setIconVisibleInMenu(True)
        self.actionCurrent = QAction(MainWindow)
        self.actionCurrent.setObjectName(u"actionCurrent")
        icon1 = QIcon()
        icon1.addFile(u":/file-code.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionCurrent.setIcon(icon1)
        self.actionPre = QAction(MainWindow)
        self.actionPre.setObjectName(u"actionPre")
        icon2 = QIcon()
        icon2.addFile(u":/cancel.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionPre.setIcon(icon2)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy1)
        self.verticalLayout_4 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(10, 0, 10, 0)
        self.type_combox = QComboBox(self.centralwidget)
        self.type_combox.addItem("")
        self.type_combox.addItem("")
        self.type_combox.setObjectName(u"type_combox")
        self.type_combox.setMinimumSize(QSize(100, 0))
        self.type_combox.setMaximumSize(QSize(16777215, 28))

        self.horizontalLayout.addWidget(self.type_combox)

        self.input_edit = QLineEdit(self.centralwidget)
        self.input_edit.setObjectName(u"input_edit")
        self.input_edit.setMaximumSize(QSize(16777215, 28))

        self.horizontalLayout.addWidget(self.input_edit)

        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 10)

        self.verticalLayout_4.addLayout(self.horizontalLayout)

        self.splitter_3 = QSplitter(self.centralwidget)
        self.splitter_3.setObjectName(u"splitter_3")
        self.splitter_3.setOrientation(Qt.Horizontal)
        self.splitter_3.setHandleWidth(1)
        self.widget = QWidget(self.splitter_3)
        self.widget.setObjectName(u"widget")
        self.verticalLayout = QVBoxLayout(self.widget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.listView = QListView(self.widget)
        self.listView.setObjectName(u"listView")
        self.listView.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.verticalLayout.addWidget(self.listView)

        self.splitter_3.addWidget(self.widget)
        self.widget_2 = QWidget(self.splitter_3)
        self.widget_2.setObjectName(u"widget_2")
        self.verticalLayout_2 = QVBoxLayout(self.widget_2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.splitter = QSplitter(self.widget_2)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Vertical)
        self.textEdit = QTextEdit(self.splitter)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setReadOnly(True)
        self.splitter.addWidget(self.textEdit)
        self.tableView = QTableViewDat(self.splitter)
        self.tableView.setObjectName(u"tableView")
        self.splitter.addWidget(self.tableView)
        self.scrollArea = QScrollArea(self.splitter)
        self.scrollArea.setObjectName(u"scrollArea")
        sizePolicy1.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy1)
        self.scrollArea.setFocusPolicy(Qt.StrongFocus)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 531, 69))
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.splitter.addWidget(self.scrollArea)

        self.verticalLayout_2.addWidget(self.splitter)

        self.splitter_3.addWidget(self.widget_2)

        self.verticalLayout_4.addWidget(self.splitter_3)

        self.verticalLayout_4.setStretch(0, 1)
        self.verticalLayout_4.setStretch(1, 20)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QToolBar(MainWindow)
        self.toolBar.setObjectName(u"toolBar")
        self.toolBar.setFocusPolicy(Qt.NoFocus)
        self.toolBar.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.toolBar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        MainWindow.addToolBar(Qt.TopToolBarArea, self.toolBar)

        self.toolBar.addAction(self.actionCurrent)
        self.toolBar.addAction(self.actionRefresh)
        self.toolBar.addAction(self.actionPre)

        self.retranslateUi(MainWindow)
        self.actionRefresh.triggered.connect(MainWindow.refresh_dirs)
        self.actionCurrent.triggered.connect(MainWindow.last_dirs)
        self.actionPre.triggered.connect(MainWindow.pre_page)
        self.type_combox.currentTextChanged.connect(MainWindow.switch)
        self.input_edit.returnPressed.connect(MainWindow.find_path)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"File System", None))
        self.actionRefresh.setText(QCoreApplication.translate("MainWindow", u"Refresh", None))
#if QT_CONFIG(tooltip)
        self.actionRefresh.setToolTip(QCoreApplication.translate("MainWindow", u"Experiment Root Dirs", None))
#endif // QT_CONFIG(tooltip)
        self.actionCurrent.setText(QCoreApplication.translate("MainWindow", u"Current", None))
#if QT_CONFIG(tooltip)
        self.actionCurrent.setToolTip(QCoreApplication.translate("MainWindow", u"Current Experment Dirs", None))
#endif // QT_CONFIG(tooltip)
        self.actionPre.setText(QCoreApplication.translate("MainWindow", u"PrePage", None))
#if QT_CONFIG(tooltip)
        self.actionPre.setToolTip(QCoreApplication.translate("MainWindow", u"PrePage", None))
#endif // QT_CONFIG(tooltip)
        self.type_combox.setItemText(0, QCoreApplication.translate("MainWindow", u"local", None))
        self.type_combox.setItemText(1, QCoreApplication.translate("MainWindow", u"s3", None))

        self.toolBar.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar", None))
    # retranslateUi

