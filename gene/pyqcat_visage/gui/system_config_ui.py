# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'system_config_ui.ui'
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
from PySide6.QtWidgets import (QApplication, QGroupBox, QHBoxLayout, QHeaderView,
    QLineEdit, QMainWindow, QPushButton, QSizePolicy,
    QStatusBar, QVBoxLayout, QWidget)

from .widgets.config.tree_view_config import QTreeViewConfig

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(566, 651)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setAlignment(Qt.AlignCenter)
        self.horizontalLayout_3 = QHBoxLayout(self.groupBox)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 9, 0, 0)
        self.tree_view_config = QTreeViewConfig(self.groupBox)
        self.tree_view_config.setObjectName(u"tree_view_config")
        self.tree_view_config.setSortingEnabled(False)
        self.tree_view_config.setExpandsOnDoubleClick(True)

        self.horizontalLayout_3.addWidget(self.tree_view_config)


        self.verticalLayout.addWidget(self.groupBox)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.file_edit = QLineEdit(self.centralwidget)
        self.file_edit.setObjectName(u"file_edit")

        self.horizontalLayout_2.addWidget(self.file_edit)

        self.browse_button = QPushButton(self.centralwidget)
        self.browse_button.setObjectName(u"browse_button")
        icon = QIcon()
        icon.addFile(u"_imgs/import.png", QSize(), QIcon.Normal, QIcon.Off)
        self.browse_button.setIcon(icon)

        self.horizontalLayout_2.addWidget(self.browse_button)

        self.import_button = QPushButton(self.centralwidget)
        self.import_button.setObjectName(u"import_button")
        icon1 = QIcon()
        icon1.addFile(u"_imgs/save_as.png", QSize(), QIcon.Normal, QIcon.Off)
        self.import_button.setIcon(icon1)

        self.horizontalLayout_2.addWidget(self.import_button)

        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setObjectName(u"pushButton")
        icon2 = QIcon()
        icon2.addFile(u"_imgs/ok.png", QSize(), QIcon.Normal, QIcon.Off)
        self.pushButton.setIcon(icon2)

        self.horizontalLayout_2.addWidget(self.pushButton)

        self.horizontalLayout_2.setStretch(0, 14)
        self.horizontalLayout_2.setStretch(1, 2)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.verticalLayout.setStretch(0, 7)
        self.verticalLayout.setStretch(1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.browse_button.clicked.connect(MainWindow.import_config)
        self.import_button.clicked.connect(MainWindow.export_config)
        self.pushButton.clicked.connect(MainWindow.save_config)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"System Config", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"System Config", None))
        self.file_edit.setInputMask("")
        self.file_edit.setText("")
        self.file_edit.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Load local conig file...", None))
        self.browse_button.setText(QCoreApplication.translate("MainWindow", u"Import", None))
        self.import_button.setText(QCoreApplication.translate("MainWindow", u"Export", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"Save", None))
    # retranslateUi

