# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'user_manage_ui.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QSizePolicy, QSpacerItem, QStatusBar,
    QToolBar, QWidget)

from pyqcat_visage.gui.widgets.manager.table_view_groups import QTableViewGroupWidget
from pyqcat_visage.gui.widgets.manager.table_view_users import QTableViewUserWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(940, 646)
        self.actionAll_Groups = QAction(MainWindow)
        self.actionAll_Groups.setObjectName(u"actionAll_Groups")
        self.actionCreate_Group = QAction(MainWindow)
        self.actionCreate_Group.setObjectName(u"actionCreate_Group")
        self.actionChange_Password = QAction(MainWindow)
        self.actionChange_Password.setObjectName(u"actionChange_Password")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout_2 = QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")

        self.horizontalLayout_3.addWidget(self.label)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer)

        self.username_edit = QLineEdit(self.groupBox)
        self.username_edit.setObjectName(u"username_edit")
        self.username_edit.setEnabled(False)

        self.horizontalLayout_3.addWidget(self.username_edit)

        self.horizontalLayout_3.setStretch(0, 2)
        self.horizontalLayout_3.setStretch(1, 2)
        self.horizontalLayout_3.setStretch(2, 6)

        self.gridLayout_2.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_4.addWidget(self.label_2)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_2)

        self.group_edit = QLineEdit(self.groupBox)
        self.group_edit.setObjectName(u"group_edit")
        self.group_edit.setEnabled(False)

        self.horizontalLayout_4.addWidget(self.group_edit)

        self.horizontalLayout_4.setStretch(0, 2)
        self.horizontalLayout_4.setStretch(1, 2)
        self.horizontalLayout_4.setStretch(2, 6)

        self.gridLayout_2.addLayout(self.horizontalLayout_4, 1, 0, 1, 1)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_5.addWidget(self.label_3)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_3)

        self.email_edit = QLineEdit(self.groupBox)
        self.email_edit.setObjectName(u"email_edit")
        self.email_edit.setEnabled(False)

        self.horizontalLayout_5.addWidget(self.email_edit)

        self.horizontalLayout_5.setStretch(0, 2)
        self.horizontalLayout_5.setStretch(1, 2)
        self.horizontalLayout_5.setStretch(2, 6)

        self.gridLayout_2.addLayout(self.horizontalLayout_5, 2, 0, 1, 1)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.is_super_check = QCheckBox(self.groupBox)
        self.is_super_check.setObjectName(u"is_super_check")
        self.is_super_check.setEnabled(False)

        self.horizontalLayout_6.addWidget(self.is_super_check)

        self.is_admin_check = QCheckBox(self.groupBox)
        self.is_admin_check.setObjectName(u"is_admin_check")
        self.is_admin_check.setEnabled(False)

        self.horizontalLayout_6.addWidget(self.is_admin_check)


        self.gridLayout_2.addLayout(self.horizontalLayout_6, 3, 0, 1, 1)


        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 1)

        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.horizontalLayout = QHBoxLayout(self.groupBox_2)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.table_view_group = QTableViewGroupWidget(self.groupBox_2)
        self.table_view_group.setObjectName(u"table_view_group")

        self.horizontalLayout.addWidget(self.table_view_group)


        self.gridLayout.addWidget(self.groupBox_2, 0, 1, 1, 1)

        self.groupBox_3 = QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.horizontalLayout_2 = QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.table_view_user = QTableViewUserWidget(self.groupBox_3)
        self.table_view_user.setObjectName(u"table_view_user")

        self.horizontalLayout_2.addWidget(self.table_view_user)


        self.gridLayout.addWidget(self.groupBox_3, 1, 0, 1, 2)

        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QToolBar(MainWindow)
        self.toolBar.setObjectName(u"toolBar")
        self.toolBar.setEnabled(True)
        self.toolBar.setFloatable(True)
        MainWindow.addToolBar(Qt.TopToolBarArea, self.toolBar)

        self.toolBar.addAction(self.actionCreate_Group)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionChange_Password)

        self.retranslateUi(MainWindow)
        self.actionAll_Groups.triggered.connect(MainWindow.query_all_groups)
        self.actionCreate_Group.triggered.connect(MainWindow.create_group)
        self.actionChange_Password.triggered.connect(MainWindow.change_password)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"User Manager", None))
        self.actionAll_Groups.setText(QCoreApplication.translate("MainWindow", u"All Groups", None))
        self.actionCreate_Group.setText(QCoreApplication.translate("MainWindow", u"Create Group", None))
        self.actionChange_Password.setText(QCoreApplication.translate("MainWindow", u"Change Password", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"ID Card", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"username", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"group", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"email", None))
        self.is_super_check.setText(QCoreApplication.translate("MainWindow", u"is_super", None))
        self.is_admin_check.setText(QCoreApplication.translate("MainWindow", u"is_admin", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Group", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"User", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar", None))
    # retranslateUi

