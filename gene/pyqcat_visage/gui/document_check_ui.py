# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'document_check_ui.ui'
##
## Created by: Qt User Interface Compiler version 6.3.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QMetaObject, QSize, Qt)
from PySide6.QtGui import (QAction, QIcon)
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout,
                               QGroupBox, QLabel, QLineEdit,
                               QSizePolicy, QSplitter, QStatusBar,
                               QToolBar, QVBoxLayout, QWidget)

from .widgets.document.doc_tree_view import QTreeViewDocument
from .widgets.document.schedule_chart_view import QCharViewSchedule
from pyqcat_visage.gui.widgets.combox_custom.combox_multi import QMultiComboBox


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1115, 705)
        self.actionQuery = QAction(MainWindow)
        self.actionQuery.setObjectName(u"actionQuery")
        icon = QIcon()
        icon.addFile(u":/setting.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionQuery.setIcon(icon)
        self.actionEnlarge = QAction(MainWindow)
        self.actionEnlarge.setObjectName(u"actionEnlarge")
        icon1 = QIcon()
        icon1.addFile(u":/full-screen.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionEnlarge.setIcon(icon1)
        self.actionNarrow = QAction(MainWindow)
        self.actionNarrow.setObjectName(u"actionNarrow")
        icon2 = QIcon()
        icon2.addFile(u":/collapse.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionNarrow.setIcon(icon2)
        self.actionReset = QAction(MainWindow)
        self.actionReset.setObjectName(u"actionReset")
        icon3 = QIcon()
        icon3.addFile(u":/reset.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionReset.setIcon(icon3)
        self.actionCompare = QAction(MainWindow)
        self.actionCompare.setObjectName(u"actionCompare")
        icon4 = QIcon()
        icon4.addFile(u":/live-fill.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionCompare.setIcon(icon4)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_4 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        self.gridLayout = QGridLayout(self.widget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label = QLabel(self.widget)
        self.label.setObjectName(u"label")
        self.label.setLayoutDirection(Qt.LeftToRight)

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.exp_id_edit = QLineEdit(self.widget)
        self.exp_id_edit.setObjectName(u"exp_id_edit")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.exp_id_edit.sizePolicy().hasHeightForWidth())
        self.exp_id_edit.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.exp_id_edit, 0, 1, 1, 1)

        self.show_delay = QCheckBox(self.widget)
        self.show_delay.setObjectName(u"show_delay")

        self.gridLayout.addWidget(self.show_delay, 0, 2, 1, 1)

        self.fix_canvas = QCheckBox(self.widget)
        self.fix_canvas.setObjectName(u"fix_canvas")

        self.gridLayout.addWidget(self.fix_canvas, 0, 3, 1, 1)

        self.label_5 = QLabel(self.widget)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 0, 4, 1, 1)

        self.combo_com = QMultiComboBox(self.widget)
        self.combo_com.setObjectName(u"combo_com")
        sizePolicy.setHeightForWidth(self.combo_com.sizePolicy().hasHeightForWidth())
        self.combo_com.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.combo_com, 0, 5, 1, 1)

        self.label_2 = QLabel(self.widget)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 0, 6, 1, 1)

        self.module_com = QComboBox(self.widget)
        self.module_com.setObjectName(u"module_com")
        sizePolicy.setHeightForWidth(self.module_com.sizePolicy().hasHeightForWidth())
        self.module_com.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.module_com, 0, 7, 1, 1)

        self.label_3 = QLabel(self.widget)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 0, 8, 1, 1)

        self.channel_com = QComboBox(self.widget)
        self.channel_com.setObjectName(u"channel_com")
        sizePolicy.setHeightForWidth(self.channel_com.sizePolicy().hasHeightForWidth())
        self.channel_com.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.channel_com, 0, 9, 1, 1)

        self.label_4 = QLabel(self.widget)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 0, 10, 1, 1)

        self.loop_com = QComboBox(self.widget)
        self.loop_com.setObjectName(u"loop_com")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.loop_com.sizePolicy().hasHeightForWidth())
        self.loop_com.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.loop_com, 0, 11, 1, 1)

        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 3)
        self.gridLayout.setColumnStretch(2, 1)
        self.gridLayout.setColumnStretch(3, 1)
        self.gridLayout.setColumnStretch(4, 1)
        self.gridLayout.setColumnStretch(5, 3)
        self.gridLayout.setColumnStretch(6, 1)
        self.gridLayout.setColumnStretch(7, 2)
        self.gridLayout.setColumnStretch(8, 1)
        self.gridLayout.setColumnStretch(9, 1)
        self.gridLayout.setColumnStretch(10, 1)
        self.gridLayout.setColumnStretch(11, 1)

        self.verticalLayout_4.addWidget(self.widget)

        self.widget_2 = QWidget(self.centralwidget)
        self.widget_2.setObjectName(u"widget_2")
        self.verticalLayout_3 = QVBoxLayout(self.widget_2)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.splitter = QSplitter(self.widget_2)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.groupBox = QGroupBox(self.splitter)
        self.groupBox.setObjectName(u"groupBox")
        self.verticalLayout_2 = QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.treeView = QTreeViewDocument(self.groupBox)
        self.treeView.setObjectName(u"treeView")

        self.verticalLayout_2.addWidget(self.treeView)

        self.splitter.addWidget(self.groupBox)
        self.groupBox_3 = QGroupBox(self.splitter)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.verticalLayout = QVBoxLayout(self.groupBox_3)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.schedule_chart_view = QCharViewSchedule(self.groupBox_3)
        self.schedule_chart_view.setObjectName(u"schedule_chart_view")

        self.verticalLayout.addWidget(self.schedule_chart_view)

        self.splitter.addWidget(self.groupBox_3)

        self.verticalLayout_3.addWidget(self.splitter)


        self.verticalLayout_4.addWidget(self.widget_2)

        self.verticalLayout_4.setStretch(0, 1)
        self.verticalLayout_4.setStretch(1, 9)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QToolBar(MainWindow)
        self.toolBar.setObjectName(u"toolBar")
        self.toolBar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        MainWindow.addToolBar(Qt.TopToolBarArea, self.toolBar)

        self.toolBar.addAction(self.actionQuery)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionEnlarge)
        self.toolBar.addAction(self.actionNarrow)
        self.toolBar.addAction(self.actionReset)
        self.toolBar.addAction(self.actionCompare)

        self.retranslateUi(MainWindow)
        self.actionQuery.triggered.connect(MainWindow.query)
        self.actionNarrow.triggered.connect(MainWindow.narrow)
        self.actionReset.triggered.connect(MainWindow.reset)
        self.actionEnlarge.triggered.connect(MainWindow.enlarge)
        self.module_com.currentTextChanged.connect(MainWindow.change_module)
        self.channel_com.currentTextChanged.connect(MainWindow.change_channel)
        self.loop_com.currentTextChanged.connect(MainWindow.change_loop)
        self.actionCompare.triggered.connect(MainWindow.compare_pulse)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Document Check", None))
        self.actionQuery.setText(QCoreApplication.translate("MainWindow", u"Query Task", None))
#if QT_CONFIG(tooltip)
        self.actionQuery.setToolTip(QCoreApplication.translate("MainWindow", u"Query a history expeirment", None))
#endif // QT_CONFIG(tooltip)
        self.actionEnlarge.setText(QCoreApplication.translate("MainWindow", u"Enlarge", None))
        self.actionNarrow.setText(QCoreApplication.translate("MainWindow", u"Narrow", None))
        self.actionReset.setText(QCoreApplication.translate("MainWindow", u"Reset", None))
        self.actionCompare.setText(QCoreApplication.translate("MainWindow", u"Compare", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"EXP ID", None))
        self.show_delay.setText(QCoreApplication.translate("MainWindow", u"Show Delay", None))
        self.fix_canvas.setText(QCoreApplication.translate("MainWindow", u"Fixed Canvas", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"combo", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"module", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"channel", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"loop", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Experiment Document", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Schedule", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar", None))
    # retranslateUi

