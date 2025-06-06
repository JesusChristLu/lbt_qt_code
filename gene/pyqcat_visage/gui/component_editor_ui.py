# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'component_editor_ui.ui'
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
from PySide6.QtWidgets import (QApplication, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QSizePolicy,
    QSpacerItem, QSpinBox, QSplitter, QStatusBar,
    QToolBar, QVBoxLayout, QWidget)

from .widgets.component.table_view_component import QTableViewComponentWidget
from .widgets.component.tree_view_component import QTreeViewComponentWidget
from .widgets.result.table_view_dat import QTableViewDat


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1215, 736)
        icon = QIcon()
        icon.addFile(u":/qubit.png", QSize(), QIcon.Normal, QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.actionImport = QAction(MainWindow)
        self.actionImport.setObjectName(u"actionImport")
        icon1 = QIcon()
        icon1.addFile(u":/local.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionImport.setIcon(icon1)
        self.actionToFile = QAction(MainWindow)
        self.actionToFile.setObjectName(u"actionToFile")
        icon2 = QIcon()
        icon2.addFile(u":/file-code.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionToFile.setIcon(icon2)
        self.actionSave = QAction(MainWindow)
        self.actionSave.setObjectName(u"actionSave")
        icon3 = QIcon()
        icon3.addFile(u":/save.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionSave.setIcon(icon3)
        self.actionSaveAs = QAction(MainWindow)
        self.actionSaveAs.setObjectName(u"actionSaveAs")
        icon4 = QIcon()
        icon4.addFile(u":/save_as.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionSaveAs.setIcon(icon4)
        self.actionQuery = QAction(MainWindow)
        self.actionQuery.setObjectName(u"actionQuery")
        icon5 = QIcon()
        icon5.addFile(u":/database-search.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionQuery.setIcon(icon5)
        self.actionRefresh = QAction(MainWindow)
        self.actionRefresh.setObjectName(u"actionRefresh")
        icon6 = QIcon()
        icon6.addFile(u":/refresh.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionRefresh.setIcon(icon6)
        self.actionQueryAll = QAction(MainWindow)
        self.actionQueryAll.setObjectName(u"actionQueryAll")
        icon7 = QIcon()
        icon7.addFile(u":/database-download.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionQueryAll.setIcon(icon7)
        self.actionChipCheck = QAction(MainWindow)
        self.actionChipCheck.setObjectName(u"actionChipCheck")
        icon8 = QIcon()
        icon8.addFile(u":/tool.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionChipCheck.setIcon(icon8)
        self.actionQueryHistory = QAction(MainWindow)
        self.actionQueryHistory.setObjectName(u"actionQueryHistory")
        icon9 = QIcon()
        icon9.addFile(u":/database-success.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionQueryHistory.setIcon(icon9)
        self.actionRevertBits = QAction(MainWindow)
        self.actionRevertBits.setObjectName(u"actionRevertBits")
        icon10 = QIcon()
        icon10.addFile(u":/update.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionRevertBits.setIcon(icon10)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_5 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.splitter_2 = QSplitter(self.centralwidget)
        self.splitter_2.setObjectName(u"splitter_2")
        self.splitter_2.setOrientation(Qt.Horizontal)
        self.layoutWidget = QWidget(self.splitter_2)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.verticalLayout_2 = QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.describe_group = QGroupBox(self.layoutWidget)
        self.describe_group.setObjectName(u"describe_group")
        self.horizontalLayout_5 = QHBoxLayout(self.describe_group)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.child_widget = QWidget(self.describe_group)
        self.child_widget.setObjectName(u"child_widget")
        self.verticalLayout_3 = QVBoxLayout(self.child_widget)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_2 = QLabel(self.child_widget)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_4.addWidget(self.label_2)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_2)

        self.qid_label = QLineEdit(self.child_widget)
        self.qid_label.setObjectName(u"qid_label")

        self.horizontalLayout_4.addWidget(self.qid_label)

        self.horizontalLayout_4.setStretch(0, 1)
        self.horizontalLayout_4.setStretch(1, 1)
        self.horizontalLayout_4.setStretch(2, 5)

        self.verticalLayout_3.addLayout(self.horizontalLayout_4)

        self.name_layout = QHBoxLayout()
        self.name_layout.setObjectName(u"name_layout")
        self.name_label = QLabel(self.child_widget)
        self.name_label.setObjectName(u"name_label")

        self.name_layout.addWidget(self.name_label)

        self.name_space = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.name_layout.addItem(self.name_space)

        self.name_edit = QLineEdit(self.child_widget)
        self.name_edit.setObjectName(u"name_edit")

        self.name_layout.addWidget(self.name_edit)

        self.name_layout.setStretch(0, 1)
        self.name_layout.setStretch(1, 1)
        self.name_layout.setStretch(2, 5)

        self.verticalLayout_3.addLayout(self.name_layout)

        self.user_layout = QHBoxLayout()
        self.user_layout.setObjectName(u"user_layout")
        self.user_label = QLabel(self.child_widget)
        self.user_label.setObjectName(u"user_label")

        self.user_layout.addWidget(self.user_label)

        self.user_space = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.user_layout.addItem(self.user_space)

        self.user_edit = QLineEdit(self.child_widget)
        self.user_edit.setObjectName(u"user_edit")

        self.user_layout.addWidget(self.user_edit)

        self.user_layout.setStretch(0, 1)
        self.user_layout.setStretch(1, 1)
        self.user_layout.setStretch(2, 5)

        self.verticalLayout_3.addLayout(self.user_layout)

        self.point_layout = QHBoxLayout()
        self.point_layout.setObjectName(u"point_layout")
        self.point_label = QLabel(self.child_widget)
        self.point_label.setObjectName(u"point_label")

        self.point_layout.addWidget(self.point_label)

        self.point_space = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.point_layout.addItem(self.point_space)

        self.point_edit = QLineEdit(self.child_widget)
        self.point_edit.setObjectName(u"point_edit")

        self.point_layout.addWidget(self.point_edit)

        self.point_layout.setStretch(0, 1)
        self.point_layout.setStretch(1, 1)
        self.point_layout.setStretch(2, 5)

        self.verticalLayout_3.addLayout(self.point_layout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label = QLabel(self.child_widget)
        self.label.setObjectName(u"label")

        self.horizontalLayout_2.addWidget(self.label)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.env_name_edit = QLineEdit(self.child_widget)
        self.env_name_edit.setObjectName(u"env_name_edit")

        self.horizontalLayout_2.addWidget(self.env_name_edit)

        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 1)
        self.horizontalLayout_2.setStretch(2, 5)

        self.verticalLayout_3.addLayout(self.horizontalLayout_2)

        self.sample_layout = QHBoxLayout()
        self.sample_layout.setObjectName(u"sample_layout")
        self.sample_layout.setContentsMargins(0, 0, 0, 0)
        self.sample_label = QLabel(self.child_widget)
        self.sample_label.setObjectName(u"sample_label")

        self.sample_layout.addWidget(self.sample_label)

        self.sample_spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.sample_layout.addItem(self.sample_spacer)

        self.sample_editor = QLineEdit(self.child_widget)
        self.sample_editor.setObjectName(u"sample_editor")

        self.sample_layout.addWidget(self.sample_editor)

        self.sample_layout.setStretch(0, 1)
        self.sample_layout.setStretch(1, 1)
        self.sample_layout.setStretch(2, 5)

        self.verticalLayout_3.addLayout(self.sample_layout)


        self.horizontalLayout_5.addWidget(self.child_widget)

        self.bit_pic_label = QLabel(self.describe_group)
        self.bit_pic_label.setObjectName(u"bit_pic_label")

        self.horizontalLayout_5.addWidget(self.bit_pic_label)

        self.horizontalLayout_5.setStretch(0, 3)
        self.horizontalLayout_5.setStretch(1, 1)

        self.verticalLayout_2.addWidget(self.describe_group)

        self.index_group = QGroupBox(self.layoutWidget)
        self.index_group.setObjectName(u"index_group")
        self.verticalLayout = QVBoxLayout(self.index_group)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.page_layout = QHBoxLayout()
        self.page_layout.setObjectName(u"page_layout")
        self.label_3 = QLabel(self.index_group)
        self.label_3.setObjectName(u"label_3")

        self.page_layout.addWidget(self.label_3)

        self.page_spinBox = QSpinBox(self.index_group)
        self.page_spinBox.setObjectName(u"page_spinBox")
        self.page_spinBox.setEnabled(True)
        self.page_spinBox.setMinimum(1)

        self.page_layout.addWidget(self.page_spinBox)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.page_layout.addItem(self.horizontalSpacer_4)

        self.label_4 = QLabel(self.index_group)
        self.label_4.setObjectName(u"label_4")

        self.page_layout.addWidget(self.label_4)

        self.volume_spinBox = QSpinBox(self.index_group)
        self.volume_spinBox.setObjectName(u"volume_spinBox")
        self.volume_spinBox.setEnabled(True)
        self.volume_spinBox.setMinimum(1)
        self.volume_spinBox.setMaximum(10000000)
        self.volume_spinBox.setValue(10)
        self.volume_spinBox.setDisplayIntegerBase(10)

        self.page_layout.addWidget(self.volume_spinBox)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.page_layout.addItem(self.horizontalSpacer_3)

        self.page_layout.setStretch(0, 1)
        self.page_layout.setStretch(1, 1)
        self.page_layout.setStretch(2, 1)
        self.page_layout.setStretch(3, 1)
        self.page_layout.setStretch(4, 1)
        self.page_layout.setStretch(5, 1)

        self.verticalLayout.addLayout(self.page_layout)

        self.table_view_component = QTableViewComponentWidget(MainWindow)
        self.table_view_component.setObjectName(u"table_view_component")
        self.table_view_component.setAutoFillBackground(False)

        self.verticalLayout.addWidget(self.table_view_component)


        self.verticalLayout_2.addWidget(self.index_group)

        self.splitter_2.addWidget(self.layoutWidget)
        self.editor_group = QGroupBox(self.splitter_2)
        self.editor_group.setObjectName(u"editor_group")
        self.verticalLayout_4 = QVBoxLayout(self.editor_group)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.splitter = QSplitter(self.editor_group)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Vertical)
        self.tree_view_component = QTreeViewComponentWidget(self.splitter)
        self.tree_view_component.setObjectName(u"tree_view_component")
        self.splitter.addWidget(self.tree_view_component)
        self.table_view_dat = QTableViewDat(self.splitter)
        self.table_view_dat.setObjectName(u"table_view_dat")
        self.splitter.addWidget(self.table_view_dat)

        self.verticalLayout_4.addWidget(self.splitter)

        self.splitter_2.addWidget(self.editor_group)

        self.verticalLayout_5.addWidget(self.splitter_2)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QToolBar(MainWindow)
        self.toolBar.setObjectName(u"toolBar")
        self.toolBar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        MainWindow.addToolBar(Qt.TopToolBarArea, self.toolBar)

        self.toolBar.addAction(self.actionQueryAll)
        self.toolBar.addAction(self.actionQuery)
        self.toolBar.addAction(self.actionQueryHistory)
        self.toolBar.addAction(self.actionSave)
        self.toolBar.addAction(self.actionImport)
        self.toolBar.addAction(self.actionToFile)
        self.toolBar.addAction(self.actionSaveAs)
        self.toolBar.addAction(self.actionRefresh)
        self.toolBar.addAction(self.actionChipCheck)
        self.toolBar.addAction(self.actionRevertBits)

        self.retranslateUi(MainWindow)
        self.actionQueryAll.triggered.connect(MainWindow.query_all)
        self.actionQuery.triggered.connect(MainWindow.query_one)
        self.actionSave.triggered.connect(MainWindow.save_one)
        self.actionImport.triggered.connect(MainWindow.bit_import)
        self.actionToFile.triggered.connect(MainWindow.save_to_file)
        self.actionRefresh.triggered.connect(MainWindow.refresh)
        self.actionSaveAs.triggered.connect(MainWindow.save_as)
        self.actionChipCheck.triggered.connect(MainWindow.chip_check)
        self.actionQueryHistory.triggered.connect(MainWindow.query_history)
        self.page_spinBox.valueChanged.connect(MainWindow.page_change)
        self.volume_spinBox.valueChanged.connect(MainWindow.volume_change)
        self.actionRevertBits.triggered.connect(MainWindow.show_revert_bits)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Qubit Editor", None))
        self.actionImport.setText(QCoreApplication.translate("MainWindow", u"Import", None))
        self.actionToFile.setText(QCoreApplication.translate("MainWindow", u"ToFile", None))
        self.actionSave.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.actionSaveAs.setText(QCoreApplication.translate("MainWindow", u"SaveAs", None))
        self.actionQuery.setText(QCoreApplication.translate("MainWindow", u"Query", None))
        self.actionRefresh.setText(QCoreApplication.translate("MainWindow", u"Refresh", None))
        self.actionQueryAll.setText(QCoreApplication.translate("MainWindow", u"QueryAll", None))
#if QT_CONFIG(tooltip)
        self.actionQueryAll.setToolTip(QCoreApplication.translate("MainWindow", u"Query All Qubit", None))
#endif // QT_CONFIG(tooltip)
        self.actionChipCheck.setText(QCoreApplication.translate("MainWindow", u"ChipCheck", None))
        self.actionQueryHistory.setText(QCoreApplication.translate("MainWindow", u"QueryHistory", None))
        self.actionRevertBits.setText(QCoreApplication.translate("MainWindow", u"RevertBits", None))
        self.describe_group.setTitle(QCoreApplication.translate("MainWindow", u"Component Describe", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"id", None))
        self.name_label.setText(QCoreApplication.translate("MainWindow", u"name", None))
        self.user_label.setText(QCoreApplication.translate("MainWindow", u"user", None))
        self.point_label.setText(QCoreApplication.translate("MainWindow", u"point", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"env name", None))
        self.sample_label.setText(QCoreApplication.translate("MainWindow", u"sample", None))
        self.bit_pic_label.setText("")
        self.index_group.setTitle(QCoreApplication.translate("MainWindow", u"Component Collector", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"        Page", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"      Volume", None))
        self.editor_group.setTitle(QCoreApplication.translate("MainWindow", u"Component Editor", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar", None))
    # retranslateUi

