# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'heatmap_ui.ui'
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
from PySide6.QtWidgets import (QAbstractItemView, QAbstractSpinBox, QApplication, QCheckBox,
    QComboBox, QDoubleSpinBox, QGraphicsView, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMainWindow, QPushButton, QSizePolicy,
    QSpacerItem, QSpinBox, QSplitter, QStatusBar,
    QTabWidget, QTableView, QToolBar, QVBoxLayout,
    QWidget)

from .widgets.heatmap.divide_tree_view import DivideTreeView
from .widgets.heatmap.struct_tree_view import StructTreeView
import pyqcat_visage.gui._imgs_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1139, 678)
        MainWindow.setContextMenuPolicy(Qt.ActionsContextMenu)
        icon = QIcon()
        icon.addFile(u":/logo.png", QSize(), QIcon.Normal, QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.actionSavetoDB = QAction(MainWindow)
        self.actionSavetoDB.setObjectName(u"actionSavetoDB")
        icon1 = QIcon()
        icon1.addFile(u":/database-sync.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionSavetoDB.setIcon(icon1)
        self.actionQuery = QAction(MainWindow)
        self.actionQuery.setObjectName(u"actionQuery")
        icon2 = QIcon()
        icon2.addFile(u":/database-search.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionQuery.setIcon(icon2)
        self.actionLoadFromLocal = QAction(MainWindow)
        self.actionLoadFromLocal.setObjectName(u"actionLoadFromLocal")
        icon3 = QIcon()
        icon3.addFile(u":/local.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionLoadFromLocal.setIcon(icon3)
        self.actionSaveLocal = QAction(MainWindow)
        self.actionSaveLocal.setObjectName(u"actionSaveLocal")
        icon4 = QIcon()
        icon4.addFile(u":/file-code.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionSaveLocal.setIcon(icon4)
        self.actionRefresh = QAction(MainWindow)
        self.actionRefresh.setObjectName(u"actionRefresh")
        icon5 = QIcon()
        icon5.addFile(u":/refresh.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionRefresh.setIcon(icon5)
        self.actionSaveHeatmap = QAction(MainWindow)
        self.actionSaveHeatmap.setObjectName(u"actionSaveHeatmap")
        icon6 = QIcon()
        icon6.addFile(u":/screenshot.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionSaveHeatmap.setIcon(icon6)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_3 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.heat_map_tab = QWidget()
        self.heat_map_tab.setObjectName(u"heat_map_tab")
        self.verticalLayout_4 = QVBoxLayout(self.heat_map_tab)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.splitter = QSplitter(self.heat_map_tab)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.layoutWidget = QWidget(self.splitter)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.verticalLayout_2 = QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.graphicsViewHeatmap = QGraphicsView(self.layoutWidget)
        self.graphicsViewHeatmap.setObjectName(u"graphicsViewHeatmap")
        self.graphicsViewHeatmap.setMinimumSize(QSize(0, 0))

        self.verticalLayout_2.addWidget(self.graphicsViewHeatmap)

        self.splitter.addWidget(self.layoutWidget)
        self.layoutWidget1 = QWidget(self.splitter)
        self.layoutWidget1.setObjectName(u"layoutWidget1")
        self.verticalLayout = QVBoxLayout(self.layoutWidget1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_1 = QGridLayout()
        self.gridLayout_1.setObjectName(u"gridLayout_1")
        self.label_21 = QLabel(self.layoutWidget1)
        self.label_21.setObjectName(u"label_21")

        self.gridLayout_1.addWidget(self.label_21, 0, 0, 1, 1)

        self.component_combox = QComboBox(self.layoutWidget1)
        self.component_combox.addItem("")
        self.component_combox.addItem("")
        self.component_combox.addItem("")
        self.component_combox.setObjectName(u"component_combox")

        self.gridLayout_1.addWidget(self.component_combox, 0, 1, 1, 1)

        self.label_11 = QLabel(self.layoutWidget1)
        self.label_11.setObjectName(u"label_11")

        self.gridLayout_1.addWidget(self.label_11, 0, 2, 1, 1)

        self.show_value_box = QCheckBox(self.layoutWidget1)
        self.show_value_box.setObjectName(u"show_value_box")
#if QT_CONFIG(tooltip)
        self.show_value_box.setToolTip(u"")
#endif // QT_CONFIG(tooltip)
        self.show_value_box.setToolTipDuration(0)
        self.show_value_box.setChecked(True)

        self.gridLayout_1.addWidget(self.show_value_box, 1, 0, 1, 1)

        self.cmap_box = QComboBox(self.layoutWidget1)
        self.cmap_box.addItem("")
        self.cmap_box.addItem("")
        self.cmap_box.addItem("")
        self.cmap_box.addItem("")
        self.cmap_box.addItem("")
        self.cmap_box.addItem("")
        self.cmap_box.addItem("")
        self.cmap_box.addItem("")
        self.cmap_box.addItem("")
        self.cmap_box.addItem("")
        self.cmap_box.addItem("")
        self.cmap_box.addItem("")
        self.cmap_box.addItem("")
        self.cmap_box.addItem("")
        self.cmap_box.addItem("")
        self.cmap_box.addItem("")
        self.cmap_box.addItem("")
        self.cmap_box.addItem("")
        self.cmap_box.addItem("")
        self.cmap_box.setObjectName(u"cmap_box")

        self.gridLayout_1.addWidget(self.cmap_box, 0, 3, 1, 1)

        self.label_22 = QLabel(self.layoutWidget1)
        self.label_22.setObjectName(u"label_22")

        self.gridLayout_1.addWidget(self.label_22, 1, 2, 1, 1)

        self.precision_box = QSpinBox(self.layoutWidget1)
        self.precision_box.setObjectName(u"precision_box")
        self.precision_box.setMinimum(1)
        self.precision_box.setMaximum(7)
        self.precision_box.setValue(4)

        self.gridLayout_1.addWidget(self.precision_box, 1, 3, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout_1)

        self.structTreeView = StructTreeView(self.layoutWidget1)
        self.structTreeView.setObjectName(u"structTreeView")

        self.verticalLayout.addWidget(self.structTreeView)

        self.noteLabel = QLabel(self.layoutWidget1)
        self.noteLabel.setObjectName(u"noteLabel")
        self.noteLabel.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)
        self.noteLabel.setWordWrap(True)

        self.verticalLayout.addWidget(self.noteLabel)

        self.splitter.addWidget(self.layoutWidget1)

        self.verticalLayout_4.addWidget(self.splitter)

        self.tabWidget.addTab(self.heat_map_tab, "")
        self.if_divide_tab = QWidget()
        self.if_divide_tab.setObjectName(u"if_divide_tab")
        self.verticalLayout_8 = QVBoxLayout(self.if_divide_tab)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_5 = QLabel(self.if_divide_tab)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 1, 4, 1, 1)

        self.label_3 = QLabel(self.if_divide_tab)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 1, 6, 1, 1)

        self.label = QLabel(self.if_divide_tab)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)

        self.right_lo1_box = QDoubleSpinBox(self.if_divide_tab)
        self.right_lo1_box.setObjectName(u"right_lo1_box")
        self.right_lo1_box.setEnabled(False)
        self.right_lo1_box.setDecimals(3)
        self.right_lo1_box.setMinimum(5900.000000000000000)
        self.right_lo1_box.setMaximum(5900.000000000000000)
        self.right_lo1_box.setSingleStep(0.001000000000000)
        self.right_lo1_box.setStepType(QAbstractSpinBox.DefaultStepType)
        self.right_lo1_box.setValue(5900.000000000000000)

        self.gridLayout.addWidget(self.right_lo1_box, 1, 5, 1, 1)

        self.divide_button = QPushButton(self.if_divide_tab)
        self.divide_button.setObjectName(u"divide_button")

        self.gridLayout.addWidget(self.divide_button, 0, 0, 1, 2)

        self.label_4 = QLabel(self.if_divide_tab)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 0, 4, 1, 1)

        self.label_8 = QLabel(self.if_divide_tab)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout.addWidget(self.label_8, 2, 6, 1, 1)

        self.label_2 = QLabel(self.if_divide_tab)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)

        self.group_combo = QComboBox(self.if_divide_tab)
        self.group_combo.setObjectName(u"group_combo")

        self.gridLayout.addWidget(self.group_combo, 2, 1, 1, 1)

        self.expect_if_box = QDoubleSpinBox(self.if_divide_tab)
        self.expect_if_box.setObjectName(u"expect_if_box")
        self.expect_if_box.setEnabled(False)
        self.expect_if_box.setDecimals(3)
        self.expect_if_box.setMinimum(800.000000000000000)
        self.expect_if_box.setMaximum(1300.000000000000000)
        self.expect_if_box.setSingleStep(0.001000000000000)
        self.expect_if_box.setStepType(QAbstractSpinBox.DefaultStepType)
        self.expect_if_box.setValue(1050.000000000000000)

        self.gridLayout.addWidget(self.expect_if_box, 2, 5, 1, 1)

        self.accuracy_box = QSpinBox(self.if_divide_tab)
        self.accuracy_box.setObjectName(u"accuracy_box")
        self.accuracy_box.setMaximum(5)
        self.accuracy_box.setValue(3)

        self.gridLayout.addWidget(self.accuracy_box, 1, 3, 1, 1)

        self.label_10 = QLabel(self.if_divide_tab)
        self.label_10.setObjectName(u"label_10")

        self.gridLayout.addWidget(self.label_10, 0, 6, 1, 1)

        self.label_6 = QLabel(self.if_divide_tab)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setEnabled(True)

        self.gridLayout.addWidget(self.label_6, 2, 4, 1, 1)

        self.max_freq_box = QDoubleSpinBox(self.if_divide_tab)
        self.max_freq_box.setObjectName(u"max_freq_box")
        self.max_freq_box.setEnabled(False)
        self.max_freq_box.setDecimals(3)
        self.max_freq_box.setMinimum(8000.000000000000000)
        self.max_freq_box.setMaximum(8000.000000000000000)
        self.max_freq_box.setSingleStep(0.001000000000000)
        self.max_freq_box.setStepType(QAbstractSpinBox.DefaultStepType)
        self.max_freq_box.setValue(8000.000000000000000)

        self.gridLayout.addWidget(self.max_freq_box, 2, 7, 1, 1)

        self.min_freq_box = QDoubleSpinBox(self.if_divide_tab)
        self.min_freq_box.setObjectName(u"min_freq_box")
        self.min_freq_box.setEnabled(False)
        self.min_freq_box.setDecimals(3)
        self.min_freq_box.setMinimum(4000.000000000000000)
        self.min_freq_box.setMaximum(4000.000000000000000)
        self.min_freq_box.setSingleStep(0.001000000000000)
        self.min_freq_box.setStepType(QAbstractSpinBox.DefaultStepType)
        self.min_freq_box.setValue(4000.000000000000000)

        self.gridLayout.addWidget(self.min_freq_box, 0, 7, 1, 1)

        self.max_gap_box = QDoubleSpinBox(self.if_divide_tab)
        self.max_gap_box.setObjectName(u"max_gap_box")
        self.max_gap_box.setEnabled(False)
        self.max_gap_box.setDecimals(3)
        self.max_gap_box.setMinimum(400.000000000000000)
        self.max_gap_box.setMaximum(500.000000000000000)
        self.max_gap_box.setSingleStep(0.001000000000000)
        self.max_gap_box.setStepType(QAbstractSpinBox.DefaultStepType)
        self.max_gap_box.setValue(500.000000000000000)

        self.gridLayout.addWidget(self.max_gap_box, 0, 3, 1, 1)

        self.label_7 = QLabel(self.if_divide_tab)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout.addWidget(self.label_7, 1, 2, 1, 1)

        self.left_lo1_box = QDoubleSpinBox(self.if_divide_tab)
        self.left_lo1_box.setObjectName(u"left_lo1_box")
        self.left_lo1_box.setEnabled(False)
        self.left_lo1_box.setDecimals(3)
        self.left_lo1_box.setMinimum(8100.000000000000000)
        self.left_lo1_box.setMaximum(8100.000000000000000)
        self.left_lo1_box.setSingleStep(0.001000000000000)
        self.left_lo1_box.setStepType(QAbstractSpinBox.DefaultStepType)
        self.left_lo1_box.setValue(8100.000000000000000)

        self.gridLayout.addWidget(self.left_lo1_box, 0, 5, 1, 1)

        self.save_button = QPushButton(self.if_divide_tab)
        self.save_button.setObjectName(u"save_button")

        self.gridLayout.addWidget(self.save_button, 1, 0, 1, 2)

        self.mid_freq_box = QDoubleSpinBox(self.if_divide_tab)
        self.mid_freq_box.setObjectName(u"mid_freq_box")
        self.mid_freq_box.setEnabled(False)
        self.mid_freq_box.setDecimals(3)
        self.mid_freq_box.setMinimum(4000.000000000000000)
        self.mid_freq_box.setMaximum(8000.000000000000000)
        self.mid_freq_box.setSingleStep(0.001000000000000)
        self.mid_freq_box.setStepType(QAbstractSpinBox.DefaultStepType)
        self.mid_freq_box.setValue(6000.000000000000000)

        self.gridLayout.addWidget(self.mid_freq_box, 1, 7, 1, 1)

        self.label_9 = QLabel(self.if_divide_tab)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout.addWidget(self.label_9, 2, 2, 1, 1)

        self.goal_gap_edit = QLineEdit(self.if_divide_tab)
        self.goal_gap_edit.setObjectName(u"goal_gap_edit")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.goal_gap_edit.sizePolicy().hasHeightForWidth())
        self.goal_gap_edit.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.goal_gap_edit, 2, 3, 1, 1)


        self.verticalLayout_8.addLayout(self.gridLayout)

        self.groupBox_2 = QGroupBox(self.if_divide_tab)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.verticalLayout_9 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.info_tree_view = QTableView(self.groupBox_2)
        self.info_tree_view.setObjectName(u"info_tree_view")
        self.info_tree_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.info_tree_view.setTextElideMode(Qt.ElideNone)
        self.info_tree_view.horizontalHeader().setStretchLastSection(False)

        self.verticalLayout_9.addWidget(self.info_tree_view)


        self.verticalLayout_8.addWidget(self.groupBox_2)

        self.verticalLayout_8.setStretch(0, 1)
        self.verticalLayout_8.setStretch(1, 3)
        self.tabWidget.addTab(self.if_divide_tab, "")
        self.amp_divide_tab = QWidget()
        self.amp_divide_tab.setObjectName(u"amp_divide_tab")
        self.verticalLayout_5 = QVBoxLayout(self.amp_divide_tab)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.label_12 = QLabel(self.amp_divide_tab)
        self.label_12.setObjectName(u"label_12")

        self.gridLayout_2.addWidget(self.label_12, 0, 2, 1, 1)

        self.divide_button_amp = QPushButton(self.amp_divide_tab)
        self.divide_button_amp.setObjectName(u"divide_button_amp")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.divide_button_amp.sizePolicy().hasHeightForWidth())
        self.divide_button_amp.setSizePolicy(sizePolicy1)

        self.gridLayout_2.addWidget(self.divide_button_amp, 0, 0, 1, 2)

        self.save_button_amp = QPushButton(self.amp_divide_tab)
        self.save_button_amp.setObjectName(u"save_button_amp")
        sizePolicy1.setHeightForWidth(self.save_button_amp.sizePolicy().hasHeightForWidth())
        self.save_button_amp.setSizePolicy(sizePolicy1)

        self.gridLayout_2.addWidget(self.save_button_amp, 1, 0, 1, 2)

        self.horizontalSpacer = QSpacerItem(500, 20, QSizePolicy.Maximum, QSizePolicy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer, 0, 5, 1, 1)

        self.std_amp_edit = QLineEdit(self.amp_divide_tab)
        self.std_amp_edit.setObjectName(u"std_amp_edit")
        sizePolicy2 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.std_amp_edit.sizePolicy().hasHeightForWidth())
        self.std_amp_edit.setSizePolicy(sizePolicy2)
        self.std_amp_edit.setCursorMoveStyle(Qt.LogicalMoveStyle)
        self.std_amp_edit.setClearButtonEnabled(False)

        self.gridLayout_2.addWidget(self.std_amp_edit, 0, 3, 1, 1)

        self.label_13 = QLabel(self.amp_divide_tab)
        self.label_13.setObjectName(u"label_13")

        self.gridLayout_2.addWidget(self.label_13, 1, 2, 1, 1)

        self.bus_combo = QComboBox(self.amp_divide_tab)
        self.bus_combo.setObjectName(u"bus_combo")

        self.gridLayout_2.addWidget(self.bus_combo, 1, 3, 1, 1)

        self.syncUnionCheckBox = QCheckBox(self.amp_divide_tab)
        self.syncUnionCheckBox.setObjectName(u"syncUnionCheckBox")
        self.syncUnionCheckBox.setLayoutDirection(Qt.RightToLeft)
        self.syncUnionCheckBox.setChecked(False)

        self.gridLayout_2.addWidget(self.syncUnionCheckBox, 0, 4, 1, 1)


        self.verticalLayout_5.addLayout(self.gridLayout_2)

        self.verticalSpacer = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.verticalLayout_5.addItem(self.verticalSpacer)

        self.groupBox_3 = QGroupBox(self.amp_divide_tab)
        self.groupBox_3.setObjectName(u"groupBox_3")
        sizePolicy3 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.groupBox_3.sizePolicy().hasHeightForWidth())
        self.groupBox_3.setSizePolicy(sizePolicy3)
        self.verticalLayout_10 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.verticalLayout_10.setContentsMargins(9, 9, 9, 9)
        self.info_table_view = QTableView(self.groupBox_3)
        self.info_table_view.setObjectName(u"info_table_view")
        self.info_table_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.info_table_view.setTextElideMode(Qt.ElideNone)
        self.info_table_view.horizontalHeader().setStretchLastSection(False)

        self.verticalLayout_10.addWidget(self.info_table_view)


        self.verticalLayout_5.addWidget(self.groupBox_3)

        self.tabWidget.addTab(self.amp_divide_tab, "")
        self.parallel_divide_tab = QWidget()
        self.parallel_divide_tab.setObjectName(u"parallel_divide_tab")
        self.verticalLayout_7 = QVBoxLayout(self.parallel_divide_tab)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.parallel_group = QComboBox(self.parallel_divide_tab)
        self.parallel_group.setObjectName(u"parallel_group")
        sizePolicy1.setHeightForWidth(self.parallel_group.sizePolicy().hasHeightForWidth())
        self.parallel_group.setSizePolicy(sizePolicy1)
        self.parallel_group.setMinimumSize(QSize(150, 0))

        self.gridLayout_3.addWidget(self.parallel_group, 0, 2, 1, 1)

        self.label_20 = QLabel(self.parallel_divide_tab)
        self.label_20.setObjectName(u"label_20")

        self.gridLayout_3.addWidget(self.label_20, 0, 3, 1, 1)

        self.label_17 = QLabel(self.parallel_divide_tab)
        self.label_17.setObjectName(u"label_17")

        self.gridLayout_3.addWidget(self.label_17, 1, 1, 1, 1)

        self.context_group = QComboBox(self.parallel_divide_tab)
        self.context_group.setObjectName(u"context_group")

        self.gridLayout_3.addWidget(self.context_group, 1, 2, 1, 3)

        self.parallel_divide_button = QPushButton(self.parallel_divide_tab)
        self.parallel_divide_button.setObjectName(u"parallel_divide_button")
        sizePolicy1.setHeightForWidth(self.parallel_divide_button.sizePolicy().hasHeightForWidth())
        self.parallel_divide_button.setSizePolicy(sizePolicy1)

        self.gridLayout_3.addWidget(self.parallel_divide_button, 0, 0, 1, 1)

        self.label_16 = QLabel(self.parallel_divide_tab)
        self.label_16.setObjectName(u"label_16")

        self.gridLayout_3.addWidget(self.label_16, 0, 1, 1, 1)

        self.mode_group = QComboBox(self.parallel_divide_tab)
        self.mode_group.setObjectName(u"mode_group")
        self.mode_group.setMinimumSize(QSize(150, 0))

        self.gridLayout_3.addWidget(self.mode_group, 0, 4, 1, 1)

        self.parallel_set_button = QPushButton(self.parallel_divide_tab)
        self.parallel_set_button.setObjectName(u"parallel_set_button")
        sizePolicy1.setHeightForWidth(self.parallel_set_button.sizePolicy().hasHeightForWidth())
        self.parallel_set_button.setSizePolicy(sizePolicy1)
        self.parallel_set_button.setMinimumSize(QSize(80, 0))

        self.gridLayout_3.addWidget(self.parallel_set_button, 1, 0, 1, 1)

        self.physical_unit_edit = QLineEdit(self.parallel_divide_tab)
        self.physical_unit_edit.setObjectName(u"physical_unit_edit")

        self.gridLayout_3.addWidget(self.physical_unit_edit, 1, 5, 1, 1)

        self.label_14 = QLabel(self.parallel_divide_tab)
        self.label_14.setObjectName(u"label_14")

        self.gridLayout_3.addWidget(self.label_14, 0, 5, 1, 1)


        self.verticalLayout_7.addLayout(self.gridLayout_3)

        self.splitter_2 = QSplitter(self.parallel_divide_tab)
        self.splitter_2.setObjectName(u"splitter_2")
        self.splitter_2.setOrientation(Qt.Horizontal)
        self.groupBox = QGroupBox(self.splitter_2)
        self.groupBox.setObjectName(u"groupBox")
        self.horizontalLayout_2 = QHBoxLayout(self.groupBox)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.parallel_table_view = QTableView(self.groupBox)
        self.parallel_table_view.setObjectName(u"parallel_table_view")

        self.horizontalLayout_2.addWidget(self.parallel_table_view)

        self.splitter_2.addWidget(self.groupBox)
        self.groupBox_4 = QGroupBox(self.splitter_2)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.verticalLayout_6 = QVBoxLayout(self.groupBox_4)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.parallel_tree_view = DivideTreeView(self.groupBox_4)
        self.parallel_tree_view.setObjectName(u"parallel_tree_view")

        self.verticalLayout_6.addWidget(self.parallel_tree_view)

        self.splitter_2.addWidget(self.groupBox_4)

        self.verticalLayout_7.addWidget(self.splitter_2)

        self.verticalLayout_7.setStretch(0, 1)
        self.verticalLayout_7.setStretch(1, 8)
        self.tabWidget.addTab(self.parallel_divide_tab, "")

        self.verticalLayout_3.addWidget(self.tabWidget)

        MainWindow.setCentralWidget(self.centralwidget)
        self.toolBar = QToolBar(MainWindow)
        self.toolBar.setObjectName(u"toolBar")
        self.toolBar.setIconSize(QSize(29, 29))
        self.toolBar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        MainWindow.addToolBar(Qt.TopToolBarArea, self.toolBar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.toolBar.addAction(self.actionQuery)
        self.toolBar.addAction(self.actionRefresh)
        self.toolBar.addAction(self.actionLoadFromLocal)
        self.toolBar.addAction(self.actionSaveLocal)
        self.toolBar.addAction(self.actionSavetoDB)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionSaveHeatmap)

        self.retranslateUi(MainWindow)
        self.actionQuery.triggered.connect(MainWindow.tool_query)
        self.actionRefresh.triggered.connect(MainWindow.tool_refresh)
        self.actionSavetoDB.triggered.connect(MainWindow.tool_save_db)
        self.actionSaveLocal.triggered.connect(MainWindow.tool_save_local)
        self.actionLoadFromLocal.triggered.connect(MainWindow.tool_import)
        self.actionSaveHeatmap.triggered.connect(MainWindow.save_picture)
        self.parallel_divide_button.clicked.connect(MainWindow.parallel_divide)
        self.parallel_group.currentTextChanged.connect(MainWindow.change_parallel_group)
        self.group_combo.currentTextChanged.connect(MainWindow.change_group)
        self.parallel_set_button.clicked.connect(MainWindow.parallel_set)
        self.save_button_amp.clicked.connect(MainWindow.save_readout_param)
        self.bus_combo.currentTextChanged.connect(MainWindow.change_bus_group)
        self.show_value_box.stateChanged.connect(MainWindow.show_value_change)
        self.divide_button.clicked.connect(MainWindow.divide_baseband_freq)
        self.component_combox.currentTextChanged.connect(MainWindow.change_component)
        self.save_button.clicked.connect(MainWindow.save_baseband_freq)
        self.divide_button_amp.clicked.connect(MainWindow.divide_readout_amp)
        self.cmap_box.currentTextChanged.connect(MainWindow.change_cmap)
        self.context_group.currentTextChanged.connect(MainWindow.change_parallel_context)
        self.precision_box.valueChanged.connect(MainWindow.precision_change)

        self.tabWidget.setCurrentIndex(3)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Quantum Chip Heatmap", None))
        self.actionSavetoDB.setText(QCoreApplication.translate("MainWindow", u"Save", None))
#if QT_CONFIG(tooltip)
        self.actionSavetoDB.setToolTip(QCoreApplication.translate("MainWindow", u"Save changed values to database", None))
#endif // QT_CONFIG(tooltip)
        self.actionQuery.setText(QCoreApplication.translate("MainWindow", u"Query", None))
#if QT_CONFIG(tooltip)
        self.actionQuery.setToolTip(QCoreApplication.translate("MainWindow", u"Query user work points", None))
#endif // QT_CONFIG(tooltip)
        self.actionLoadFromLocal.setText(QCoreApplication.translate("MainWindow", u"Import", None))
#if QT_CONFIG(tooltip)
        self.actionLoadFromLocal.setToolTip(QCoreApplication.translate("MainWindow", u"Loading parameters from local file", None))
#endif // QT_CONFIG(tooltip)
        self.actionSaveLocal.setText(QCoreApplication.translate("MainWindow", u"Export", None))
#if QT_CONFIG(tooltip)
        self.actionSaveLocal.setToolTip(QCoreApplication.translate("MainWindow", u"Save parameters to local file.", None))
#endif // QT_CONFIG(tooltip)
        self.actionRefresh.setText(QCoreApplication.translate("MainWindow", u"Refresh", None))
#if QT_CONFIG(tooltip)
        self.actionRefresh.setToolTip(QCoreApplication.translate("MainWindow", u"Refresh data and loading from database", None))
#endif // QT_CONFIG(tooltip)
        self.actionSaveHeatmap.setText(QCoreApplication.translate("MainWindow", u"Save Heatmap", None))
#if QT_CONFIG(tooltip)
        self.actionSaveHeatmap.setToolTip(QCoreApplication.translate("MainWindow", u"save current heatmap picture", None))
#endif // QT_CONFIG(tooltip)
        self.label_21.setText(QCoreApplication.translate("MainWindow", u"Component", None))
        self.component_combox.setItemText(0, QCoreApplication.translate("MainWindow", u"Qubit", None))
        self.component_combox.setItemText(1, QCoreApplication.translate("MainWindow", u"Coupler", None))
        self.component_combox.setItemText(2, QCoreApplication.translate("MainWindow", u"QubitPair", None))

        self.label_11.setText(QCoreApplication.translate("MainWindow", u"CMap", None))
        self.show_value_box.setText(QCoreApplication.translate("MainWindow", u"Show Value", None))
#if QT_CONFIG(shortcut)
        self.show_value_box.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Space", None))
#endif // QT_CONFIG(shortcut)
        self.cmap_box.setItemText(0, QCoreApplication.translate("MainWindow", u"viridis", None))
        self.cmap_box.setItemText(1, QCoreApplication.translate("MainWindow", u"inferno", None))
        self.cmap_box.setItemText(2, QCoreApplication.translate("MainWindow", u"magma", None))
        self.cmap_box.setItemText(3, QCoreApplication.translate("MainWindow", u"plasma", None))
        self.cmap_box.setItemText(4, QCoreApplication.translate("MainWindow", u"cividis", None))
        self.cmap_box.setItemText(5, QCoreApplication.translate("MainWindow", u"Greys", None))
        self.cmap_box.setItemText(6, QCoreApplication.translate("MainWindow", u"Purples", None))
        self.cmap_box.setItemText(7, QCoreApplication.translate("MainWindow", u"Blues", None))
        self.cmap_box.setItemText(8, QCoreApplication.translate("MainWindow", u"Greens", None))
        self.cmap_box.setItemText(9, QCoreApplication.translate("MainWindow", u"Oranges", None))
        self.cmap_box.setItemText(10, QCoreApplication.translate("MainWindow", u"Reds", None))
        self.cmap_box.setItemText(11, QCoreApplication.translate("MainWindow", u"YlOrBr", None))
        self.cmap_box.setItemText(12, QCoreApplication.translate("MainWindow", u"YlOrRd", None))
        self.cmap_box.setItemText(13, QCoreApplication.translate("MainWindow", u"OrRd", None))
        self.cmap_box.setItemText(14, QCoreApplication.translate("MainWindow", u"PuRd", None))
        self.cmap_box.setItemText(15, QCoreApplication.translate("MainWindow", u"YlGn", None))
        self.cmap_box.setItemText(16, QCoreApplication.translate("MainWindow", u"PuBu", None))
        self.cmap_box.setItemText(17, QCoreApplication.translate("MainWindow", u"BuPu", None))
        self.cmap_box.setItemText(18, QCoreApplication.translate("MainWindow", u"GnBu", None))

        self.label_22.setText(QCoreApplication.translate("MainWindow", u"Significant digit", None))
        self.noteLabel.setText(QCoreApplication.translate("MainWindow", u"Note: Users can edit the relevant parameters of Qubit or Coupler through the heatmap, and after clicking the save button, load them into the user's data cache, which will be automatically loaded during the experiment.", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.heat_map_tab), QCoreApplication.translate("MainWindow", u"HeatMap", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"RIGHT_LO1", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"MID_FRQ", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Group", None))
        self.divide_button.setText(QCoreApplication.translate("MainWindow", u"Divide", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"LEFT_LO1", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"MAX FRQ", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"MAX GAP", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"MIN FRQ", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"EXPECT_IF", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"ACCURACY", None))
        self.save_button.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Goal Gap", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Divide Information", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.if_divide_tab), QCoreApplication.translate("MainWindow", u"IF Divide", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Standard Amp", None))
        self.divide_button_amp.setText(QCoreApplication.translate("MainWindow", u"Divide", None))
        self.save_button_amp.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.std_amp_edit.setText(QCoreApplication.translate("MainWindow", u"0.15", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"BUS", None))
        self.syncUnionCheckBox.setText(QCoreApplication.translate("MainWindow", u"Sync to union readout", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Divide Information", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.amp_divide_tab), QCoreApplication.translate("MainWindow", u"AMP Divide", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u"mode", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"   Context", None))
        self.parallel_divide_button.setText(QCoreApplication.translate("MainWindow", u"Divide", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"    Group", None))
        self.parallel_set_button.setText(QCoreApplication.translate("MainWindow", u"Set", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"Input Parallel Physical Units", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Divide Information", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("MainWindow", u"Divide Config", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.parallel_divide_tab), QCoreApplication.translate("MainWindow", u"Parallel Divide", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar", None))
    # retranslateUi

