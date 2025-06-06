# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'context_sidebat_ui.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QHBoxLayout,
    QHeaderView, QLabel, QPushButton, QSizePolicy,
    QSpacerItem, QTabWidget, QVBoxLayout, QWidget)

from .table_view_point import QPointTableView
from pyqcat_visage.gui.widgets.combox_custom.combox_multi import QMultiComboBox

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(327, 269)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        icon = QIcon()
        iconThemeName = u"system-run"
        if QIcon.hasThemeIcon(iconThemeName):
            icon = QIcon.fromTheme(iconThemeName)
        else:
            icon.addFile(u".", QSize(), QIcon.Normal, QIcon.Off)
        
        Form.setWindowIcon(icon)
        self.verticalLayout = QVBoxLayout(Form)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.tabWidget = QTabWidget(Form)
        self.tabWidget.setObjectName(u"tabWidget")
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.default_tab = QWidget()
        self.default_tab.setObjectName(u"default_tab")
        self.verticalLayout_4 = QVBoxLayout(self.default_tab)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalSpacer = QSpacerItem(20, 12, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_4.addItem(self.verticalSpacer)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label_10 = QLabel(self.default_tab)
        self.label_10.setObjectName(u"label_10")

        self.horizontalLayout.addWidget(self.label_10)

        self.default_context_com = QComboBox(self.default_tab)
        self.default_context_com.addItem("")
        self.default_context_com.addItem("")
        self.default_context_com.addItem("")
        self.default_context_com.addItem("")
        self.default_context_com.addItem("")
        self.default_context_com.addItem("")
        self.default_context_com.addItem("")
        self.default_context_com.setObjectName(u"default_context_com")
        self.default_context_com.setEditable(True)

        self.horizontalLayout.addWidget(self.default_context_com)


        self.verticalLayout_4.addLayout(self.horizontalLayout)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_6 = QLabel(self.default_tab)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_3.addWidget(self.label_6)

        self.default_read_com = QComboBox(self.default_tab)
        self.default_read_com.setObjectName(u"default_read_com")
        self.default_read_com.setEnabled(True)
        self.default_read_com.setEditable(True)

        self.horizontalLayout_3.addWidget(self.default_read_com)


        self.verticalLayout_4.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_2 = QLabel(self.default_tab)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_4.addWidget(self.label_2)

        self.default_physical_unit_com = QMultiComboBox(self.default_tab)
        self.default_physical_unit_com.setObjectName(u"default_physical_unit_com")
        self.default_physical_unit_com.setEnabled(True)
        self.default_physical_unit_com.setEditable(True)

        self.horizontalLayout_4.addWidget(self.default_physical_unit_com)


        self.verticalLayout_4.addLayout(self.horizontalLayout_4)

        self.verticalSpacer_3 = QSpacerItem(20, 12, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_4.addItem(self.verticalSpacer_3)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer)

        self.btn_default_refresh = QPushButton(self.default_tab)
        self.btn_default_refresh.setObjectName(u"btn_default_refresh")

        self.horizontalLayout_5.addWidget(self.btn_default_refresh)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_4)

        self.horizontalLayout_5.setStretch(0, 4)
        self.horizontalLayout_5.setStretch(1, 2)
        self.horizontalLayout_5.setStretch(2, 1)

        self.verticalLayout_4.addLayout(self.horizontalLayout_5)

        self.verticalSpacer_2 = QSpacerItem(20, 12, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_4.addItem(self.verticalSpacer_2)

        self.verticalLayout_4.setStretch(0, 1)
        self.verticalLayout_4.setStretch(1, 2)
        self.verticalLayout_4.setStretch(2, 2)
        self.verticalLayout_4.setStretch(3, 2)
        self.verticalLayout_4.setStretch(4, 1)
        self.verticalLayout_4.setStretch(5, 2)
        self.verticalLayout_4.setStretch(6, 1)
        self.tabWidget.addTab(self.default_tab, "")
        self.exp_tab = QWidget()
        self.exp_tab.setObjectName(u"exp_tab")
        self.verticalLayout_3 = QVBoxLayout(self.exp_tab)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalSpacer_4 = QSpacerItem(20, 12, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer_4)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_9 = QLabel(self.exp_tab)
        self.label_9.setObjectName(u"label_9")

        self.horizontalLayout_6.addWidget(self.label_9)

        self.exp_context_com = QComboBox(self.exp_tab)
        self.exp_context_com.setObjectName(u"exp_context_com")
        self.exp_context_com.setEditable(True)

        self.horizontalLayout_6.addWidget(self.exp_context_com)


        self.verticalLayout_3.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_8 = QLabel(self.exp_tab)
        self.label_8.setObjectName(u"label_8")

        self.horizontalLayout_7.addWidget(self.label_8)

        self.exp_read_com = QComboBox(self.exp_tab)
        self.exp_read_com.setObjectName(u"exp_read_com")
        self.exp_read_com.setEnabled(True)
        self.exp_read_com.setEditable(True)

        self.horizontalLayout_7.addWidget(self.exp_read_com)


        self.verticalLayout_3.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.label_7 = QLabel(self.exp_tab)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_8.addWidget(self.label_7)

        self.exp_physical_unit_com = QMultiComboBox(self.exp_tab)
        self.exp_physical_unit_com.setObjectName(u"exp_physical_unit_com")
        self.exp_physical_unit_com.setEnabled(True)
        self.exp_physical_unit_com.setEditable(True)

        self.horizontalLayout_8.addWidget(self.exp_physical_unit_com)


        self.verticalLayout_3.addLayout(self.horizontalLayout_8)

        self.verticalSpacer_5 = QSpacerItem(20, 12, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer_5)

        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_16.addItem(self.horizontalSpacer_5)

        self.btn_exp_save = QPushButton(self.exp_tab)
        self.btn_exp_save.setObjectName(u"btn_exp_save")

        self.horizontalLayout_16.addWidget(self.btn_exp_save)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_16.addItem(self.horizontalSpacer_6)

        self.horizontalLayout_16.setStretch(0, 4)
        self.horizontalLayout_16.setStretch(1, 2)
        self.horizontalLayout_16.setStretch(2, 1)

        self.verticalLayout_3.addLayout(self.horizontalLayout_16)

        self.verticalSpacer_6 = QSpacerItem(20, 12, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer_6)

        self.verticalLayout_3.setStretch(0, 1)
        self.verticalLayout_3.setStretch(1, 2)
        self.verticalLayout_3.setStretch(2, 2)
        self.verticalLayout_3.setStretch(3, 2)
        self.verticalLayout_3.setStretch(4, 1)
        self.verticalLayout_3.setStretch(5, 2)
        self.verticalLayout_3.setStretch(6, 1)
        self.tabWidget.addTab(self.exp_tab, "")
        self.point_tab = QWidget()
        self.point_tab.setObjectName(u"point_tab")
        self.verticalLayout_2 = QVBoxLayout(self.point_tab)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.point_table_view = QPointTableView(self.point_tab)
        self.point_table_view.setObjectName(u"point_table_view")

        self.verticalLayout_2.addWidget(self.point_table_view)

        self.tabWidget.addTab(self.point_tab, "")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.verticalLayout_6 = QVBoxLayout(self.tab)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.label_glo_1 = QLabel(self.tab)
        self.label_glo_1.setObjectName(u"label_glo_1")

        self.horizontalLayout_9.addWidget(self.label_glo_1)

        self.com_work_type = QComboBox(self.tab)
        self.com_work_type.addItem("")
        self.com_work_type.addItem("")
        self.com_work_type.addItem("")
        self.com_work_type.setObjectName(u"com_work_type")
        self.com_work_type.setEnabled(True)
        self.com_work_type.setEditable(True)
        self.com_work_type.setFrame(True)

        self.horizontalLayout_9.addWidget(self.com_work_type)


        self.verticalLayout_6.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.label_glo_2 = QLabel(self.tab)
        self.label_glo_2.setObjectName(u"label_glo_2")

        self.horizontalLayout_10.addWidget(self.label_glo_2)

        self.com_divide_type = QComboBox(self.tab)
        self.com_divide_type.addItem("")
        self.com_divide_type.addItem("")
        self.com_divide_type.addItem("")
        self.com_divide_type.addItem("")
        self.com_divide_type.addItem("")
        self.com_divide_type.setObjectName(u"com_divide_type")
        self.com_divide_type.setEnabled(True)
        self.com_divide_type.setEditable(True)

        self.horizontalLayout_10.addWidget(self.com_divide_type)


        self.verticalLayout_6.addLayout(self.horizontalLayout_10)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.label_glo_3 = QLabel(self.tab)
        self.label_glo_3.setObjectName(u"label_glo_3")

        self.horizontalLayout_11.addWidget(self.label_glo_3)

        self.com_max_qubit = QMultiComboBox(self.tab)
        self.com_max_qubit.setObjectName(u"com_max_qubit")
        self.com_max_qubit.setEnabled(True)
        self.com_max_qubit.setEditable(True)

        self.horizontalLayout_11.addWidget(self.com_max_qubit)


        self.verticalLayout_6.addLayout(self.horizontalLayout_11)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.label_glo_4 = QLabel(self.tab)
        self.label_glo_4.setObjectName(u"label_glo_4")

        self.horizontalLayout_12.addWidget(self.label_glo_4)

        self.com_online_qubit = QMultiComboBox(self.tab)
        self.com_online_qubit.setObjectName(u"com_online_qubit")
        self.com_online_qubit.setEnabled(True)
        self.com_online_qubit.setEditable(True)

        self.horizontalLayout_12.addWidget(self.com_online_qubit)


        self.verticalLayout_6.addLayout(self.horizontalLayout_12)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.label_glo_5 = QLabel(self.tab)
        self.label_glo_5.setObjectName(u"label_glo_5")

        self.horizontalLayout_13.addWidget(self.label_glo_5)

        self.f02_opt_qubit = QMultiComboBox(self.tab)
        self.f02_opt_qubit.setObjectName(u"f02_opt_qubit")
        self.f02_opt_qubit.setEnabled(True)
        self.f02_opt_qubit.setEditable(True)

        self.horizontalLayout_13.addWidget(self.f02_opt_qubit)


        self.verticalLayout_6.addLayout(self.horizontalLayout_13)

        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.crosstalk_check = QCheckBox(self.tab)
        self.crosstalk_check.setObjectName(u"crosstalk_check")
        self.crosstalk_check.setEnabled(True)
        self.crosstalk_check.setCheckable(True)
        self.crosstalk_check.setChecked(False)
        self.crosstalk_check.setTristate(False)

        self.horizontalLayout_14.addWidget(self.crosstalk_check)

        self.online_check = QCheckBox(self.tab)
        self.online_check.setObjectName(u"online_check")
        self.online_check.setEnabled(True)
        self.online_check.setCheckable(True)
        self.online_check.setChecked(False)
        self.online_check.setTristate(False)

        self.horizontalLayout_14.addWidget(self.online_check)


        self.verticalLayout_5.addLayout(self.horizontalLayout_14)

        self.horizontalLayout_15 = QHBoxLayout()
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.xy_crosstalk_check = QCheckBox(self.tab)
        self.xy_crosstalk_check.setObjectName(u"xy_crosstalk_check")
        self.xy_crosstalk_check.setEnabled(True)
        self.xy_crosstalk_check.setCheckable(True)
        self.xy_crosstalk_check.setChecked(False)
        self.xy_crosstalk_check.setTristate(False)

        self.horizontalLayout_15.addWidget(self.xy_crosstalk_check)

        self.custom_point_check = QCheckBox(self.tab)
        self.custom_point_check.setObjectName(u"custom_point_check")
        self.custom_point_check.setEnabled(True)
        self.custom_point_check.setCheckable(True)
        self.custom_point_check.setChecked(False)
        self.custom_point_check.setTristate(False)

        self.horizontalLayout_15.addWidget(self.custom_point_check)


        self.verticalLayout_5.addLayout(self.horizontalLayout_15)


        self.horizontalLayout_17.addLayout(self.verticalLayout_5)

        self.btn_global_set = QPushButton(self.tab)
        self.btn_global_set.setObjectName(u"btn_global_set")

        self.horizontalLayout_17.addWidget(self.btn_global_set)


        self.verticalLayout_6.addLayout(self.horizontalLayout_17)

        self.tabWidget.addTab(self.tab, "")

        self.verticalLayout.addWidget(self.tabWidget)


        self.retranslateUi(Form)
        self.default_context_com.currentTextChanged.connect(Form.refresh_com)
        self.exp_context_com.currentTextChanged.connect(Form.refresh_com)
        self.btn_default_refresh.clicked.connect(Form.default_refresh)
        self.btn_exp_save.clicked.connect(Form.context_save)
        self.btn_global_set.clicked.connect(Form.update_global)

        self.tabWidget.setCurrentIndex(3)


        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.label_10.setText(QCoreApplication.translate("Form", u"context name", None))
        self.default_context_com.setItemText(0, QCoreApplication.translate("Form", u"qubit_calibration", None))
        self.default_context_com.setItemText(1, QCoreApplication.translate("Form", u"coupler_probe_calibration", None))
        self.default_context_com.setItemText(2, QCoreApplication.translate("Form", u"coupler_calibration", None))
        self.default_context_com.setItemText(3, QCoreApplication.translate("Form", u"cz_gate_calibration", None))
        self.default_context_com.setItemText(4, QCoreApplication.translate("Form", u"crosstalk_measure", None))
        self.default_context_com.setItemText(5, QCoreApplication.translate("Form", u"union_read_measure", None))
        self.default_context_com.setItemText(6, QCoreApplication.translate("Form", u"net_tunable", None))

        self.label_6.setText(QCoreApplication.translate("Form", u"readout type", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"physical unit", None))
        self.btn_default_refresh.setText(QCoreApplication.translate("Form", u"refresh", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.default_tab), QCoreApplication.translate("Form", u"default", None))
        self.label_9.setText(QCoreApplication.translate("Form", u"context name", None))
        self.label_8.setText(QCoreApplication.translate("Form", u"readout type", None))
        self.label_7.setText(QCoreApplication.translate("Form", u"physical unit", None))
        self.btn_exp_save.setText(QCoreApplication.translate("Form", u"save", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.exp_tab), QCoreApplication.translate("Form", u"experiment", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.point_tab), QCoreApplication.translate("Form", u"point", None))
        self.label_glo_1.setText(QCoreApplication.translate("Form", u"working type", None))
        self.com_work_type.setItemText(0, QCoreApplication.translate("Form", u"awg_bias", None))
        self.com_work_type.setItemText(1, QCoreApplication.translate("Form", u"ac", None))
        self.com_work_type.setItemText(2, QCoreApplication.translate("Form", u"dc", None))

        self.com_work_type.setCurrentText(QCoreApplication.translate("Form", u"awg_bias", None))
        self.label_glo_2.setText(QCoreApplication.translate("Form", u"divide type", None))
        self.com_divide_type.setItemText(0, QCoreApplication.translate("Form", u"character_idle_point", None))
        self.com_divide_type.setItemText(1, QCoreApplication.translate("Form", u"calibrate_idle_point", None))
        self.com_divide_type.setItemText(2, QCoreApplication.translate("Form", u"character_point", None))
        self.com_divide_type.setItemText(3, QCoreApplication.translate("Form", u"sweet_point", None))
        self.com_divide_type.setItemText(4, QCoreApplication.translate("Form", u"calibration_point", None))

        self.label_glo_3.setText(QCoreApplication.translate("Form", u"max point unit", None))
        self.label_glo_4.setText(QCoreApplication.translate("Form", u"online unit", None))
        self.label_glo_5.setText(QCoreApplication.translate("Form", u"02 opt unit", None))
        self.crosstalk_check.setText(QCoreApplication.translate("Form", u"crosstalk", None))
        self.online_check.setText(QCoreApplication.translate("Form", u"online", None))
        self.xy_crosstalk_check.setText(QCoreApplication.translate("Form", u"xy_crosstalk", None))
        self.custom_point_check.setText(QCoreApplication.translate("Form", u"custom_point", None))
        self.btn_global_set.setText(QCoreApplication.translate("Form", u"update", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("Form", u"Global", None))
    # retranslateUi

