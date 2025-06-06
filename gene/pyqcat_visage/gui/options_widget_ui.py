# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'options_widget_ui.ui'
##
## Created by: Qt User Interface Compiler version 6.3.2
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
from PySide6.QtWidgets import (QApplication, QHeaderView, QSizePolicy, QTabWidget,
    QVBoxLayout, QWidget)

from pyqcat_visage.gui.widgets.options.tree_view_options import QTreeViewOptionsWidget

class Ui_TabWidget(object):
    def setupUi(self, TabWidget):
        if not TabWidget.objectName():
            TabWidget.setObjectName(u"TabWidget")
        TabWidget.resize(627, 551)
        TabWidget.setTabPosition(QTabWidget.West)
        TabWidget.setTabShape(QTabWidget.Triangular)
        self.tab_exp_options = QWidget()
        self.tab_exp_options.setObjectName(u"tab_exp_options")
        self.verticalLayout_2 = QVBoxLayout(self.tab_exp_options)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(1, 1, 1, 1)
        self.exp_tree_view = QTreeViewOptionsWidget(self.tab_exp_options)
        self.exp_tree_view.setObjectName(u"exp_tree_view")

        self.verticalLayout_2.addWidget(self.exp_tree_view)

        TabWidget.addTab(self.tab_exp_options, "")
        self.tab_ana_options = QWidget()
        self.tab_ana_options.setObjectName(u"tab_ana_options")
        self.verticalLayout = QVBoxLayout(self.tab_ana_options)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(1, 1, 1, 1)
        self.ana_tree_view = QTreeViewOptionsWidget(self.tab_ana_options)
        self.ana_tree_view.setObjectName(u"ana_tree_view")

        self.verticalLayout.addWidget(self.ana_tree_view)

        TabWidget.addTab(self.tab_ana_options, "")

        self.retranslateUi(TabWidget)

        TabWidget.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(TabWidget)
    # setupUi

    def retranslateUi(self, TabWidget):
        TabWidget.setWindowTitle(QCoreApplication.translate("TabWidget", u"TabWidget", None))
        TabWidget.setTabText(TabWidget.indexOf(self.tab_exp_options), QCoreApplication.translate("TabWidget", u"Experiment", None))
        TabWidget.setTabText(TabWidget.indexOf(self.tab_ana_options), QCoreApplication.translate("TabWidget", u"Analysis", None))
    # retranslateUi

