# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'task_manage_ui.ui'
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
from PySide6.QtWidgets import (QApplication, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow,
    QSizePolicy, QStatusBar, QToolBar, QVBoxLayout,
    QWidget)

from pyqcat_visage.gui.widgets.task.table_view_task import QTableViewTaskWidget
from pyqcat_visage.gui.widgets.task.tree_view_task_info import QTreeViewTaskInfoWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(910, 682)
        self.actionQueryTask = QAction(MainWindow)
        self.actionQueryTask.setObjectName(u"actionQueryTask")
        self.actionRefresh = QAction(MainWindow)
        self.actionRefresh.setObjectName(u"actionRefresh")
        self.actionQueryHistory = QAction(MainWindow)
        self.actionQueryHistory.setObjectName(u"actionQueryHistory")
        self.actionQuery = QAction(MainWindow)
        self.actionQuery.setObjectName(u"actionQuery")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout_2 = QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.task_widget = QWidget(self.groupBox)
        self.task_widget.setObjectName(u"task_widget")
        self.verticalLayout_2 = QVBoxLayout(self.task_widget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.task_search_widget = QWidget(self.task_widget)
        self.task_search_widget.setObjectName(u"task_search_widget")
        self.verticalLayout = QVBoxLayout(self.task_search_widget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.task_id_widget = QWidget(self.task_search_widget)
        self.task_id_widget.setObjectName(u"task_id_widget")
        self.horizontalLayout = QHBoxLayout(self.task_id_widget)
        self.horizontalLayout.setSpacing(9)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 9, 9)
        self.label = QLabel(self.task_id_widget)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.task_id = QLineEdit(self.task_id_widget)
        self.task_id.setObjectName(u"task_id")

        self.horizontalLayout.addWidget(self.task_id)

        self.horizontalLayout.setStretch(0, 2)
        self.horizontalLayout.setStretch(1, 8)

        self.verticalLayout.addWidget(self.task_id_widget)

        self.name_widget = QWidget(self.task_search_widget)
        self.name_widget.setObjectName(u"name_widget")
        self.horizontalLayout_2 = QHBoxLayout(self.name_widget)
        self.horizontalLayout_2.setSpacing(9)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, -1, -1)
        self.label_2 = QLabel(self.name_widget)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_2.addWidget(self.label_2)

        self.task_name = QLineEdit(self.name_widget)
        self.task_name.setObjectName(u"task_name")

        self.horizontalLayout_2.addWidget(self.task_name)

        self.horizontalLayout_2.setStretch(0, 2)
        self.horizontalLayout_2.setStretch(1, 8)

        self.verticalLayout.addWidget(self.name_widget)


        self.verticalLayout_2.addWidget(self.task_search_widget)

        self.tableTaskView = QTableViewTaskWidget(self.task_widget)
        self.tableTaskView.setObjectName(u"tableTaskView")

        self.verticalLayout_2.addWidget(self.tableTaskView)


        self.gridLayout_2.addWidget(self.task_widget, 0, 0, 1, 1)

        self.info_widget = QWidget(self.groupBox)
        self.info_widget.setObjectName(u"info_widget")
        self.verticalLayout_3 = QVBoxLayout(self.info_widget)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.task_info_tree_view = QTreeViewTaskInfoWidget(self.info_widget)
        self.task_info_tree_view.setObjectName(u"task_info_tree_view")

        self.verticalLayout_3.addWidget(self.task_info_tree_view)


        self.gridLayout_2.addWidget(self.info_widget, 0, 1, 1, 1)


        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QToolBar(MainWindow)
        self.toolBar.setObjectName(u"toolBar")
        MainWindow.addToolBar(Qt.TopToolBarArea, self.toolBar)

        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionQueryTask)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionQueryHistory)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionQuery)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionRefresh)

        self.retranslateUi(MainWindow)
        self.actionQueryTask.triggered.connect(MainWindow.query_task)
        self.actionRefresh.triggered.connect(MainWindow.refresh)
        self.actionQueryHistory.triggered.connect(MainWindow.query_history)
        self.actionQuery.triggered.connect(MainWindow.query)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Dag Manage", None))
        self.actionQueryTask.setText(QCoreApplication.translate("MainWindow", u"QueryAll", None))
#if QT_CONFIG(tooltip)
        self.actionQueryTask.setToolTip(QCoreApplication.translate("MainWindow", u"QueryAll", None))
#endif // QT_CONFIG(tooltip)
        self.actionRefresh.setText(QCoreApplication.translate("MainWindow", u"Refresh", None))
        self.actionQueryHistory.setText(QCoreApplication.translate("MainWindow", u"QueryHistory", None))
        self.actionQuery.setText(QCoreApplication.translate("MainWindow", u"Query", None))
#if QT_CONFIG(tooltip)
        self.actionQuery.setToolTip(QCoreApplication.translate("MainWindow", u"Query", None))
#endif // QT_CONFIG(tooltip)
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Task list", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"task_id", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"task_name", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar", None))
    # retranslateUi

