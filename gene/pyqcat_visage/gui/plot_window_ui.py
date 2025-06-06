# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'plot_window_ui.ui'
##
## Created by: Qt User Interface Compiler version 6.3.2
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
from PySide6.QtWidgets import (QApplication, QMainWindow, QMenuBar, QSizePolicy,
    QStatusBar, QVBoxLayout, QWidget)

from pyqcat_visage.gui.widgets.bases.expanding_toolbar import QToolBarExpanding
import _imgs_rc

class Ui_MainWindowPlot(object):
    def setupUi(self, MainWindowPlot):
        if not MainWindowPlot.objectName():
            MainWindowPlot.setObjectName(u"MainWindowPlot")
        MainWindowPlot.resize(800, 600)
        MainWindowPlot.setIconSize(QSize(24, 24))
        self.actionPan = QAction(MainWindowPlot)
        self.actionPan.setObjectName(u"actionPan")
        icon = QIcon()
        icon.addFile(u":/plot/pan", QSize(), QIcon.Normal, QIcon.On)
        self.actionPan.setIcon(icon)
        self.actionZoom = QAction(MainWindowPlot)
        self.actionZoom.setObjectName(u"actionZoom")
        icon1 = QIcon()
        icon1.addFile(u":/plot/zoom", QSize(), QIcon.Normal, QIcon.Off)
        self.actionZoom.setIcon(icon1)
        self.actionConnectors = QAction(MainWindowPlot)
        self.actionConnectors.setObjectName(u"actionConnectors")
        self.actionConnectors.setCheckable(True)
        self.actionConnectors.setChecked(False)
        icon2 = QIcon()
        icon2.addFile(u":/connectors", QSize(), QIcon.Normal, QIcon.Off)
        self.actionConnectors.setIcon(icon2)
        self.actionCoords = QAction(MainWindowPlot)
        self.actionCoords.setObjectName(u"actionCoords")
        self.actionCoords.setCheckable(True)
        self.actionCoords.setChecked(True)
        icon3 = QIcon()
        icon3.addFile(u":/plot/point", QSize(), QIcon.Normal, QIcon.Off)
        self.actionCoords.setIcon(icon3)
        self.actionAuto = QAction(MainWindowPlot)
        self.actionAuto.setObjectName(u"actionAuto")
        icon4 = QIcon()
        icon4.addFile(u":/plot/autozoom", QSize(), QIcon.Normal, QIcon.Off)
        self.actionAuto.setIcon(icon4)
        self.actionReplot = QAction(MainWindowPlot)
        self.actionReplot.setObjectName(u"actionReplot")
        icon5 = QIcon()
        icon5.addFile(u":/plot/refresh_plot", QSize(), QIcon.Normal, QIcon.Off)
        self.actionReplot.setIcon(icon5)
        self.actionRuler = QAction(MainWindowPlot)
        self.actionRuler.setObjectName(u"actionRuler")
        icon6 = QIcon()
        icon6.addFile(u":/plot/ruler", QSize(), QIcon.Normal, QIcon.Off)
        self.actionRuler.setIcon(icon6)
        self.centralwidget = QWidget(MainWindowPlot)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        MainWindowPlot.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindowPlot)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 22))
        MainWindowPlot.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindowPlot)
        self.statusbar.setObjectName(u"statusbar")
        self.statusbar.setEnabled(True)
        MainWindowPlot.setStatusBar(self.statusbar)
        self.toolBar = QToolBarExpanding(MainWindowPlot)
        self.toolBar.setObjectName(u"toolBar")
        self.toolBar.setIconSize(QSize(20, 20))
        self.toolBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        MainWindowPlot.addToolBar(Qt.TopToolBarArea, self.toolBar)

        self.toolBar.addAction(self.actionPan)
        self.toolBar.addAction(self.actionAuto)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionCoords)
        self.toolBar.addAction(self.actionConnectors)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionRuler)

        self.retranslateUi(MainWindowPlot)
        self.actionAuto.triggered.connect(MainWindowPlot.auto_scale)
        self.actionConnectors.triggered["bool"].connect(MainWindowPlot.set_show_pins)
        self.actionCoords.triggered["bool"].connect(MainWindowPlot.set_position_track)
        self.actionPan.triggered.connect(MainWindowPlot.pan)
        self.actionZoom.triggered.connect(MainWindowPlot.zoom)
        self.actionReplot.triggered.connect(MainWindowPlot.replot)

        QMetaObject.connectSlotsByName(MainWindowPlot)
    # setupUi

    def retranslateUi(self, MainWindowPlot):
        MainWindowPlot.setWindowTitle(QCoreApplication.translate("MainWindowPlot", u"MainWindow", None))
        self.actionPan.setText(QCoreApplication.translate("MainWindowPlot", u"Help", None))
#if QT_CONFIG(shortcut)
        self.actionPan.setShortcut(QCoreApplication.translate("MainWindowPlot", u"P", None))
#endif // QT_CONFIG(shortcut)
        self.actionZoom.setText(QCoreApplication.translate("MainWindowPlot", u"Zoom", None))
#if QT_CONFIG(tooltip)
        self.actionZoom.setToolTip(QCoreApplication.translate("MainWindowPlot", u"Zoom control", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.actionZoom.setShortcut(QCoreApplication.translate("MainWindowPlot", u"Z", None))
#endif // QT_CONFIG(shortcut)
        self.actionConnectors.setText(QCoreApplication.translate("MainWindowPlot", u"Pins", None))
#if QT_CONFIG(tooltip)
        self.actionConnectors.setToolTip(QCoreApplication.translate("MainWindowPlot", u"Show connector pins for selected qcomponents", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.actionConnectors.setShortcut(QCoreApplication.translate("MainWindowPlot", u"C", None))
#endif // QT_CONFIG(shortcut)
        self.actionCoords.setText(QCoreApplication.translate("MainWindowPlot", u"Get point", None))
#if QT_CONFIG(tooltip)
        self.actionCoords.setToolTip(QCoreApplication.translate("MainWindowPlot", u"Click for position --- Enable this to click on the plot and log the (x,y) position", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.actionCoords.setShortcut(QCoreApplication.translate("MainWindowPlot", u"P", None))
#endif // QT_CONFIG(shortcut)
        self.actionAuto.setText(QCoreApplication.translate("MainWindowPlot", u"Autoscale", None))
#if QT_CONFIG(tooltip)
        self.actionAuto.setToolTip(QCoreApplication.translate("MainWindowPlot", u"Auto Zoom", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.actionAuto.setShortcut(QCoreApplication.translate("MainWindowPlot", u"A", None))
#endif // QT_CONFIG(shortcut)
        self.actionReplot.setText(QCoreApplication.translate("MainWindowPlot", u"Replot", None))
#if QT_CONFIG(shortcut)
        self.actionReplot.setShortcut(QCoreApplication.translate("MainWindowPlot", u"Ctrl+R", None))
#endif // QT_CONFIG(shortcut)
        self.actionRuler.setText(QCoreApplication.translate("MainWindowPlot", u"Ruler", None))
#if QT_CONFIG(tooltip)
        self.actionRuler.setToolTip(QCoreApplication.translate("MainWindowPlot", u"Activate the ruler", None))
#endif // QT_CONFIG(tooltip)
        self.toolBar.setWindowTitle(QCoreApplication.translate("MainWindowPlot", u"toolBar", None))
    # retranslateUi

