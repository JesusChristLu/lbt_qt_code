# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'title_window_ui.ui'
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
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QSpacerItem, QWidget)


class Ui_TitleWindow(object):
    def setupUi(self, TitleWindow):
        if not TitleWindow.objectName():
            TitleWindow.setObjectName(u"TitleWindow")
        TitleWindow.resize(780, 347)
        TitleWindow.setStyleSheet(u"")
        self.gridLayoutTitle = QGridLayout(TitleWindow)
        self.gridLayoutTitle.setSpacing(0)
        self.gridLayoutTitle.setObjectName(u"gridLayoutTitle")
        self.gridLayoutTitle.setContentsMargins(0, 0, 0, 0)
        self.TitleWidget = QWidget(TitleWindow)
        self.TitleWidget.setObjectName(u"TitleWidget")
        self.TitleWidget.setEnabled(True)
        self.TitleWidget.setMaximumSize(QSize(16777215, 50))
        self.TitleWidget.setCursor(QCursor(Qt.ArrowCursor))
        self.TitleWidget.setStyleSheet(u"")
        self.TitleWidget.setInputMethodHints(Qt.ImhNone)
        self.horizontalLayoutTitle = QHBoxLayout(self.TitleWidget)
        self.horizontalLayoutTitle.setSpacing(5)
        self.horizontalLayoutTitle.setObjectName(u"horizontalLayoutTitle")
        self.horizontalLayoutTitle.setContentsMargins(5, 5, 5, 5)
        self.IconLabel = QLabel(self.TitleWidget)
        self.IconLabel.setObjectName(u"IconLabel")
        self.IconLabel.setMinimumSize(QSize(20, 20))
        self.IconLabel.setMaximumSize(QSize(20, 20))
        self.IconLabel.setTextFormat(Qt.AutoText)
        self.IconLabel.setPixmap(QPixmap(u":/login-logo.png"))
        self.IconLabel.setScaledContents(True)
        self.IconLabel.setWordWrap(False)
        self.IconLabel.setOpenExternalLinks(True)
        self.IconLabel.setTextInteractionFlags(Qt.NoTextInteraction)

        self.horizontalLayoutTitle.addWidget(self.IconLabel)

        self.TitleLabel = QLabel(self.TitleWidget)
        self.TitleLabel.setObjectName(u"TitleLabel")
        self.TitleLabel.setTextFormat(Qt.AutoText)

        self.horizontalLayoutTitle.addWidget(self.TitleLabel)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayoutTitle.addItem(self.horizontalSpacer_3)

        self.WindowMin = QPushButton(self.TitleWidget)
        self.WindowMin.setObjectName(u"WindowMin")
        self.WindowMin.setMinimumSize(QSize(35, 20))
        self.WindowMin.setMaximumSize(QSize(35, 20))
        self.WindowMin.setCursor(QCursor(Qt.PointingHandCursor))
        self.WindowMin.setIconSize(QSize(10, 10))
        self.WindowMin.setAutoDefault(True)

        self.horizontalLayoutTitle.addWidget(self.WindowMin)

        self.WindowMax = QPushButton(self.TitleWidget)
        self.WindowMax.setObjectName(u"WindowMax")
        self.WindowMax.setMinimumSize(QSize(35, 20))
        self.WindowMax.setMaximumSize(QSize(35, 20))
        self.WindowMax.setCursor(QCursor(Qt.PointingHandCursor))
        icon = QIcon()
        iconThemeName = u"1"
        if QIcon.hasThemeIcon(iconThemeName):
            icon = QIcon.fromTheme(iconThemeName)
        else:
            icon.addFile(u".", QSize(), QIcon.Normal, QIcon.Off)
        
        self.WindowMax.setIcon(icon)

        self.horizontalLayoutTitle.addWidget(self.WindowMax)

        self.WindowClose = QPushButton(self.TitleWidget)
        self.WindowClose.setObjectName(u"WindowClose")
        self.WindowClose.setMinimumSize(QSize(35, 20))
        self.WindowClose.setMaximumSize(QSize(35, 20))
        self.WindowClose.setCursor(QCursor(Qt.PointingHandCursor))

        self.horizontalLayoutTitle.addWidget(self.WindowClose)

        self.horizontalLayoutTitle.setStretch(0, 1)
        self.horizontalLayoutTitle.setStretch(1, 3)
        self.horizontalLayoutTitle.setStretch(2, 5)
        self.horizontalLayoutTitle.setStretch(3, 1)
        self.horizontalLayoutTitle.setStretch(4, 1)
        self.horizontalLayoutTitle.setStretch(5, 1)

        self.gridLayoutTitle.addWidget(self.TitleWidget, 0, 0, 1, 1)

        self.RealWidget = QWidget(TitleWindow)
        self.RealWidget.setObjectName(u"RealWidget")
        self.RealWidget.setEnabled(True)
        self.RealWidget.setMinimumSize(QSize(35, 30))

        self.gridLayoutTitle.addWidget(self.RealWidget, 1, 0, 1, 1)


        self.retranslateUi(TitleWindow)

        QMetaObject.connectSlotsByName(TitleWindow)
    # setupUi

    def retranslateUi(self, TitleWindow):
        TitleWindow.setWindowTitle(QCoreApplication.translate("TitleWindow", u"PyQCat-Visage", None))
        self.IconLabel.setText("")
        self.TitleLabel.setText(QCoreApplication.translate("TitleWindow", u"PyQCat: Quantum Chip Calibration", None))
#if QT_CONFIG(tooltip)
        self.WindowMin.setToolTip(QCoreApplication.translate("TitleWindow", u"\u6700\u5c0f\u5316", None))
#endif // QT_CONFIG(tooltip)
        self.WindowMin.setText("")
#if QT_CONFIG(tooltip)
        self.WindowMax.setToolTip(QCoreApplication.translate("TitleWindow", u"\u6700\u5927\u5316", None))
#endif // QT_CONFIG(tooltip)
        self.WindowMax.setText("")
#if QT_CONFIG(tooltip)
        self.WindowClose.setToolTip(QCoreApplication.translate("TitleWindow", u"\u5173\u95ed", None))
#endif // QT_CONFIG(tooltip)
        self.WindowClose.setText("")
    # retranslateUi

