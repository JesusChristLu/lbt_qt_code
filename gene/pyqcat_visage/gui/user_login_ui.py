# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'user_login_ui.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QSizePolicy, QSpacerItem,
    QStackedWidget, QVBoxLayout, QWidget)

from .widgets.combox_custom.combox_search import SearchComboBox

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1079, 335)
        icon = QIcon()
        icon.addFile(u":/logo.png", QSize(), QIcon.Normal, QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSpacer_12 = QSpacerItem(40, 20, QSizePolicy.Preferred, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_12)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalSpacer_11 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_7.addItem(self.horizontalSpacer_11)

        self.logo_label = QLabel(self.centralwidget)
        self.logo_label.setObjectName(u"logo_label")
        self.logo_label.setTextFormat(Qt.AutoText)
        self.logo_label.setPixmap(QPixmap(u":/login-logo.png"))
        self.logo_label.setWordWrap(False)

        self.horizontalLayout_7.addWidget(self.logo_label)

        self.horizontalSpacer_10 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_7.addItem(self.horizontalSpacer_10)

        self.title_label = QLabel(self.centralwidget)
        self.title_label.setObjectName(u"title_label")
        font = QFont()
        font.setFamilies([u"Calibri"])
        font.setPointSize(18)
        font.setBold(True)
        self.title_label.setFont(font)
        self.title_label.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.title_label.setWordWrap(True)

        self.horizontalLayout_7.addWidget(self.title_label)

        self.horizontalSpacer_9 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_7.addItem(self.horizontalSpacer_9)


        self.verticalLayout.addLayout(self.horizontalLayout_7)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        self.note_label = QLabel(self.centralwidget)
        self.note_label.setObjectName(u"note_label")
        self.note_label.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)
        self.note_label.setWordWrap(True)

        self.verticalLayout.addWidget(self.note_label)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_2)


        self.horizontalLayout.addLayout(self.verticalLayout)

        self.horizontalSpacer_14 = QSpacerItem(40, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_14)

        self.stackedWidget = QStackedWidget(self.centralwidget)
        self.stackedWidget.setObjectName(u"stackedWidget")
        self.stackedWidget.setFrameShape(QFrame.Box)
        self.addr_page = QWidget()
        self.addr_page.setObjectName(u"addr_page")
        self.verticalLayout_2 = QVBoxLayout(self.addr_page)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.test_conntext_label = QLabel(self.addr_page)
        self.test_conntext_label.setObjectName(u"test_conntext_label")
        palette = QPalette()
        brush = QBrush(QColor(181, 181, 181, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Button, brush)
        brush1 = QBrush(QColor(223, 214, 207, 255))
        brush1.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Light, brush1)
        brush2 = QBrush(QColor(186, 178, 172, 255))
        brush2.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Midlight, brush2)
        brush3 = QBrush(QColor(74, 71, 69, 255))
        brush3.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Dark, brush3)
        brush4 = QBrush(QColor(99, 95, 92, 255))
        brush4.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Mid, brush4)
        palette.setBrush(QPalette.Active, QPalette.Base, brush)
        palette.setBrush(QPalette.Active, QPalette.Window, brush)
        brush5 = QBrush(QColor(0, 0, 0, 255))
        brush5.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Shadow, brush5)
        brush6 = QBrush(QColor(202, 199, 196, 255))
        brush6.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.AlternateBase, brush6)
        brush7 = QBrush(QColor(0, 0, 0, 127))
        brush7.setStyle(Qt.SolidPattern)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Active, QPalette.PlaceholderText, brush7)
#endif
        palette.setBrush(QPalette.Inactive, QPalette.Button, brush)
        brush8 = QBrush(QColor(255, 255, 255, 255))
        brush8.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Inactive, QPalette.Light, brush8)
        brush9 = QBrush(QColor(227, 227, 227, 255))
        brush9.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Inactive, QPalette.Midlight, brush9)
        brush10 = QBrush(QColor(160, 160, 160, 255))
        brush10.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Inactive, QPalette.Dark, brush10)
        palette.setBrush(QPalette.Inactive, QPalette.Mid, brush10)
        palette.setBrush(QPalette.Inactive, QPalette.Base, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Window, brush)
        brush11 = QBrush(QColor(105, 105, 105, 255))
        brush11.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Inactive, QPalette.Shadow, brush11)
        brush12 = QBrush(QColor(245, 245, 245, 255))
        brush12.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Inactive, QPalette.AlternateBase, brush12)
        brush13 = QBrush(QColor(0, 0, 0, 128))
        brush13.setStyle(Qt.SolidPattern)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush13)
#endif
        palette.setBrush(QPalette.Disabled, QPalette.Button, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Light, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Midlight, brush2)
        palette.setBrush(QPalette.Disabled, QPalette.Dark, brush3)
        palette.setBrush(QPalette.Disabled, QPalette.Mid, brush4)
        palette.setBrush(QPalette.Disabled, QPalette.Base, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Window, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Shadow, brush5)
        palette.setBrush(QPalette.Disabled, QPalette.AlternateBase, brush12)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush13)
#endif
        self.test_conntext_label.setPalette(palette)
        font1 = QFont()
        font1.setPointSize(11)
        font1.setBold(True)
        self.test_conntext_label.setFont(font1)
        self.test_conntext_label.setStyleSheet(u"background-color: rgb(181, 181, 181);")

        self.verticalLayout_2.addWidget(self.test_conntext_label)

        self.verticalSpacer_6 = QSpacerItem(20, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer_6)

        self.widget = QWidget(self.addr_page)
        self.widget.setObjectName(u"widget")
        self.horizontalLayout_13 = QHBoxLayout(self.widget)
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.horizontalSpacer_27 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_13.addItem(self.horizontalSpacer_27)

        self.ip_label = QLabel(self.widget)
        self.ip_label.setObjectName(u"ip_label")

        self.horizontalLayout_13.addWidget(self.ip_label)

        self.ip_edit = QLineEdit(self.widget)
        self.ip_edit.setObjectName(u"ip_edit")

        self.horizontalLayout_13.addWidget(self.ip_edit)

        self.horizontalSpacer_28 = QSpacerItem(79, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_13.addItem(self.horizontalSpacer_28)

        self.horizontalLayout_13.setStretch(0, 2)
        self.horizontalLayout_13.setStretch(1, 2)
        self.horizontalLayout_13.setStretch(2, 6)
        self.horizontalLayout_13.setStretch(3, 3)

        self.verticalLayout_2.addWidget(self.widget)

        self.widget_2 = QWidget(self.addr_page)
        self.widget_2.setObjectName(u"widget_2")
        self.horizontalLayout_14 = QHBoxLayout(self.widget_2)
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.horizontalSpacer_29 = QSpacerItem(51, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_14.addItem(self.horizontalSpacer_29)

        self.port_label = QLabel(self.widget_2)
        self.port_label.setObjectName(u"port_label")

        self.horizontalLayout_14.addWidget(self.port_label)

        self.port_edit = QLineEdit(self.widget_2)
        self.port_edit.setObjectName(u"port_edit")

        self.horizontalLayout_14.addWidget(self.port_edit)

        self.horizontalSpacer_30 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_14.addItem(self.horizontalSpacer_30)

        self.horizontalLayout_14.setStretch(0, 2)
        self.horizontalLayout_14.setStretch(1, 2)
        self.horizontalLayout_14.setStretch(2, 6)
        self.horizontalLayout_14.setStretch(3, 3)

        self.verticalLayout_2.addWidget(self.widget_2)

        self.verticalSpacer_7 = QSpacerItem(20, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer_7)

        self.horizontalLayout_15 = QHBoxLayout()
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.horizontalSpacer_34 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_15.addItem(self.horizontalSpacer_34)

        self.state_label = QLabel(self.addr_page)
        self.state_label.setObjectName(u"state_label")

        self.horizontalLayout_15.addWidget(self.state_label)

        self.test_connect_button = QPushButton(self.addr_page)
        self.test_connect_button.setObjectName(u"test_connect_button")
        icon1 = QIcon()
        icon1.addFile(u":/wifi-line.png", QSize(), QIcon.Normal, QIcon.Off)
        self.test_connect_button.setIcon(icon1)

        self.horizontalLayout_15.addWidget(self.test_connect_button)

        self.horizontalSpacer_35 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_15.addItem(self.horizontalSpacer_35)

        self.connect_button = QPushButton(self.addr_page)
        self.connect_button.setObjectName(u"connect_button")
        font2 = QFont()
        font2.setBold(True)
        self.connect_button.setFont(font2)
        icon2 = QIcon()
        icon2.addFile(u":/connect.png", QSize(), QIcon.Normal, QIcon.Off)
        self.connect_button.setIcon(icon2)

        self.horizontalLayout_15.addWidget(self.connect_button)

        self.horizontalSpacer_33 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_15.addItem(self.horizontalSpacer_33)

        self.horizontalLayout_15.setStretch(0, 2)
        self.horizontalLayout_15.setStretch(1, 1)
        self.horizontalLayout_15.setStretch(2, 3)
        self.horizontalLayout_15.setStretch(3, 1)
        self.horizontalLayout_15.setStretch(4, 3)
        self.horizontalLayout_15.setStretch(5, 2)

        self.verticalLayout_2.addLayout(self.horizontalLayout_15)

        self.verticalSpacer_8 = QSpacerItem(20, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer_8)

        self.verticalLayout_2.setStretch(0, 10)
        self.verticalLayout_2.setStretch(1, 5)
        self.verticalLayout_2.setStretch(2, 1)
        self.verticalLayout_2.setStretch(3, 1)
        self.verticalLayout_2.setStretch(4, 1)
        self.verticalLayout_2.setStretch(5, 1)
        self.verticalLayout_2.setStretch(6, 1)
        self.stackedWidget.addWidget(self.addr_page)
        self.login_page = QWidget()
        self.login_page.setObjectName(u"login_page")
        self.gridLayout = QGridLayout(self.login_page)
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_8)

        self.is_remember_box = QCheckBox(self.login_page)
        self.is_remember_box.setObjectName(u"is_remember_box")
        self.is_remember_box.setChecked(False)

        self.horizontalLayout_5.addWidget(self.is_remember_box)

        self.login_pushButton = QPushButton(self.login_page)
        self.login_pushButton.setObjectName(u"login_pushButton")
        palette1 = QPalette()
        brush14 = QBrush(QColor(50, 206, 42, 255))
        brush14.setStyle(Qt.SolidPattern)
        palette1.setBrush(QPalette.Active, QPalette.ButtonText, brush14)
        palette1.setBrush(QPalette.Inactive, QPalette.ButtonText, brush5)
        brush15 = QBrush(QColor(120, 120, 120, 255))
        brush15.setStyle(Qt.SolidPattern)
        palette1.setBrush(QPalette.Disabled, QPalette.ButtonText, brush15)
        self.login_pushButton.setPalette(palette1)
        self.login_pushButton.setFont(font2)
        icon3 = QIcon()
        icon3.addFile(u":/login.png", QSize(), QIcon.Normal, QIcon.Off)
        self.login_pushButton.setIcon(icon3)

        self.horizontalLayout_5.addWidget(self.login_pushButton)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_2)

        self.horizontalLayout_5.setStretch(1, 2)
        self.horizontalLayout_5.setStretch(2, 2)

        self.gridLayout.addLayout(self.horizontalLayout_5, 4, 0, 1, 1)

        self.label = QLabel(self.login_page)
        self.label.setObjectName(u"label")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font3 = QFont()
        font3.setPointSize(11)
        font3.setBold(True)
        font3.setUnderline(False)
        font3.setStrikeOut(False)
        self.label.setFont(font3)
        self.label.setStyleSheet(u"background-color:rgb(181, 181, 181)")
        self.label.setFrameShape(QFrame.Panel)
        self.label.setFrameShadow(QFrame.Raised)
        self.label.setTextFormat(Qt.AutoText)
        self.label.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_7)

        self.new_user_label = QLabel(self.login_page)
        self.new_user_label.setObjectName(u"new_user_label")
        palette2 = QPalette()
        brush16 = QBrush(QColor(0, 0, 212, 255))
        brush16.setStyle(Qt.SolidPattern)
        palette2.setBrush(QPalette.Active, QPalette.WindowText, brush16)
        palette2.setBrush(QPalette.Inactive, QPalette.WindowText, brush5)
        palette2.setBrush(QPalette.Disabled, QPalette.WindowText, brush15)
        self.new_user_label.setPalette(palette2)
        self.new_user_label.setFont(font2)

        self.horizontalLayout_2.addWidget(self.new_user_label)

        self.create_pushButton = QPushButton(self.login_page)
        self.create_pushButton.setObjectName(u"create_pushButton")
        palette3 = QPalette()
        brush17 = QBrush(QColor(0, 0, 218, 255))
        brush17.setStyle(Qt.SolidPattern)
        palette3.setBrush(QPalette.Active, QPalette.ButtonText, brush17)
        palette3.setBrush(QPalette.Inactive, QPalette.ButtonText, brush5)
        palette3.setBrush(QPalette.Disabled, QPalette.ButtonText, brush15)
        self.create_pushButton.setPalette(palette3)
        icon4 = QIcon()
        icon4.addFile(u":/register-an-account.png", QSize(), QIcon.Normal, QIcon.Off)
        self.create_pushButton.setIcon(icon4)
        self.create_pushButton.setIconSize(QSize(24, 24))

        self.horizontalLayout_2.addWidget(self.create_pushButton)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.horizontalLayout_2.setStretch(1, 2)
        self.horizontalLayout_2.setStretch(2, 2)

        self.gridLayout.addLayout(self.horizontalLayout_2, 5, 0, 1, 1)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_5)

        self.user_name_label = QLabel(self.login_page)
        self.user_name_label.setObjectName(u"user_name_label")

        self.horizontalLayout_4.addWidget(self.user_name_label)

        self.user_name_box = QComboBox(self.login_page)
        self.user_name_box.setObjectName(u"user_name_box")
        self.user_name_box.setEditable(True)

        self.horizontalLayout_4.addWidget(self.user_name_box)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_6)

        self.horizontalLayout_4.setStretch(1, 1)
        self.horizontalLayout_4.setStretch(2, 2)

        self.gridLayout.addLayout(self.horizontalLayout_4, 2, 0, 1, 1)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_3)

        self.password_label = QLabel(self.login_page)
        self.password_label.setObjectName(u"password_label")

        self.horizontalLayout_3.addWidget(self.password_label)

        self.pwd_linedit_login = QLineEdit(self.login_page)
        self.pwd_linedit_login.setObjectName(u"pwd_linedit_login")
        self.pwd_linedit_login.setEchoMode(QLineEdit.Password)

        self.horizontalLayout_3.addWidget(self.pwd_linedit_login)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_4)

        self.horizontalLayout_3.setStretch(1, 1)
        self.horizontalLayout_3.setStretch(2, 2)

        self.gridLayout.addLayout(self.horizontalLayout_3, 3, 0, 1, 1)

        self.verticalSpacer_5 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout.addItem(self.verticalSpacer_5, 6, 0, 1, 1)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalSpacer_15 = QSpacerItem(40, 20, QSizePolicy.Preferred, QSizePolicy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_15)

        self.forget_button = QPushButton(self.login_page)
        self.forget_button.setObjectName(u"forget_button")
        palette4 = QPalette()
        brush18 = QBrush(QColor(255, 0, 0, 255))
        brush18.setStyle(Qt.SolidPattern)
        palette4.setBrush(QPalette.Active, QPalette.ButtonText, brush18)
        palette4.setBrush(QPalette.Inactive, QPalette.ButtonText, brush5)
        palette4.setBrush(QPalette.Disabled, QPalette.ButtonText, brush15)
        self.forget_button.setPalette(palette4)
        font4 = QFont()
        font4.setBold(True)
        font4.setItalic(True)
        font4.setUnderline(True)
        self.forget_button.setFont(font4)
        self.forget_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.forget_button.setFlat(True)

        self.horizontalLayout_6.addWidget(self.forget_button)

        self.back_connect_putton = QPushButton(self.login_page)
        self.back_connect_putton.setObjectName(u"back_connect_putton")
        icon5 = QIcon()
        icon5.addFile(u":/cancel.png", QSize(), QIcon.Normal, QIcon.Off)
        self.back_connect_putton.setIcon(icon5)

        self.horizontalLayout_6.addWidget(self.back_connect_putton)

        self.horizontalSpacer_16 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_16)

        self.horizontalLayout_6.setStretch(1, 3)
        self.horizontalLayout_6.setStretch(2, 1)

        self.gridLayout.addLayout(self.horizontalLayout_6, 7, 0, 1, 1)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.verticalSpacer_3, 1, 0, 1, 1)

        self.gridLayout.setRowStretch(0, 2)
        self.gridLayout.setRowStretch(1, 1)
        self.gridLayout.setRowStretch(2, 2)
        self.gridLayout.setRowStretch(3, 2)
        self.gridLayout.setRowStretch(4, 2)
        self.gridLayout.setRowStretch(5, 2)
        self.gridLayout.setRowStretch(6, 5)
        self.gridLayout.setRowStretch(7, 2)
        self.stackedWidget.addWidget(self.login_page)
        self.sign_page = QWidget()
        self.sign_page.setObjectName(u"sign_page")
        self.verticalLayout_3 = QVBoxLayout(self.sign_page)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.title_label_register = QLabel(self.sign_page)
        self.title_label_register.setObjectName(u"title_label_register")
        sizePolicy.setHeightForWidth(self.title_label_register.sizePolicy().hasHeightForWidth())
        self.title_label_register.setSizePolicy(sizePolicy)
        self.title_label_register.setFont(font3)
        self.title_label_register.setStyleSheet(u"background-color:rgb(181, 181, 181)")
        self.title_label_register.setFrameShape(QFrame.Panel)
        self.title_label_register.setFrameShadow(QFrame.Raised)
        self.title_label_register.setTextFormat(Qt.AutoText)
        self.title_label_register.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.verticalLayout_3.addWidget(self.title_label_register)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.verticalLayout_3.addItem(self.verticalSpacer_4)

        self.horizontalLayout_21 = QHBoxLayout()
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.horizontalSpacer_46 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_21.addItem(self.horizontalSpacer_46)

        self.user_name_label_register_3 = QLabel(self.sign_page)
        self.user_name_label_register_3.setObjectName(u"user_name_label_register_3")

        self.horizontalLayout_21.addWidget(self.user_name_label_register_3)

        self.GroupComboBox = SearchComboBox(self.sign_page)
        self.GroupComboBox.setObjectName(u"GroupComboBox")

        self.horizontalLayout_21.addWidget(self.GroupComboBox)

        self.horizontalSpacer_47 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_21.addItem(self.horizontalSpacer_47)

        self.horizontalLayout_21.setStretch(1, 1)
        self.horizontalLayout_21.setStretch(2, 2)

        self.verticalLayout_3.addLayout(self.horizontalLayout_21)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.horizontalSpacer_17 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_11.addItem(self.horizontalSpacer_17)

        self.user_name_label_register = QLabel(self.sign_page)
        self.user_name_label_register.setObjectName(u"user_name_label_register")

        self.horizontalLayout_11.addWidget(self.user_name_label_register)

        self.user_name_lineEdit_register = QLineEdit(self.sign_page)
        self.user_name_lineEdit_register.setObjectName(u"user_name_lineEdit_register")

        self.horizontalLayout_11.addWidget(self.user_name_lineEdit_register)

        self.horizontalSpacer_18 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_11.addItem(self.horizontalSpacer_18)

        self.horizontalLayout_11.setStretch(1, 1)
        self.horizontalLayout_11.setStretch(2, 2)

        self.verticalLayout_3.addLayout(self.horizontalLayout_11)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.horizontalSpacer_19 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_10.addItem(self.horizontalSpacer_19)

        self.pwd_label_register = QLabel(self.sign_page)
        self.pwd_label_register.setObjectName(u"pwd_label_register")

        self.horizontalLayout_10.addWidget(self.pwd_label_register)

        self.pwd_lineEdit_register = QLineEdit(self.sign_page)
        self.pwd_lineEdit_register.setObjectName(u"pwd_lineEdit_register")
        self.pwd_lineEdit_register.setEchoMode(QLineEdit.Password)

        self.horizontalLayout_10.addWidget(self.pwd_lineEdit_register)

        self.horizontalSpacer_20 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_10.addItem(self.horizontalSpacer_20)

        self.horizontalLayout_10.setStretch(1, 1)
        self.horizontalLayout_10.setStretch(2, 2)

        self.verticalLayout_3.addLayout(self.horizontalLayout_10)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalSpacer_21 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer_21)

        self.rpwd_label_register = QLabel(self.sign_page)
        self.rpwd_label_register.setObjectName(u"rpwd_label_register")

        self.horizontalLayout_9.addWidget(self.rpwd_label_register)

        self.rpwd_lineEdit = QLineEdit(self.sign_page)
        self.rpwd_lineEdit.setObjectName(u"rpwd_lineEdit")
        self.rpwd_lineEdit.setEchoMode(QLineEdit.Password)

        self.horizontalLayout_9.addWidget(self.rpwd_lineEdit)

        self.horizontalSpacer_22 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer_22)

        self.horizontalLayout_9.setStretch(1, 1)
        self.horizontalLayout_9.setStretch(2, 2)

        self.verticalLayout_3.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalSpacer_23 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_8.addItem(self.horizontalSpacer_23)

        self.mail_label_register = QLabel(self.sign_page)
        self.mail_label_register.setObjectName(u"mail_label_register")
        self.mail_label_register.setAutoFillBackground(False)
        self.mail_label_register.setStyleSheet(u"image: url(:/label_mail-register.png);")
        self.mail_label_register.setTextFormat(Qt.AutoText)
        self.mail_label_register.setScaledContents(False)
        self.mail_label_register.setWordWrap(False)

        self.horizontalLayout_8.addWidget(self.mail_label_register)

        self.mail_lineEdit_register = QLineEdit(self.sign_page)
        self.mail_lineEdit_register.setObjectName(u"mail_lineEdit_register")

        self.horizontalLayout_8.addWidget(self.mail_lineEdit_register)

        self.horizontalSpacer_24 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_8.addItem(self.horizontalSpacer_24)

        self.horizontalLayout_8.setStretch(1, 1)
        self.horizontalLayout_8.setStretch(2, 2)

        self.verticalLayout_3.addLayout(self.horizontalLayout_8)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.horizontalSpacer_25 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_12.addItem(self.horizontalSpacer_25)

        self.back_putton = QPushButton(self.sign_page)
        self.back_putton.setObjectName(u"back_putton")
        self.back_putton.setIcon(icon5)

        self.horizontalLayout_12.addWidget(self.back_putton)

        self.create_account_putton = QPushButton(self.sign_page)
        self.create_account_putton.setObjectName(u"create_account_putton")
        self.create_account_putton.setIcon(icon4)

        self.horizontalLayout_12.addWidget(self.create_account_putton)

        self.horizontalSpacer_26 = QSpacerItem(40, 20, QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.horizontalLayout_12.addItem(self.horizontalSpacer_26)


        self.verticalLayout_3.addLayout(self.horizontalLayout_12)

        self.stackedWidget.addWidget(self.sign_page)
        self.find_page = QWidget()
        self.find_page.setObjectName(u"find_page")
        self.gridLayout_3 = QGridLayout(self.find_page)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.title_label_register_2 = QLabel(self.find_page)
        self.title_label_register_2.setObjectName(u"title_label_register_2")
        sizePolicy.setHeightForWidth(self.title_label_register_2.sizePolicy().hasHeightForWidth())
        self.title_label_register_2.setSizePolicy(sizePolicy)
        self.title_label_register_2.setFont(font3)
        self.title_label_register_2.setStyleSheet(u"background-color:rgb(181, 181, 181)")
        self.title_label_register_2.setFrameShape(QFrame.Panel)
        self.title_label_register_2.setFrameShadow(QFrame.Raised)
        self.title_label_register_2.setTextFormat(Qt.AutoText)
        self.title_label_register_2.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.title_label_register_2, 0, 0, 1, 1)

        self.verticalSpacer_9 = QSpacerItem(20, 41, QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.gridLayout_3.addItem(self.verticalSpacer_9, 1, 0, 1, 1)

        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.horizontalSpacer_42 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_16.addItem(self.horizontalSpacer_42)

        self.user_name_label_register_2 = QLabel(self.find_page)
        self.user_name_label_register_2.setObjectName(u"user_name_label_register_2")

        self.horizontalLayout_16.addWidget(self.user_name_label_register_2)

        self.user_name_lineEdit_find = QLineEdit(self.find_page)
        self.user_name_lineEdit_find.setObjectName(u"user_name_lineEdit_find")

        self.horizontalLayout_16.addWidget(self.user_name_lineEdit_find)

        self.horizontalSpacer_43 = QSpacerItem(40, 20, QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.horizontalLayout_16.addItem(self.horizontalSpacer_43)

        self.horizontalLayout_16.setStretch(1, 1)
        self.horizontalLayout_16.setStretch(2, 2)

        self.gridLayout_3.addLayout(self.horizontalLayout_16, 2, 0, 1, 1)

        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.horizontalSpacer_40 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_17.addItem(self.horizontalSpacer_40)

        self.pwd_label_register_2 = QLabel(self.find_page)
        self.pwd_label_register_2.setObjectName(u"pwd_label_register_2")

        self.horizontalLayout_17.addWidget(self.pwd_label_register_2)

        self.pre_pwd_edit = QLineEdit(self.find_page)
        self.pre_pwd_edit.setObjectName(u"pre_pwd_edit")
        self.pre_pwd_edit.setEchoMode(QLineEdit.Password)

        self.horizontalLayout_17.addWidget(self.pre_pwd_edit)

        self.horizontalSpacer_41 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_17.addItem(self.horizontalSpacer_41)

        self.horizontalLayout_17.setStretch(1, 1)
        self.horizontalLayout_17.setStretch(2, 2)

        self.gridLayout_3.addLayout(self.horizontalLayout_17, 3, 0, 1, 1)

        self.horizontalLayout_18 = QHBoxLayout()
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.horizontalSpacer_37 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_18.addItem(self.horizontalSpacer_37)

        self.mail_label_register_2 = QLabel(self.find_page)
        self.mail_label_register_2.setObjectName(u"mail_label_register_2")
        self.mail_label_register_2.setAutoFillBackground(False)
        self.mail_label_register_2.setStyleSheet(u"image: url(:/label_mail-register.png);")
        self.mail_label_register_2.setTextFormat(Qt.AutoText)
        self.mail_label_register_2.setScaledContents(False)
        self.mail_label_register_2.setWordWrap(False)

        self.horizontalLayout_18.addWidget(self.mail_label_register_2)

        self.mail_lineEdit_find = QLineEdit(self.find_page)
        self.mail_lineEdit_find.setObjectName(u"mail_lineEdit_find")

        self.horizontalLayout_18.addWidget(self.mail_lineEdit_find)

        self.horizontalSpacer_36 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_18.addItem(self.horizontalSpacer_36)

        self.horizontalLayout_18.setStretch(1, 1)
        self.horizontalLayout_18.setStretch(2, 2)

        self.gridLayout_3.addLayout(self.horizontalLayout_18, 4, 0, 1, 1)

        self.horizontalLayout_20 = QHBoxLayout()
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.horizontalSpacer_44 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_20.addItem(self.horizontalSpacer_44)

        self.label_2 = QLabel(self.find_page)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_20.addWidget(self.label_2)

        self.new_pwd_edit = QLineEdit(self.find_page)
        self.new_pwd_edit.setObjectName(u"new_pwd_edit")

        self.horizontalLayout_20.addWidget(self.new_pwd_edit)

        self.horizontalSpacer_45 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_20.addItem(self.horizontalSpacer_45)

        self.horizontalLayout_20.setStretch(1, 1)
        self.horizontalLayout_20.setStretch(2, 2)

        self.gridLayout_3.addLayout(self.horizontalLayout_20, 5, 0, 1, 1)

        self.horizontalLayout_19 = QHBoxLayout()
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.horizontalSpacer_39 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_19.addItem(self.horizontalSpacer_39)

        self.back_putton_find = QPushButton(self.find_page)
        self.back_putton_find.setObjectName(u"back_putton_find")
        self.back_putton_find.setIcon(icon5)

        self.horizontalLayout_19.addWidget(self.back_putton_find)

        self.rebuild_putton = QPushButton(self.find_page)
        self.rebuild_putton.setObjectName(u"rebuild_putton")
        icon6 = QIcon()
        icon6.addFile(u":/refresh.png", QSize(), QIcon.Normal, QIcon.Off)
        self.rebuild_putton.setIcon(icon6)

        self.horizontalLayout_19.addWidget(self.rebuild_putton)

        self.horizontalSpacer_38 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_19.addItem(self.horizontalSpacer_38)

        self.horizontalLayout_19.setStretch(0, 3)
        self.horizontalLayout_19.setStretch(1, 1)
        self.horizontalLayout_19.setStretch(2, 1)

        self.gridLayout_3.addLayout(self.horizontalLayout_19, 6, 0, 1, 1)

        self.stackedWidget.addWidget(self.find_page)

        self.horizontalLayout.addWidget(self.stackedWidget)

        self.horizontalSpacer_13 = QSpacerItem(40, 20, QSizePolicy.Preferred, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_13)

        MainWindow.setCentralWidget(self.centralwidget)
        QWidget.setTabOrder(self.pwd_linedit_login, self.is_remember_box)
        QWidget.setTabOrder(self.is_remember_box, self.login_pushButton)
        QWidget.setTabOrder(self.login_pushButton, self.create_pushButton)
        QWidget.setTabOrder(self.create_pushButton, self.pwd_lineEdit_register)
        QWidget.setTabOrder(self.pwd_lineEdit_register, self.create_account_putton)
        QWidget.setTabOrder(self.create_account_putton, self.user_name_lineEdit_register)
        QWidget.setTabOrder(self.user_name_lineEdit_register, self.rpwd_lineEdit)
        QWidget.setTabOrder(self.rpwd_lineEdit, self.mail_lineEdit_register)

        self.retranslateUi(MainWindow)
        self.create_pushButton.clicked.connect(MainWindow.show_create_account)
        self.login_pushButton.clicked.connect(MainWindow.login_to_system)
        self.back_connect_putton.clicked.connect(MainWindow.back_up_page)
        self.user_name_box.currentIndexChanged.connect(MainWindow.choose_user)
        self.back_putton.clicked.connect(MainWindow.back_up_page)
        self.create_account_putton.clicked.connect(MainWindow.create_account)
        self.test_connect_button.clicked.connect(MainWindow.test_connect)
        self.back_putton_find.clicked.connect(MainWindow.back_up_page)
        self.forget_button.clicked.connect(MainWindow.forget_password_link)
        self.rebuild_putton.clicked.connect(MainWindow.find_account)
        self.connect_button.clicked.connect(MainWindow.take_connect)

        self.stackedWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Login", None))
        self.logo_label.setText("")
        self.title_label.setText(QCoreApplication.translate("MainWindow", u"Calibrate quantum chip.\n"
"Right at your fingertips.", None))
        self.note_label.setText(QCoreApplication.translate("MainWindow", u"Origin Quantum offers pyQCat series software packages and applications for users to do measure, characterization and calibtration on quantum chip.\n"
"\n"
"If you encounter any problems or have some good suggestions, please contact our development team. \n"
"\n"
"Mail: shq@originqc.com", None))
        self.test_conntext_label.setText(QCoreApplication.translate("MainWindow", u"User Data Serive Test Connect !", None))
        self.ip_label.setText(QCoreApplication.translate("MainWindow", u"IP", None))
        self.port_label.setText(QCoreApplication.translate("MainWindow", u"PORT", None))
        self.state_label.setText("")
        self.test_connect_button.setText(QCoreApplication.translate("MainWindow", u"TestConnect", None))
        self.connect_button.setText(QCoreApplication.translate("MainWindow", u"Connect", None))
        self.is_remember_box.setText(QCoreApplication.translate("MainWindow", u"Remember me", None))
        self.login_pushButton.setText(QCoreApplication.translate("MainWindow", u"Login", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Login to pyQCat !", None))
        self.new_user_label.setText(QCoreApplication.translate("MainWindow", u"New to pyQCat ?", None))
#if QT_CONFIG(statustip)
        self.create_pushButton.setStatusTip(QCoreApplication.translate("MainWindow", u"Create a pyQCat account.", None))
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        self.create_pushButton.setWhatsThis(QCoreApplication.translate("MainWindow", u"Create a pyQCat account.", None))
#endif // QT_CONFIG(whatsthis)
        self.create_pushButton.setText(QCoreApplication.translate("MainWindow", u"Create an account.", None))
        self.user_name_label.setText(QCoreApplication.translate("MainWindow", u"User Name", None))
        self.password_label.setText(QCoreApplication.translate("MainWindow", u"Password", None))
        self.pwd_linedit_login.setText("")
        self.forget_button.setText(QCoreApplication.translate("MainWindow", u"Forget your password or accout ?", None))
        self.back_connect_putton.setText(QCoreApplication.translate("MainWindow", u"Back", None))
        self.title_label_register.setText(QCoreApplication.translate("MainWindow", u"Create a pyQCat account !", None))
        self.user_name_label_register_3.setText(QCoreApplication.translate("MainWindow", u"Group", None))
        self.user_name_label_register.setText(QCoreApplication.translate("MainWindow", u"User Name", None))
        self.pwd_label_register.setText(QCoreApplication.translate("MainWindow", u"Password", None))
        self.rpwd_label_register.setText(QCoreApplication.translate("MainWindow", u"Repeat ", None))
        self.mail_label_register.setText(QCoreApplication.translate("MainWindow", u"E-mail", None))
        self.back_putton.setText(QCoreApplication.translate("MainWindow", u"Back", None))
        self.create_account_putton.setText(QCoreApplication.translate("MainWindow", u"Create", None))
        self.title_label_register_2.setText(QCoreApplication.translate("MainWindow", u"Find a pyQCat account !", None))
        self.user_name_label_register_2.setText(QCoreApplication.translate("MainWindow", u"User Name", None))
        self.pwd_label_register_2.setText(QCoreApplication.translate("MainWindow", u"Pre Password", None))
        self.mail_label_register_2.setText(QCoreApplication.translate("MainWindow", u"E-mail", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"New Password", None))
        self.back_putton_find.setText(QCoreApplication.translate("MainWindow", u"Back", None))
        self.rebuild_putton.setText(QCoreApplication.translate("MainWindow", u"Rebuild", None))
    # retranslateUi

