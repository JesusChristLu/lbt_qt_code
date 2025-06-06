import sys

from pyqcat_visage.backend.backend import Backend
from pyqcat_visage.gui.main_window import VisageGUI

if __name__ == '__main__':
    backend = Backend()
    mainWindow = VisageGUI(backend)
    sys.exit(mainWindow.qApp.exec())
