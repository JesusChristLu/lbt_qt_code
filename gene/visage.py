# import faulthandler

import os


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from multiprocessing import Process, freeze_support

from pyqcat_visage.backend.backend import Backend
from pyqcat_visage.execute.start import async_process
from pyqcat_visage.gui.main_window import VisageGUI
from pyqcat_visage.tool.utilies import kill_process_by_pid


if __name__ == '__main__':
    # faulthandler.enable()
    freeze_support()
    proc = Process(target=async_process)
    proc.start()
    backend = Backend()
    mainWindow = VisageGUI(backend, proc)
    state = mainWindow.qApp.exec()
    kill_process_by_pid(backend.sub_proc.pid)
    sys.exit(state)
