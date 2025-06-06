import ctypes
import inspect
import os
import pickle
import time
import sys
from threading import Event, Thread

import PyHook3
import pythoncom
from pymouse import PyMouse


class Stop(Exception):
    pass


class Clicck(Thread):

    def __init__(self,
                 group: None = None,
                 target=None,
                 name=None,
                 daemon=None,
                 step=None,
                 offset = [0,0],
                 time_sleep=3) -> None:
        super().__init__(group, target, name, daemon=daemon)
        self.step_list = step or []
        self.time_sleep = time_sleep
        self.event = Event()
        self.offset = offset

        self.mouse = PyMouse()

    def run(self) -> None:
        if self.step_list:
            while True:
                for step in self.step_list:
                    sleep_time, polition, button = step
                    self.event.wait(timeout=sleep_time)
                    self.mouse.move(polition[0] - self.offset[0], polition[1] - self.offset[1])
                    self.event.wait(timeout=0.01)
                    self.mouse.click(polition[0], polition[1], button=button)
                self.event.wait(timeout=self.time_sleep)

    def close(self, msg: str = "press"):
        self.stop_thread()

    def stop_thread(self, exctype=SystemExit):
        tid = ctypes.c_long(self.ident)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            tid, ctypes.py_object(exctype))
        if res == 0:
            msg = f"stop thread error: invalid thread id, res: {res}"
            raise EnvironmentError(msg)
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            msg = f"PyThreadState_SetAsyncExc failed, res:{res}"
            raise EnvironmentError(msg)


class ClickNote():

    help_msg = """
    use keyboard could control transcribe mouse and repeat mouse event.
    
    F1: start transcribe mouse.
    F2: end transcribe mouse.
    F3: save mouse step to mouse.
    F4: load local mouse step to programe.
    F5: start run auto mouse test.
    F6: stop auto nouse test.
    Esc: exit.
    F8: show help.
    """

    def __init__(self):
        self.hook_manager = PyHook3.HookManager()
        self.mouse_thread: Thread = None

        self.step_list = []
        self.step_file = "auto.bat"
        self.timestemp = 0
        self.transcribe_flag = False
        self.start_cali = False
        self.offset = (0,0)

        self.key_bind_dict = {
            "F1": self.transcribe_start,
            "F2": self.transcribe_end,
            "F3": self.step_save_local,
            "F4": self.step_load,
            "F5": self.auto_start,
            "F6": self.auto_stop,
            "Escape": self.end,
            "F8": self.show_help,
        }

    def get_mouse_event_click(self, event):
        if self.start_cali:
            print("calibrate mouse")
            self.offset = event.Position
            return True
        
        position = event.Position
        timestemp = time.time_ns()
        if "left" in str(event.MessageName):
            button = 1
        elif "right" in str(event.MessageName):
            button = 2
        else:
            button = 3
        step = [
            round((timestemp - self.timestemp) / 10e8, 2), position, button
        ]
        print(step)
        self.timestemp = timestemp
        self.step_list.append(step)
        
        # if event.WindowName in [
        #         "PyQCat-Visage", "pyQCat Visage", "pyQCat-Visage"
        # ]:
        #     position = event.Position
        #     timestemp = time.time_ns()
        #     if "left" in str(event.MessageName):
        #         button = 1
        #     elif "right" in str(event.MessageName):
        #         button = 2
        #     else:
        #         button = 3
        #     step = [
        #         round((timestemp - self.timestemp) / 10e8, 2), position, button
        #     ]
        #     print(step)
        #     self.timestemp = timestemp
        #     self.step_list.append(step)
        # else:
        #     print("MessageName", event.MessageName)
        #     print("Position", event.Position)
        #     print("window", event.WindowName)
        return True

    def get_keyboard_event_click(self, event):
        key = str(event.Key)
        if key in self.key_bind_dict:
            print("Key", event.Key)
            self.key_bind_dict[key]()
        else:
            print("KeyID", event.KeyID)
            print("ScanCode", event.ScanCode)
            print("Extended", event.Extended)
        return True

    def step_save_local(self):
        with open(self.step_file, "wb") as f:
            data = pickle.dumps(self.step_list)
            f.write(data)

    def step_load(self, file_name: str = None):

        if file_name is None:
            file_name = self.step_file

        if not os.path.exists(file_name):
            print("can't find step file")
            return

        with open(file_name, "rb") as f:
            data = f.read()
            self.step_list = pickle.loads(data)

        print("load step list", self.step_list)

    def transcribe_start(self):
        if self.transcribe_flag:
            print("is transcribing")
            return
        self.step_list = []
        self.timestemp = time.time_ns()
        self.transcribe_flag = True
        self.hook_manager.HookMouse()

    def transcribe_end(self):
        if self.transcribe_flag:
            self.transcribe_flag = False
            self.hook_manager.UnhookMouse()

    def auto_start(self):
        if self.mouse_thread and self.mouse_thread.is_alive():
            print("auto test is running!")
            return

        if not self.step_list:
            print("no step, please transcribe or load first.")
            return

        print("show step", self.step_list)
        self.mouse_thread = Clicck(name="test",
                                   step=self.step_list,
                                   daemon=True)
        self.mouse_thread.start()

    def auto_stop(self):
        if self.mouse_thread and self.mouse_thread.is_alive():
            self.mouse_thread.close()

    def end(self):
        self.hook_manager.UnhookKeyboard()
        self.hook_manager.UnhookMouse()
        self.auto_stop()
        sys.exit(0)
        

    def start(self):
        self.show_help()
        self.hook_manager.MouseAllButtonsDown = self.get_mouse_event_click
        self.hook_manager.KeyDown = self.get_keyboard_event_click

        self.hook_manager.HookKeyboard()
        # self.calibrate_mouse()
        pythoncom.PumpMessages()

    def show_help(self):
        print(self.help_msg)
        
    def calibrate_mouse(self):
        self.start_cali = True
        mouse = PyMouse()
        self.hook_manager.HookMouse()
        mouse.click(100,100, button=2)
        time.sleep(0.01)
        self.start_cali = False
        print("Mouse offset", self.offset)
        self.hook_manager.UnhookMouse()
        
        
        
        


if __name__ == '__main__':
    tt = ClickNote()
    tt.start()