import pickle

from pymouse import PyMouse
import PyHook3
import pythoncom
import time
class Stop(Exception):
    pass



class ClickNote():

    def __init__(self):
        self.step_list = []
        self.mouse = PyMouse()
        self.hook_manager = PyHook3.HookManager()
        self.step_file ="step.bat"
        self.timestemp = 0

    def load_step(self, file_name: str = None):
        if file_name is None:
            file_name = self.step_file

        with open(file_name, "rb") as f:
            data = f.read()
            self.step_list = pickle.loads(data)


    def run(self):

        self.load_step()
        while True:
            for step in self.step_list:
                sleep_time, polition = step
                time.sleep(sleep_time)
                self.mouse.click(polition[0], polition[1])

            time.sleep(4)

    def get_mouse_event_click(self, event):
        if event.WindowName in ["PyQCat-Visage", "pyQCat Visage", "pyQCat-Visage"]:
            position = event.Position
            timestemp = time.time_ns()
            step = [round((timestemp - self.timestemp) / 10e9, 2), position]
            print(step)
            self.timestemp = timestemp
            self.step_list.append(step)
        else:
            print("MessageName", event.MessageName)
            print("Position", event.Position)
            print("window", event.WindowName)
        return True

    def stop_write_step(self, event):
        with open(self.step_file, "wb") as f:
            data = pickle.dumps(self.step_list)
            f.write(data)
        self.hook_manager.UnhookMouse()
        exit(0)



    def write_step(self):
        self.step_list = []
        # self.hook_manager.MouseAllButtonsDown = self.get_mouse_event_click
        self.hook_manager.MouseRightDown = self.stop_write_step
        self.hook_manager.MouseLeftDown = self.get_mouse_event_click
        self.timestemp = time.time_ns()
        self.hook_manager.HookMouse()
        self.hook_manager.HookKeyboard()
        pythoncom.PumpMessages()


if __name__ == '__main__':
    tt = ClickNote()
    # tt.write_step()
    tt.run()