# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage.
#
# Copyright (c) 2021-2025 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11/02
# __author:       HanQing Shi
"""visage log."""

from datetime import datetime

import zmq
from PySide6.QtCore import QThread, Signal

from pyqcat_visage.config import GUI_CONFIG
from pyqcat_visage.tool.utilies import kill_old_process
from pyQCat.invoker import DEFAULT_PORT

DEFAULT_LOG_LEVEL = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50
}


def init_logger_service(qaio_log_addr: str = None):
    log_service = LogService()
    log_service.start()
    return log_service


class LogService(QThread):
    log_message = Signal(str, int)

    qaio_message = Signal(str, int)
    def __init__(self) -> None:
        super().__init__()
        self.sock_monster: zmq.Socket = None
        self.sock_visage: zmq.Socket = None
        self.port_monster: int = DEFAULT_PORT
        self.port_visage: int = DEFAULT_PORT + 1
        self.qaio_log_addr: str = None
        self.sock_qaio = None
        self.update_sock = True

    def _init_sock(self, port: int) -> zmq.Socket:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.bind(f"tcp://127.0.0.1:{port}")
        return sock

    @staticmethod
    def adapter(level_no: int) -> str:
        for value in GUI_CONFIG.log_conf.set_levels:
            if value.no == level_no:
                return value.name
        return "INFO"

    def trans_log(self, level, msg, log_type: int = 1):
        msg = msg.decode().strip()
        if msg == "close_visage":
            return True
        level = level.decode()
        level_no = 10
        level_name = "INFO"
        if level in DEFAULT_LOG_LEVEL:
            level_name = level
            level_no = DEFAULT_LOG_LEVEL[level]
        if isinstance(level, str) and level.startswith("Level"):
            level_no = int(level[-2:])
            level_name = self.adapter(level_no)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = "|".join([timestamp, level_name, str(level_no), level, msg])
        if log_type in [1, 2]:
            self.log_message.emit(msg, log_type)
        elif log_type == 3:
            self.qaio_message.emit(msg, log_type)
        return False

    def set_qaio_addr(self, addr):
        self.qaio_log_addr = f"tcp://{addr}"

    def refresh_qaio_sock(self, addr):
        self.set_qaio_addr(addr)
        self.update_sock = True

    def run(self) -> None:
        #  monster log socket
        kill_old_process(self.port_monster)
        kill_old_process(self.port_visage)

        self.sock_monster = self._init_sock(self.port_monster)
        # visage log socket
        self.sock_visage = self._init_sock(self.port_visage)

        poller = zmq.Poller()
        poller.register(self.sock_monster, zmq.POLLIN)
        poller.register(self.sock_visage, zmq.POLLIN)
        try:
            while True:
                if self.update_sock and self.qaio_log_addr:
                    if self.sock_qaio:
                        poller.unregister(self.sock_qaio)
                        self.sock_qaio.close()
                    ctx = zmq.Context.instance()
                    self.sock_qaio = ctx.socket(zmq.SUB)
                    self.sock_qaio.setsockopt(zmq.SUBSCRIBE, b"")
                    self.sock_qaio.connect(self.qaio_log_addr)
                    poller.register(self.sock_qaio, zmq.POLLIN)
                    self.update_sock = False
                res = dict(poller.poll(1000))
                if self.sock_visage in res:
                    level, msg = self.sock_visage.recv_multipart()
                    if self.trans_log(level, msg, log_type=2):
                        break
                if self.sock_monster in res:
                    level, msg = self.sock_monster.recv_multipart()
                    if self.trans_log(level, msg, log_type=1):
                        break

                if self.sock_qaio and self.sock_qaio in res:
                    level, msg = self.sock_qaio.recv_multipart()
                    if self.trans_log(level, msg, log_type=3):
                        break

            poller.unregister(self.sock_monster)
            poller.unregister(self.sock_visage)

        except Exception:
            import traceback

            print("log get error")
            print(traceback.format_exc())


# def setup_logger(log_handler=sys.stderr,
#                  log_format: Union[str, Any] = GUI_CONFIG.log_conf.format,
#                  log_level: int = GUI_CONFIG.log_conf.level,
#                  set_levels: list = GUI_CONFIG.log_conf.set_levels):
#     """Setup the logger to work with QTextEdit and command line.
#
#     `level_stream` and `level_base`:
#     You can set a different logging level for each logging handler, however you have
#     to set the logger's level to the "lowest".
#
#     Integrates logging with the warnings module.
#
#     Args:
#         logger_name (str): Name of the log.
#         log_format (format): Format of the log.
#         log_level (int): Level ot the log.
#         log_handler : loguru sink.
#         set_levels (List): Custom level config.
#
#     Returns:
#         logger: The logger instance of loguru.
#     """
#     logger.remove()
#     logger.add(sink=log_handler,
#                format=log_format,
#                level=log_level)
#     if set_levels:
#         for set_level in set_levels:
#             logger.level(**set_level)
#     return logger
