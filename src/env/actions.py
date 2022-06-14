import logging
import time

import pydirectinput
import win32gui

from .env_config import ACTION_DELAY, AGENT_KEYMAP, ENV_KEYMAP


class Actor():
    def __init__(self, handle) -> None:
        self.handle = handle
        self.agent_keymap = AGENT_KEYMAP
        self.env_keymap = ENV_KEYMAP

    def agentAction(self, key):
        if key not in self.agent_keymap:
            logging.critical("invalid agent action")
            raise RuntimeError()

        win32gui.SetForegroundWindow(self.handle)
        pydirectinput.press(self.agent_keymap[key])
        logging.debug(f"action: {key}")
        time.sleep(ACTION_DELAY)

    def envAction(self, key):
        if key not in self.env_keymap:
            logging.critical("invalid env action")
            raise RuntimeError()

        win32gui.SetForegroundWindow(self.handle)
        pydirectinput.press(self.env_keymap[key])
        logging.debug(f"env: {key}")
        time.sleep(ACTION_DELAY)
