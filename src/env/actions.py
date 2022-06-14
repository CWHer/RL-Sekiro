import logging
import time

import pydirectinput
from utils import timeLog

from .env_config import ACTION_DELAY, AGENT_KEYMAP, ENV_KEYMAP


class Actor():
    def __init__(self, handle) -> None:
        self.handle = handle
        self.agent_keymap = AGENT_KEYMAP
        self.env_keymap = ENV_KEYMAP

    # @timeLog
    def agentAction(self, key, action_delay=False):
        if key not in self.agent_keymap:
            logging.critical("invalid agent action")
            raise RuntimeError()

        pydirectinput.press(self.agent_keymap[key])
        logging.debug(f"action: {key}")

        if action_delay:
            time.sleep(ACTION_DELAY)

    def envAction(self, key, action_delay=False):
        if key not in self.env_keymap:
            logging.critical("invalid env action")
            raise RuntimeError()

        pydirectinput.press(self.env_keymap[key])
        logging.debug(f"env: {key}")

        if action_delay:
            time.sleep(ACTION_DELAY)
