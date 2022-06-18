import logging
import time

from utils import timeLog

from .env_config import AGENT_KEYMAP, ENV_KEYMAP, PRESS_RELEASE_DELAY
from .keyboard import PressKey, ReleaseKey


class Actor():
    def __init__(self, handle) -> None:
        self.handle = handle
        self.agent_keymap = AGENT_KEYMAP
        self.env_keymap = ENV_KEYMAP

    @timeLog
    def agentAction(self, key,
                    action_delay: float = 0):
        if key not in self.agent_keymap:
            logging.critical("invalid agent action")
            raise RuntimeError()

        key_code = self.agent_keymap[key]
        PressKey(key_code)
        time.sleep(PRESS_RELEASE_DELAY)
        ReleaseKey(key_code)
        logging.info(f"action: {key}")

        time.sleep(action_delay)

    @timeLog
    def envAction(self, key,
                  action_delay: float = 0):
        if key not in self.env_keymap:
            logging.critical("invalid env action")
            raise RuntimeError()

        key_code = self.env_keymap[key]
        PressKey(key_code)
        time.sleep(PRESS_RELEASE_DELAY)
        ReleaseKey(key_code)
        logging.debug(f"env: {key}")

        time.sleep(action_delay)
