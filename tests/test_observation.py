import logging

import win32gui
from icecream import ic

from env.env_config import GAME_NAME
from env.observation import Observer

if __name__ == "__main__":
    # ic.disable()
    logging.basicConfig(level=logging.DEBUG)

    handle = win32gui.FindWindow(0, GAME_NAME)
    if handle == 0:
        logging.critical(f"can't find {GAME_NAME}")
        raise RuntimeError()

    observer = Observer(handle)
    observer.shotScreen()
