import logging

import numpy as np
import win32gui
from icecream import ic
from PIL import Image

from env.env_config import GAME_NAME
from env.observation import Observer

if __name__ == "__main__":
    # ic.disable()
    logging.basicConfig(level=logging.DEBUG)

    handle = win32gui.FindWindow(0, GAME_NAME)
    if handle == 0:
        logging.critical(f"can't find {GAME_NAME}")
        raise RuntimeError()

    observer = Observer(handle, None)
    observer.shotScreen()

    screen_shot = Image.open("./debug/screen-shot.png")
    screen_shot = np.array(screen_shot, dtype=np.int16).transpose(2, 0, 1)
    observer.state(screen_shot)
