import ctypes
import logging

import numpy as np
import numpy.typing as npt
from icecream import ic
from PIL import ImageGrab
from utils import timeLog


class Observer():
    """[summary]
    yield raw observation
    """

    def __init__(self, handle) -> None:
        self.handle: int = handle

        anchor = ctypes.wintypes.RECT()
        ctypes.windll.user32.SetProcessDPIAware(2)
        DMWA_EXTENDED_FRAME_BOUNDS = 9
        ctypes.windll.dwmapi.DwmGetWindowAttribute(
            ctypes.wintypes.HWND(self.handle),
            ctypes.wintypes.DWORD(DMWA_EXTENDED_FRAME_BOUNDS),
            ctypes.byref(anchor), ctypes.sizeof(anchor))
        self.anchor = (anchor.left, anchor.top, anchor.right, anchor.bottom)
        logging.debug(anchor)

    # @timeLog
    def shotScreen(self) -> npt.NDArray[np.uint8]:
        screen_shot = ImageGrab.grab(self.anchor)
        if ic.enabled:
            from datetime import datetime
            timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
            screen_shot.save(f"screen-shot-{timestamp}.png")

        return np.array(screen_shot, dtype=np.uint8)
