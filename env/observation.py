import ctypes
import logging
from datetime import datetime
from typing import Tuple

import numpy as np
import numpy.typing as npt
from icecream import ic
from PIL import Image, ImageGrab
from utils import timeLog

from .env_config import FOCUS_ANCHOR, FOCUS_SIZE, SCREEN_ANCHOR, SCREEN_SIZE
from .memory import Memory


class Observer():
    """[summary]
    yield raw observation
    """

    def __init__(self, handle: int, memory: Memory) -> None:
        self.handle = handle
        self.memory = memory

        anchor = ctypes.wintypes.RECT()
        ctypes.windll.user32.SetProcessDPIAware(2)
        DMWA_EXTENDED_FRAME_BOUNDS = 9
        ctypes.windll.dwmapi.DwmGetWindowAttribute(
            ctypes.wintypes.HWND(self.handle),
            ctypes.wintypes.DWORD(DMWA_EXTENDED_FRAME_BOUNDS),
            ctypes.byref(anchor), ctypes.sizeof(anchor))
        self.anchor = (anchor.left, anchor.top, anchor.right, anchor.bottom)
        logging.debug(anchor)

        self.timestamp: str = ""

        # # HACK: load preset hp
        # self.boss_hp_full = pickle.load(
        #     open("./env/asset/boss-hp-full.pkl", "rb"))

    def __select(self, arr: npt.NDArray, anchor: Tuple) -> npt.NDArray:
        # NOTE: C x H x W
        left, top, right, bottom = anchor
        return arr[:, top:bottom, left:right]

    # @timeLog
    def shotScreen(self) -> npt.NDArray[np.int16]:
        screen_shot = ImageGrab.grab(self.anchor)
        # NOTE: C x H x W, "RGB"
        screen_shot = np.array(screen_shot, dtype=np.int16).transpose(2, 0, 1)
        screen_shot = self.__select(screen_shot, SCREEN_ANCHOR)

        if ic.enabled:
            self.timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
            Image.fromarray(
                screen_shot.transpose(1, 2, 0).astype(np.uint8)).save(
                f"./debug/screen-shot-{self.timestamp}.png")

        if screen_shot.shape[-2:] != SCREEN_SIZE:
            logging.critical("incorrect screenshot")
            raise RuntimeError()

        return screen_shot

    # def __calcProperty(self, arr: npt.NDArray[np.int16],
    #                    target: npt.NDArray[np.int16], threshold, prefix="") -> float:
    #     """[summary]

    #     Args:
    #         arr (npt.NDArray[np.int16]): C x H x W
    #         target (npt.NDArray[np.int16]): C x H x W
    #     """
    #     if ic.enabled:
    #         Image.fromarray(
    #             arr.transpose(1, 2, 0).astype(np.uint8), mode="HSV").convert(
    #                 "RGB").save(f"./debug/{prefix}-{self.timestamp}.png")
    #     if arr.shape != target.shape:
    #         logging.critical("incorrect arr shape")
    #         raise RuntimeError()

    #     result: npt.NDArray[np.bool_] = np.max(
    #         np.abs(target - arr), axis=0) < (threshold * 256)
    #     if ic.enabled:
    #         import matplotlib.pyplot as plt
    #         fig, ax = plt.subplots(2, 1)
    #         ax[0].spy(result)
    #         ax[1].imshow(Image.fromarray(
    #             arr.transpose(1, 2, 0).astype(np.uint8), mode="HSV").convert("RGB"))
    #         fig.subplots_adjust(hspace=-0.8)
    #         plt.savefig(f"./debug/{prefix}-content-{self.timestamp}.png")
    #         plt.close()

    #     result = np.sum(result, axis=0) > result.shape[0] / 2
    #     return np.sum(result) / result.size

    @timeLog
    def state(self, screen_shot: npt.NDArray[np.int16]) -> \
            Tuple[npt.NDArray[np.uint8], float, float, float]:
        """[summary]

        State:
            image           npt.NDArray[np.uint8]
            agent_hp        float, [0, 1]
            agent_ep        float, [0, 1]
            boss_hp         float, [0, 1]
        """
        agent_hp, agent_ep, boss_hp = self.memory.getStatus()

        # # NOTE: use HSV
        # hsv_screen_shot = np.array(Image.fromarray(
        #     screen_shot.astype(np.uint8).transpose(1, 2, 0)).convert("HSV"),
        #     dtype=np.int16).transpose(2, 0, 1)
        # boss_hp = self.__calcProperty(
        #     arr=self.__select(hsv_screen_shot, BOSS_HP_ANCHOR),
        #     target=self.boss_hp_full, threshold=0.4, prefix="boss-hp")

        focus_area = Image.fromarray(self.__select(
            screen_shot, FOCUS_ANCHOR).transpose(1, 2, 0).astype(np.uint8)).convert("L")
        if ic.enabled:
            focus_area.save(f"./debug/focus-{self.timestamp}.png")
        focus_area = np.array(
            focus_area.resize(FOCUS_SIZE), dtype=np.uint8)

        return focus_area, agent_hp, agent_ep, boss_hp
