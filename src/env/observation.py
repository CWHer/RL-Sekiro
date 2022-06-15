import ctypes
import logging
import pickle
from datetime import datetime
from typing import Tuple

import numpy as np
import numpy.typing as npt
from icecream import ic
from PIL import Image, ImageGrab
from utils import timeLog

from .env_config import (AGENT_EP_ANCHOR, AGENT_HP_ANCHOR, BOSS_EP_ANCHOR,
                         BOSS_HP_ANCHOR, SCREEN_ANCHOR, SCREEN_SIZE)


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

        self.timestamp: str = ""

        # HACK: load preset hp & ep
        self.agent_hp_full = pickle.load(
            open("./env/asset/agent-hp-full.pkl", "rb"))
        self.boss_hp_full = pickle.load(
            open("./env/asset/boss-hp-full.pkl", "rb"))

    def __select(self, arr: npt.NDArray, anchor: Tuple) -> npt.NDArray:
        # NOTE: C x H x W, "RGB"
        left, top, right, bottom = anchor
        return arr[:, top:bottom, left:right]

    # @timeLog
    def shotScreen(self) -> npt.NDArray[np.uint8]:
        screen_shot = ImageGrab.grab(self.anchor)
        # NOTE: C x H x W, "RGB"
        screen_shot = np.array(screen_shot, dtype=np.uint8).transpose(2, 0, 1)
        screen_shot = self.__select(screen_shot, SCREEN_ANCHOR)

        if ic.enabled:
            self.timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
            Image.fromarray(
                screen_shot.transpose(1, 2, 0)).save(
                f"./debug/screen-shot-{self.timestamp}.png")

        if screen_shot.shape[1:] != SCREEN_SIZE:
            logging.critical("incorrect screenshot")
            raise RuntimeError()

        return screen_shot

    def __calcProperty(self, arr: npt.NDArray[np.uint8],
                       target: npt.NDArray[np.uint8], prefix="") -> float:
        if ic.enabled:
            Image.fromarray(
                arr.transpose(1, 2, 0)).save(
                f"./debug/{prefix}-{self.timestamp}.png")
        if arr.shape != target.shape:
            logging.critical("incorrect arr shape")
            raise RuntimeError()

        result: npt.NDArray[np.bool_] = np.all(arr == target, axis=0)
        if ic.enabled:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots()
            ax.spy(result)
            plt.savefig(f"./debug/{prefix}-bool-{self.timestamp}.png")
            plt.close()
        result = np.sum(result, axis=0) > result.shape[0] / 2

        return 100 * np.sum(result) / result.size

    @timeLog
    def state(self, screen_shot: npt.NDArray[np.uint8]) -> \
            Tuple[npt.NDArray[np.uint8], float, float, float, float]:
        """[summary]

        State:
            image           npt.NDArray[np.uint8]
            agent_hp        float
            boss_hp         float
            agent_ep        float
            boss_ep         float
        """
        agent_hp = self.__calcProperty(
            arr=self.__select(screen_shot, AGENT_HP_ANCHOR),
            target=self.agent_hp_full, prefix="agent-hp")
        boss_hp = self.__calcProperty(
            arr=self.__select(screen_shot, BOSS_HP_ANCHOR),
            target=self.boss_hp_full, prefix="boss-hp")
        logging.info(f"agent hp: {agent_hp:.1f}, boss hp: {boss_hp:.1f}")

        # agent_ep = self.__calcProperty(
        #     self.__select(screen_shot, AGENT_EP_ANCHOR),
        #     target=self.agent_ep_full, prefix="agent-ep")
        # boss_ep = self.__calcProperty(
        #     self.__select(screen_shot, BOSS_EP_ANCHOR),
        #     target=self.boss_ep_full, prefix="boss-ep")
        # logging.info(f"agent ep: {agent_ep:.1f}, boss ep: {boss_ep:.1f}")

        # TODO

        return screen_shot, agent_hp, boss_hp, 0, 0
        # return screen_shot, agent_hp, boss_hp, agent_ep, boss_ep
