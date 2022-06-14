import logging
import time
from typing import List, Tuple

import numpy.typing as npt
import win32con
import win32gui
from utils import timeLog

from .actions import Actor
from .env_config import AGENT_KEYMAP, GAME_NAME
from .observation import Observer


class SekiroEnv():
    def __init__(self) -> None:
        self.handle = win32gui.FindWindow(0, GAME_NAME)
        if self.handle == 0:
            logging.critical(f"can't find {GAME_NAME}")
            raise RuntimeError()

        self.actor = Actor(self.handle)
        self.observer = Observer(self.handle)

    def actionSpace(self) -> List[str]:
        return list(AGENT_KEYMAP.keys())

    def getScore(self) -> float:
        # TODO
        raise NotImplementedError()

    @timeLog
    def step(self, action) -> Tuple[List[npt.NDArray], float, bool, None]:
        """[summary]

        Returns:
            state       List[npt.NDArray]
            reward      float
            done        bool
            info        None
        """
        # TODO
        self.actor.agentAction(action)
        return [self.observer.shotScreen()], 0, False, None

    def reset(self) -> List[npt.NDArray]:
        # TODO

        # restore window
        win32gui.SendMessage(
            self.handle, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
        # focus on window
        win32gui.SetForegroundWindow(self.handle)
        time.sleep(0.5)

        self.actor.envAction("resume", action_delay=True)
        self.actor.envAction("focus")

        return [self.observer.shotScreen()]
