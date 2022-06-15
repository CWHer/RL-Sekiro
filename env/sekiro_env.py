import logging
import time
from typing import List, Tuple

import win32con
import win32gui
from utils import timeLog

from .actions import Actor
from .env_config import AGENT_KEYMAP, GAME_NAME, REVIVE_DELAY
from .observation import Observer


class SekiroEnv():
    def __init__(self) -> None:
        self.handle = win32gui.FindWindow(0, GAME_NAME)
        if self.handle == 0:
            logging.critical(f"can't find {GAME_NAME}")
            raise RuntimeError()

        self.actor = Actor(self.handle)
        self.observer = Observer(self.handle)

        self.last_agent_hp = 0
        self.last_agent_ep = 0
        self.last_boss_hp = 0
        self.last_boss_ep = 0

    def actionSpace(self) -> List[str]:
        return list(AGENT_KEYMAP.keys())

    def __stepReward(self, state: Tuple) -> float:
        agent_hp, boss_hp, agent_ep, boss_ep = state[1:]
        # TODO
        reward = 0

        self.last_agent_hp = agent_hp
        self.last_agent_ep = agent_ep
        self.last_boss_hp = boss_hp
        self.last_boss_ep = boss_ep

        return reward

    @timeLog
    def step(self, action) -> Tuple[Tuple, float, bool, None]:
        """[summary]

        State:
            image           npt.NDArray[np.uint8]
            agent_hp        float
            boss_hp         float
            agent_ep        float
            boss_ep         float

        Returns:
            state           Tuple
            reward          float
            done            bool
            info            None
        """
        self.actor.agentAction(action)

        screen_shot = self.observer.shotScreen()
        state = self.observer.state(screen_shot)

        done = state[1] == 0
        if done:
            time.sleep(10)
            self.actor.envAction(
                "revive", action_delay=REVIVE_DELAY)
            self.actor.envAction("pause")

        return state, self.__stepReward(state), done, None

    def reset(self) -> Tuple:
        # restore window
        win32gui.SendMessage(self.handle, win32con.WM_SYSCOMMAND,
                             win32con.SC_RESTORE, 0)
        # focus on window
        win32gui.SetForegroundWindow(self.handle)
        time.sleep(0.5)

        self.actor.envAction("resume", action_delay=True)
        self.actor.envAction("focus")

        screen_shot = self.observer.shotScreen()
        state = self.observer.state(screen_shot)
        self.last_agent_hp, self.last_boss_hp, \
            self.last_agent_ep, self.last_boss_ep = state[1:]

        return state
