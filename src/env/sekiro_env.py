import logging
import time
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
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

    def actionSpace(self) -> List[int]:
        return list(range(len(AGENT_KEYMAP)))

    def __stepReward(self, state: Tuple) -> float:
        agent_hp, boss_hp, agent_ep, boss_ep = state[1:]
        # TODO: refine reward
        rewards = np.array(
            [agent_hp - self.last_agent_hp,
             self.last_boss_hp - boss_hp,
             min(0, self.last_agent_ep - agent_ep),
             max(0, boss_ep - self.last_boss_ep)])
        weights = np.array([0.5, 0.4, 0.1, 0.1])
        reward = weights.dot(rewards).item()

        reward = -100 if agent_hp < 0.1 else reward

        self.last_agent_hp = agent_hp
        self.last_agent_ep = agent_ep
        self.last_boss_hp = boss_hp
        self.last_boss_ep = boss_ep

        logging.info(f"reward: {reward:<.2f}")
        return reward

    @timeLog
    def step(self, action: int) -> Tuple[Tuple[npt.NDArray[np.uint8],
                                               float, float, float, float],
                                         float, bool, None]:
        """[summary]

        State:
            focus_area      npt.NDArray[np.uint8], "L"
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
        action_key = list(AGENT_KEYMAP.keys())[action]
        self.actor.agentAction(action_key)

        screen_shot = self.observer.shotScreen()
        state = self.observer.state(screen_shot)

        done = state[1] < 0.1
        if done:
            time.sleep(10)
            self.actor.envAction("focus", action_delay=REVIVE_DELAY)
            self.actor.envAction("revive", action_delay=REVIVE_DELAY * 1.5)
            self.actor.envAction("pause", action_delay=REVIVE_DELAY)

        if state[2] == 0:
            # TODO: succeed
            raise NotImplementedError()

        return state, self.__stepReward(state), done, None

    def reset(self) -> Tuple[npt.NDArray[np.uint8],
                             float, float, float, float]:
        # restore window
        win32gui.SendMessage(self.handle, win32con.WM_SYSCOMMAND,
                             win32con.SC_RESTORE, 0)
        # focus on window
        win32gui.SetForegroundWindow(self.handle)
        time.sleep(0.5)

        self.actor.envAction("resume", action_delay=True)

        screen_shot = self.observer.shotScreen()
        state = self.observer.state(screen_shot)
        self.last_agent_hp, self.last_boss_hp, \
            self.last_agent_ep, self.last_boss_ep = state[1:]

        return state
