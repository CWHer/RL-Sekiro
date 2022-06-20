import logging
import time
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import win32con
import win32gui
from utils import timeLog

from .actions import Actor
from .env_config import (ACTION_DELAY, AGENT_DEAD_DELAY, AGENT_KEYMAP,
                         BOSS_DEAD_DELAY, GAME_NAME, MAP_CENTER, PAUSE_DELAY,
                         STEP_DELAY)
from .memory import Memory
from .observation import Observer


class SekiroEnv():
    def __init__(self) -> None:
        self.handle = win32gui.FindWindow(0, GAME_NAME)
        if self.handle == 0:
            logging.critical(f"can't find {GAME_NAME}")
            raise RuntimeError()

        self.actor = Actor()
        self.memory = Memory()
        self.observer = Observer(self.handle, self.memory)

        self.last_agent_hp = 0
        self.last_agent_ep = 0
        self.last_boss_hp = 0

    def actionSpace(self) -> List[int]:
        return list(range(len(AGENT_KEYMAP)))

    def __stepReward(self, agent_hp, agent_ep, boss_hp) -> float:
        # TODO: refine reward
        rewards = np.array(
            [agent_hp - self.last_agent_hp,
             min(0, agent_ep - self.last_agent_ep),
             self.last_boss_hp - boss_hp])
        weights = np.array([2, 1, 5])
        reward = weights.dot(rewards).item()

        reward = -20 if agent_hp == 0 else reward
        reward = 50 if boss_hp == 0 else reward

        logging.info(f"reward: {reward:<.2f}")
        return reward

    @timeLog
    def step(self, action: int) -> Tuple[Tuple[npt.NDArray[np.uint8],
                                               float, float, float],
                                         float, bool, None]:
        """[summary]

        State:
            focus_area      npt.NDArray[np.uint8], "L"
            agent_hp        float, [0, 1]
            agent_ep        float, [0, 1]
            boss_hp         float, [0, 1]

        Returns:
            state           Tuple
            reward          float
            done            bool
            info            None
        """
        lock_state = self.memory.lockBoss()
        logging.info(f"lock state: {lock_state}")

        action_key = list(AGENT_KEYMAP.keys())[action]
        self.actor.agentAction(action_key, action_delay=STEP_DELAY)

        screen_shot = self.observer.shotScreen()
        state = self.observer.state(screen_shot)

        agent_hp, agent_ep, boss_hp = state[-3:]
        reward = self.__stepReward(agent_hp, agent_ep, boss_hp)

        # update last status
        self.last_agent_hp = agent_hp
        self.last_agent_ep = agent_ep
        self.last_boss_hp = boss_hp

        # NOTE: agent dead
        done = agent_hp == 0
        if done:
            time.sleep(10)
            self.actor.envAction("focus", action_delay=REVIVE_DELAY)
            self.actor.envAction("revive", action_delay=1.5 * REVIVE_DELAY)
            self.actor.envAction("pause", action_delay=2 * REVIVE_DELAY)

        # NOTE: boss dead
        if state[2] < 0.1:
            raise NotImplementedError()

        return state, reward, done, None

    def reset(self) -> Tuple[npt.NDArray[np.uint8], float, float, float]:
        # restore window
        win32gui.SendMessage(self.handle, win32con.WM_SYSCOMMAND,
                             win32con.SC_RESTORE, 0)
        # focus on window
        win32gui.SetForegroundWindow(self.handle)
        time.sleep(0.5)

        self.memory.transportAgent(MAP_CENTER)
        self.memory.lockBoss()
        self.actor.envAction("resume", action_delay=PAUSE_DELAY)

        screen_shot = self.observer.shotScreen()
        state = self.observer.state(screen_shot)
        self.last_agent_hp, self.last_agent_ep, self.last_boss_hp = state[-3:]

        return state
