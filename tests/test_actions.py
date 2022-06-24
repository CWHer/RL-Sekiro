import logging
import time

import win32con
import win32gui
from icecream import ic

from env.actions import Actor
from env.env_config import ACTION_DELAY, GAME_NAME, PAUSE_DELAY


if __name__ == "__main__":
    # ic.disable()
    logging.basicConfig(level=logging.DEBUG)

    handle = win32gui.FindWindow(0, GAME_NAME)
    if handle == 0:
        logging.critical(f"can't find {GAME_NAME}")
        raise RuntimeError()

    # restore window
    win32gui.SendMessage(handle, win32con.WM_SYSCOMMAND,
                         win32con.SC_RESTORE, 0)
    # focus on window
    win32gui.SetForegroundWindow(handle)
    time.sleep(0.5)

    actor = Actor()

    actor.envAction("resume", action_delay=PAUSE_DELAY)

    for _ in range(2):
        for key in ["attack", "defense", "jump",
                    "forward_dodge", "backward_dodge",
                    "leftward_dodge", "rightward_dodge"]:
            actor.agentAction(key, action_delay=ACTION_DELAY[key])

    actor.envAction("pause")
