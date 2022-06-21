import time

from env.env_config import ACTION_DELAY, AGENT_DEAD_DELAY, MAP_CENTER
from env.memory import Memory


if __name__ == "__main__":
    memory = Memory()
    time.sleep(10)

    for i in range(200):
        agent_hp, agent_ep = memory.getStatus()
        lock_state = memory.lockBoss()
        print(f"HP {agent_hp:<.2f}, EP {agent_ep:<.2f}")
        print(f"Lock State: {lock_state}")
        if i % 10 == 0:
            memory.setCritical(True)
            time.sleep(ACTION_DELAY)
            memory.setCritical(False)
        if agent_hp == 0:
            time.sleep(AGENT_DEAD_DELAY)
            memory.transportAgent(MAP_CENTER)
            lock_state = memory.lockBoss()
            memory.reviveAgent()
        time.sleep(ACTION_DELAY)
