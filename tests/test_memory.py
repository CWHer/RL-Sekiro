import time

from env.env_config import ACTION_DELAY, AGENT_DEAD_DELAY, MAP_CENTER
from env.memory import Memory


if __name__ == "__main__":
    memory = Memory()
    time.sleep(10)

    memory.setCritical()
    for i in range(200):
        agent_hp, agent_ep, boss_hp = memory.getStatus()
        lock_state = memory.lockBoss()
        print(f"{agent_hp:<.2f}, {agent_ep:<.2f}, {boss_hp:<.2f}")
        print(f"Lock State: {lock_state}")
        if i % 50 == 0:
            memory.reviveBoss()
        if agent_hp == 0:
            time.sleep(AGENT_DEAD_DELAY)
            memory.transportAgent(MAP_CENTER)
            lock_state = memory.lockBoss()
            memory.reviveAgent(need_delay=False)
        time.sleep(ACTION_DELAY)
