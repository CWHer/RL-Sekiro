import time

from env.env_config import AGENT_DEAD_DELAY
from env.memory import Memory


if __name__ == "__main__":
    memory = Memory()
    time.sleep(20)
    for i in range(200):
        agent_hp, agent_ep = memory.getStatus()
        lock_state = memory.lockBoss()
        print(f"HP {agent_hp:<.2f}, EP {agent_ep:<.2f}")
        print(f"Lock State: {lock_state}")
        if agent_hp == 0:
            time.sleep(AGENT_DEAD_DELAY)
            lock_state = memory.lockBoss()
            memory.reviveAgent()
        time.sleep(1)
