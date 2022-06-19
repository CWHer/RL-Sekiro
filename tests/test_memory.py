import time

from env.memory import Memory


if __name__ == "__main__":
    memory = Memory()
    try:
        for i in range(200):
            agent_hp, agent_ep = memory.getStatus()
            print(f"HP {agent_hp:<.2f}, EP {agent_ep:<.2f}")
            if agent_hp == 0:
                time.sleep(10)
                memory.reviveAgent()
            time.sleep(1)
    except Exception as e:
        print(e)
        memory.restoreMemory()
        raise RuntimeError()
