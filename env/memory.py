import logging
import time
from typing import Tuple

import pymem
from pymem import Pymem

from .env_config import GAME_NAME, MAX_EP, MAX_HP


class Memory():
    def __init__(self) -> None:
        # NOTE: credit to https://fearlessrevolution.com/viewtopic.php?t=8938
        self.pm = Pymem(f"{GAME_NAME}.exe")

        """[memory scan]
        E8 ** ** ** ** 48 8B CB
        66 ** ** ** 0F ** ** E8
        ** ** ** ** 66 ** ** **
        0F ** ** F3 ** ** ** 0F
        """
        bytes_pattern = b"\xe8....\x48\x8b\xcb\x66...\x0f..\xe8" \
                        b"....\x66...\x0f..\xf3...\x0f"
        module_game = pymem.process.module_from_name(
            self.pm.process_handle, f"{GAME_NAME}.exe")
        self.health_read_addr = pymem.pattern.pattern_scan_module(
            self.pm.process_handle, module_game, bytes_pattern)
        if self.health_read_addr is None:
            logging.critical("memory scan failed")
            raise RuntimeError()
        self.original_code = self.pm.read_bytes(self.health_read_addr + 5, 7)

        """[code injection]
        address: code_addr
        mov     rbx, agent_mem_addr
        mov     [rbx], rcx
        pop     rbx
        ----> original code
        mov     rcx, rbx
        movd    xmm6, eax
        <---- original code
        mov     rbx, health_read_addr + 0xc
        jmp     rbx
        ----------
        agent_mem_addr  (code_addr + 0x21)
        dq      0
        """
        self.code_addr = self.pm.allocate(2048)
        self.agent_mem_ptr = self.code_addr + 0x21
        injected_code = \
            b"\x48\xbb" + self.agent_mem_ptr.to_bytes(8, "little") + \
            b"\x48\x89\x0b" + b"\x5b" + self.original_code + \
            b"\x48\xbb" + (self.health_read_addr + 0x1b).to_bytes(8, "little") + \
            b"\xff\xe3"
        self.pm.write_bytes(self.code_addr, injected_code, len(injected_code))

        """[code injection]
        address: health_read_addr + 0xb8
        mov     rbx, code_addr
        jump    rbx
        """
        injected_code = \
            b"\x48\xbb" + self.code_addr.to_bytes(8, "little") + \
            b"\xff\xe3"
        self.pm.write_bytes(self.health_read_addr + 0xb8,
                            injected_code, len(injected_code))

        """[change original code]
        address: health_read_addr + 5
        push    rbx
        jmp     health_read_addr + 0xb8
        nop
        """
        modified_code = b"\x53" + \
            b"\xe9" + (0xb8 - 0x6 - 5).to_bytes(4, "little") + \
            b"\x90"
        self.pm.write_bytes(self.health_read_addr + 5,
                            modified_code, len(modified_code))

        time.sleep(0.5)

    def __del__(self):
        self.restoreMemory()

    def restoreMemory(self) -> None:
        self.pm.free(self.code_addr)
        self.pm.write_bytes(self.health_read_addr + 5,
                            self.original_code, len(self.original_code))
        self.pm.write_bytes(self.health_read_addr + 0xb8,
                            b"\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xcc", 10)

    def getStatus(self) -> Tuple[float, float]:
        """[summary]

        Returns:
            Tuple[float, float]:
                agent hp    [0, 1]
                agent ep    [0, 1]
        """
        agent_mem_addr = self.pm.read_ulonglong(self.agent_mem_ptr)
        agent_hp = self.pm.read_int(agent_mem_addr + 0x130)
        agent_ep = self.pm.read_int(agent_mem_addr + 0x148)
        return (agent_hp / MAX_HP, agent_ep / MAX_EP)

    def reviveAgent(self) -> None:
        agent_mem_addr = self.pm.read_ulonglong(self.agent_mem_ptr)
        self.pm.write_int(agent_mem_addr + 0x130, MAX_HP)