GAME_NAME = "Sekiro"

# ------> action
# HACK: change accordingly
PRESS_RELEASE_DELAY = 0.02

STEP_DELAY = 0.15
ACTION_DELAY = 0.25

AGENT_DEAD_DELAY = 10
ROTATION_DELAY = 1
REVIVE_DELAY = 2.2
PAUSE_DELAY = 0.8

# NOTE: directX scan codes https://www.google.com/search?q=directInputKeyboardScanCodes
AGENT_KEYMAP = {
    "attack": 0x24,
    "defense": 0x25,
    "dodge": 0x2A,
    "jump": 0x39,
}

ENV_KEYMAP = {
    "pause": 0x01,
    "resume": 0x01,
}
# <------

# ------> code injection
MIN_CODE_LEN = 6
MIN_HELPER_LEN = 13
# <------

# ------> agent attributes
MAX_AGENT_HP = 800
MAX_AGENT_EP = 300

MAX_BOSS_HP = 9887
MAX_AGENT_EP = 4083

MAP_CENTER = (-110.252, 54.077, 239.538)
# <------

# ------> screenshot
# HACK: (left, top, right, bottom)
SCREEN_SIZE = (720, 1280)
SCREEN_ANCHOR = (1, -721, -1, -1)

FOCUS_ANCHOR = (392, 108, 892, 608)
FOCUS_SIZE = (128, 128)

# BOSS_HP_ANCHOR = (75, 62, 348, 71)
# <------
