GAME_NAME = "Sekiro"

ACTION_DELAY = 0.25
REVIVE_DELAY = 1

# FIXME: change accordingly
AGENT_KEYMAP = {
    "attack": "j",
    "defense": "k",
    "dodge": "shift",
    "jump": "space",
}

ENV_KEYMAP = {
    "pause": "esc",
    "resume": "esc",
    "revive": "j",
    "focus": "l",
}

# HACK: (left, top, right, bottom)
SCREEN_SIZE = (720, 1280)
SCREEN_ANCHOR = (1, -721, -1, -1)
FOCUS_ANCHOR = (400, 100, 900, 600)

AGENT_HP_ANCHOR = (75, 651, 370, 658)
BOSS_HP_ANCHOR = (75, 62, 348, 71)

AGENT_EP_ANCHOR = (645, 621, 770, 626)
BOSS_EP_ANCHOR = (645, 42, 870, 49)
