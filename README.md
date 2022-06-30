# Sekiro Env

《只狼：影逝二度》RL训练环境


```python
import logging
import random

from icecream import ic

from env import SekiroEnv

if __name__ == "__main__":
    ic.disable()
    logging.basicConfig(level=logging.INFO)

    env = SekiroEnv()

    for _ in range(20):
        done = False
        state = env.reset()
        while not done:
            action = random.choice(env.actionSpace())
            state, reward, done, _ = env.step(action)
```

## 游戏版本要求

:warning: 本训练环境支持《只狼：影逝二度》`v1.06`，请确认本地游戏版本



## Getting Started

1. 使用键盘进行游戏，并修改游戏按键：

   - 添加”J“键为”攻击“

   - 添加”K“键为”防御“

2. 调整游戏内设置

   - 使用窗口化进行游戏，修改分辨率为`1280x720`，将”质量设定“调整为中
   
   - 关闭”血腥效果“以及”字幕显示“，设置”亮度调整“为10

3. 选择关卡："再战稀世强者"，”苇名弦一郎“，等待关卡载入结束后，按”esc“进行暂停

4. 开始运行环境，`state = env.reset() ...`

   :warning: 可能需要根据自己的实际情况调整`env_config.py`中的`*_DELAY`参数



## TODO

- [x] 被击退到墙角后，锁定敌人会失效

  每次`env.reset()`时，传送人物到地图中间，修改内存锁定敌人

- [x] 生命值/耐力值识别不准确，会有小波动，偶尔有大波动

  直接从内存读取人物的生命值/耐力值以及敌人的生命值

- [x] 尚未处理成功击败敌人的情况

  :warning: 在敌人生命值极低时，直接复活敌人，并认为成功击败了敌人

- [x] 敌人攻击力过高，负样本占比过高

  :warning: 修改人物/敌人的攻击力




## Details

- State

  :warning: 当且仅当`agent`死亡时，`done`为`True`
  
  | Variable   | Type                               | Description              |
  | ---------- | ---------------------------------- | ------------------------ |
  | focus area | `npt.NDArray[np.uint8]` (128, 128) | 截图的中心区域，RGB图像 |
  | agent hp   | `float` [0, 1]                     | 生命值，初始为1.000      |
  | agent ep   | `float` [0, 1]                     | 耐力值，初始为1.000      |
  | boss hp    | `float` [0, 1]                     | 生命值，初始为1.000      |
  
- Actions

  包括`attack`、`defense`、`jump`、`forward dodge`、`backward dodge`、`leftward dodge`、`rightward dodge`

  ~~使用`pydirectinput`模拟按键~~，将`pydirectinput`的部分代码提取到了`keyboard.py`，加快按键速度

- Observation

  `shotScreen()`获取游戏窗口的截图

- Memory

  读取游戏内存，执行代码注入，获取人物的生命值/耐力值以及敌人的生命值, 修改人物/敌人的攻击力

  `restoreMemory()`撤销代码注入，在程序终止的时候必须被执行 :warning:

- Reward
  
  $$
  \begin{align}
  \text{reward} = & w_0 \times \left(\text{agent hp}-\text{last agent hp}\right) \\
  &+ w_1 \times \left(\text{last boss hp}-\text{boss hp}\right) \\
  \end{align}
  $$
  

