# Sekiro Env

《只狼：影逝二度》RL训练环境

![env](README.assets/env.gif)

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



## Getting Started

1. 使用键盘进行游戏，并修改游戏按键：
   
   - 添加”J“键为”攻击“
   
   - 添加”K“键为”防御“
  
2. 使用窗口化进行游戏，并修改分辨率为`1280x720`

3. 安装自定义Mod，修改敌人的生命数为`0xff`（[下载地址](https://drive.google.com/file/d/1y9islX4yVQ0annRZCakzuuz32UMi1wVm/view?usp=sharing)）

4. 选择关卡："再战稀世强者"，”苇名弦一郎“，等待关卡载入结束后，按”esc“进行暂停

5. 开始运行环境，`state = env.reset() ...`

   - :warning: 确保`step time > 0.25s`

   - :warning: 可能需要根据自己的实际情况调整`env_config.py`中的`*_DELAY`参数



## TODO

- [x] 被击退到墙角后，锁定敌人会失效

  每次`env.reset()`时，传送人物到地图中间，修改内存锁定敌人

- [x] 生命值/耐力值识别不准确，会有小波动，偶尔有大波动

  直接从内存读取人物的生命值/耐力值，敌人的生命值仍使用模式匹配的方法

- [x] 尚未处理成功击败敌人的情况

  :warning: 在敌人生命值极低时，强制修改敌人的耐力值为0，简化彻底击败敌人的过程
  
  使用自定义的Mod修改敌人的生命数为`0xff`（credit to [Wendi Chen](https://github.com/ChenWendi2001)）




## Details

- State

  :warning: 当且仅当`agent`死亡时，`done`为`True`
  
  | Variable   | Type                               | Description              |
  | ---------- | ---------------------------------- | ------------------------ |
  | focus area | `npt.NDArray[np.uint8]` (128, 128) | 截图的中心区域，灰度图像 |
  | agent hp   | `float` [0, 1]                     | 生命值，初始为1.000      |
  | agent ep   | `float` [0, 1]                     | 耐力值，初始为1.000      |
  | boss hp    | `float` [0, 1]                     | 生命值，初始为1.000      |
  
- Actions

  包括`attack`、`defense`、`dodge`、`jump`

  ~~使用`pydirectinput`模拟按键~~，将`pydirectinput`的部分代码提取到了`keyboard.py`，加快按键速度

- Observation

  `shotScreen()`获取游戏窗口的截图，用传统的匹配方法来读取敌人的生命值

- Memory

  读取游戏内存，执行代码注入

  `restoreMemory()`撤销代码注入，在程序终止的时候必须被执行 :warning:

- Detect HP

  1. 使用`magic number`在截图中裁剪出生命值进度条

  2. 预先存储满生命值的图片`target`，模式为HSV（`np.int16`避免溢出）

  3. 对于当前生命值的图片`current`，进行如下操作

     ```python
     result: npt.NDArray[np.bool_] = np.max(
         np.abs(target - current), axis=0) < (threshold * 256)
     result = np.sum(result, axis=0) > result.shape[0] / 2
     return np.sum(result) / result.size
     ```

- Reward
  
  `death of agent`：-20，`death of boss`：50
  $$
  \begin{align}
  \text{reward} = & w_0 \times \left(\text{agent hp}-\text{last agent hp}\right) + \\
  &+ w_1 \times \min\left(0,\left(\text{agent ep}-\text{last agent ep}\right)\right) \\
  &+ w_2 \times \left(\text{last boss hp}-\text{boss hp}\right) \\
  \end{align}
  $$
  

