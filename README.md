# Sekiro Env

《只狼：影逝二度》RL训练环境

![env](README.assets/env.gif)

```python
from env import SekiroEnv

env = SekiroEnv()
done = False
state = env.reset()
while not done:
    action = random.choice(env.actionSpace())
    state, reward, done, _ = env.step(action)
```



## Getting Started

1. 使用键盘进行游戏，并修改游戏按键：
   - 添加”L“键为”重置视角/固定目标“
   - 添加”J“键为”攻击“
   - 添加”K“键为”防御“
2. 使用窗口化进行游戏，并修改分辨率为`1280x720`
3. 选择关卡："再战稀世强者"，”苇名弦一郎“，等待关卡载入结束后，先**锁定**敌人，再按”esc“进行暂停
4. 开始运行环境，`state = env.reset() ...`



## TODO

- [ ] 被击退到墙角后，锁定敌人会失效

- [ ] 生命值/耐力值识别不准确，会有小波动，偶尔有大波动

- [ ] 尚未处理成功击败敌人的情况



## Details

- State

  | Variable        | Type                    | Description              |
  | --------------- | ----------------------- | ------------------------ |
  | focus area      | `npt.NDArray[np.uint8]` | 截图的中心区域，灰度图像 |
  | agent hp        | `float`                 | 生命值                   |
  | boss hp         | `float`                 |                          |
  | agent endurance | `float`                 | 耐力值                   |
  | boss endurance  | `float`                 |                          |

- Actions

  使用`pydirectinput`模拟按键

- Observation

  `shotScreen()`游戏程序的窗口截图

- Detect HP/endurance

  1. 使用magic number在截图中裁剪出生命值/耐力值进度条

  2. 预先存储满生命值/耐力值的图片`target`，模式为HSV（`np.int16`）

  3. 对于当前的生命值/耐力值的图片`current`，进行如下操作

     ```python
     result: npt.NDArray[np.bool_] = np.max(
         np.abs(target - arr), axis=0) < (threshold * 256)
     return 100 * np.sum(result) / result.size
     ```

  4. 调整`threshold`

- Reward
  
  $$
  \begin{align}
  \text{reward} = & w_0 \times \left(\text{agent hp}-\text{last agent hp}\right) + \\
  &+ w_1 \times \left(\text{last boss hp}-\text{boss hp}\right) \\
  &+ w_2 \times \min\left(0,\left(\text{last agent ep}-\text{agent ep}\right)\right) \\
  &+ w_3 \times \max\left(0,\left(\text{boss ep}-\text{last boss ep}\right)\right) \\
  \end{align}
  $$
  

