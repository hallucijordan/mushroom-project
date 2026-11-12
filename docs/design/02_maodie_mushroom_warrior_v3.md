# 🍄⚔️ Maodie — 蘑菇武士角色设计 Midjourney Prompt 完整手册 v3

> **角色重塑：** 从"Professor Whiskers 教授猫"变为 **Maodie（猫蝶）** —— 一只严肃到令人发笑的年轻猫咪武士。他的父亲因误食毒蘑菇去世，他发誓要成为世界上最强的蘑菇武士，用武道守护所有人远离毒菇之害。
>
> **v3 更新：** 修正 --oref 与 Niji 7 不兼容问题，全部改为 --cref / --cw，参数说明同步更新。

---

## 目录

1. [角色档案](#角色档案)
2. [Midjourney 参数速查表](#midjourney-参数速查表)
3. [一致性策略](#一致性策略)
4. [基准图（已锁定）](#基准图已锁定)
5. [角色状态 × 情绪 Prompt](#角色状态--情绪-prompt)
6. [等级外观进化 Prompt](#等级外观进化-prompt)
7. [场景 & 背景 Prompt](#场景--背景-prompt)
8. [配件 & 道具 Prompt](#配件--道具-prompt)
9. [UI 元素 & 特效 Prompt](#ui-元素--特效-prompt)
10. [Asset 清单总表](#asset-清单总表)
11. [对话系统模板](#对话系统模板)
12. [动画状态机](#动画状态机)
13. [生成优先级](#生成优先级)

---

## 角色档案

| 属性 | 值 |
|---|---|
| **名字** | Maodie（猫蝶） — 用户可重命名 |
| **身份** | 年轻猫咪武士，蘑菇之道修行者 |
| **背景故事** | 父亲是一位传奇蘑菇猎人，某日误食极毒鹅膏菌不治身亡。年幼的 Maodie 握着父亲留下的木刀，在坟前发下宏誓：「我要成为这个世界上最牛逼的蘑菇武士！」此后踏上修行之路，立志以武道鉴别天下蘑菇，让悲剧不再重演 |
| **种族** | 橘色虎斑猫 |
| **标志配饰** | 头戴蘑菇形斗笠（红色伞帽白色斑点，像毒蝇伞），腰挎一柄腰带上有蘑菇纹样的木刀 |
| **性格** | 极度严肃认真，但因为太认真了反而很搞笑；动作夸张，经常用武士礼仪对待蘑菇；失败时极其戏剧化地倒地 |
| **视觉风格** | 游戏像素风（16-bit pixel art），Niji 7 渲染，粗像素轮廓，有限调色板，复古游戏感 |
| **尺寸** | ~200×200px 头像（响应式缩放） |
| **设计哲学** | 像素版的武士猫 × 搞笑严肃反差萌。想象一只猫穿着武士装备，用对待生死大敌的态度去嗅一朵蘑菇。越认真越好笑 |

---

## Midjourney 参数速查表

> 本项目已切换至 **Niji 7**（2026年1月上线），基于已锁定基准图生成所有后续素材。

### 核心参数

| 参数 | 语法 | 说明 | 本项目推荐值 |
|---|---|---|---|
| **版本** | `--niji 7` | Niji 7 为本项目使用版本，动漫/插画风格，细节更清晰 | `--niji 7` |
| **宽高比** | `--ar W:H` | 控制输出图像比例 | `--ar 1:1`（头像/精灵图）、`--ar 4:1`（精灵序列帧） |
| **风格化** | `--s N` | 0-1000，值越高 MJ 自由度越高 | `--s 20`（低值保持像素风格精确） |
| **混沌度** | `--c N` | 0-100，控制四张图之间的变化程度 | `--c 3`（低值确保一致性） |
| **质量** | `--q N` | 渲染耗时，1/2/4 | `--q 1`（标准质量） |
| **排除** | `--no [元素]` | 排除不想出现的元素 | `--no text, watermark, realistic, 3D, gradient, shadow, anti-aliasing, smooth shading, background, scenery` |
| **重复** | `--r N` | 同一 prompt 运行 N 次 | `--r 4`（批量生成挑选最佳） |
| **种子** | `--seed N` | 固定随机噪声起点 | 找到好图后锁定 seed 进行微调 |
| **Draft 模式** | `--draft` | 10倍快速，半价，适合快速迭代 | 初始探索阶段使用（注意：与 --cref 不兼容） |
| **Raw 风格** | `--style raw` | 更贴近 prompt 字面意思，减少 MJ 自行美化 | Niji 7 支持，本项目使用 |

### 一致性参数（关键！）

| 参数 | 语法 | 说明 | 本项目用法 |
|---|---|---|---|
| **Character Reference** | `--cref [URL]` | Niji 7 专用角色一致性参数（V7 用 --oref，Niji 7 用 --cref） | 所有情绪 prompt 必加基准图URL |
| **Character Weight** | `--cw N` | 0-100（默认100），控制 cref 参考强度 | `--cw 100` |
| **Style Reference** | `--sref [URL]` | 复制参考图的艺术风格/色调 | 与 --cref 使用同一张基准图 |
| **Style Weight** | `--sw N` | 控制 sref 强度 | `--sw 100` |

### ⚠️ 重要限制

| 限制 | 说明 |
|---|---|
| **--cref 与 --q 4 不兼容** | 使用 --cref 时最高只能用 --q 2，不能用 --q 4 |
| **--cref 与 --draft 不兼容** | 探索阶段用 --draft 时不能同时用 --cref，先探索再锁定 |
| **--cref GPU成本** | 每张使用 --cref 的图，GPU消耗是普通图的2倍，54张素材预算需翻倍计算 |
| **Niji 7 角色一致性** | Niji 7 使用 --cref，V7 使用 --oref，两者不可混用 |

---

## 一致性策略

### ✅ 基准图已锁定

基准图 URL：
```
https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png
```

> ⚠️ Discord CDN 链接有时效性，如链接失效请重新上传图片获取新链接。

### 固定后缀（可存为 /prefer option set）

```
--niji 7 --s 20 --c 3 --style raw --ar 1:1 --cref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --cw 100 --sref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --sw 100 --no text, watermark, realistic, 3D, gradient, anti-aliasing, smooth shading, shadow, background, scenery, four-legged, crawling, animal pose
```

### 省时技巧：存为快捷命令

在 Discord 输入：
```
/prefer option set name: maodie
```
value 填入上方固定后缀，之后每条 prompt 末尾只需加 `--maodie` 即可自动展开。

---

## 基准图（已锁定）

> 基准图已生成完毕，以下为存档记录。

### 基准图描述

橘色虎斑猫武士 Maodie，拟人直立站姿，头戴红白波点毒蝇伞形斗笠，身着深蓝色道袍，腰带处有蘑菇纹样，腰挎木刀。圆滚滚体型，表情极度严肃，白色胸毛，短蓬松橘色尾巴自然下垂。纯白背景，无阴影，16-bit 像素风格，Niji 7 渲染。

### 基准图生成 Prompt（存档参考）

```
16-bit pixel art character sheet, retro game sprite style, limited color palette of warm oranges, cream white, deep red, and dark blue. Visible individual pixels, crisp pixel edges, clean hard outlines, no anti-aliasing, no smooth shading, chunky pixel blocks, retro 16-bit video game sprite aesthetic.

Subject: a young chubby orange tabby cat samurai warrior, anthropomorphic cat character, standing upright on two legs like a human, bipedal stance. Round fluffy face with big expressive eyes, tiny furrowed brows, small frown mouth, absurdly deadpan serious expression, comedically over-serious. White fluffy chest fur.

Wearing: a large red-and-white-spotted amanita mushroom kasa hat on head, a dark blue samurai gi robe tied with a rope belt, a tiny wooden katana strapped to the waist with a small mushroom symbol on the belt.

Short puffy orange tabby tail drooping down. Arms at sides, chest slightly puffed, dignified samurai stance, full body visible from head to toe.

Pure white background #FFFFFF, no shadow, no ground shadow, no drop shadow, no cast shadow, isolated character only, transparent PNG style, flat background.

--niji 7 --s 20 --c 3 --style raw --ar 1:1 --no text, watermark, realistic, 3D, gradient, anti-aliasing, smooth shading, shadow, background, scenery, four-legged, crawling, animal pose --r 4
```

---

## 角色状态 × 情绪 Prompt

> 所有 prompt 已内嵌固定后缀，可直接复制粘贴到 Discord `/imagine prompt` 后使用。
> 发送前无需修改任何内容，URL 已填入。

---

### 😐 Idle — 待机

```
16-bit pixel art game sprite, Maodie the anthropomorphic orange tabby cat samurai, sitting cross-legged in formal seiza position, paws resting on knees, eyes half-closed in stern meditation, tail curled beside him, mushroom kasa hat perfectly straight, tiny wooden katana across his lap, pure white background, no shadow, isolated character, front-facing.

--niji 7 --s 20 --c 3 --style raw --ar 1:1 --cref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --cw 100 --sref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --sw 100 --no text, watermark, realistic, 3D, gradient, anti-aliasing, smooth shading, shadow, background, scenery
```

---

### 🤔 Thinking — 思考中

```
16-bit pixel art game sprite, Maodie the anthropomorphic orange tabby cat samurai, dramatic thinking pose, one paw stroking chin like a detective, eyebrows intensely furrowed, one eye squinted, leaning forward, small thought bubble with mushroom silhouette and question mark above head, mushroom kasa hat slightly tilted, pure white background, no shadow, isolated character, front-facing.

--niji 7 --s 20 --c 3 --style raw --ar 1:1 --cref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --cw 100 --sref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --sw 100 --no text, watermark, realistic, 3D, gradient, anti-aliasing, smooth shading, shadow, background, scenery
```

---

### 😄 Happy — 正确预测

```
16-bit pixel art game sprite, Maodie the anthropomorphic orange tabby cat samurai, jumping slightly with one paw raised in a subtle fist pump, face fighting between stern expression and involuntary smile breaking through, mushroom kasa hat bouncing off head slightly, tail straight up with excitement, small pixel sparkles around him, pure white background, no shadow, isolated character, front-facing.

--niji 7 --s 20 --c 3 --style raw --ar 1:1 --cref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --cw 100 --sref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --sw 100 --no text, watermark, realistic, 3D, gradient, anti-aliasing, smooth shading, shadow, background, scenery
```

---

### 🤢 Sick — 误判中毒

```
16-bit pixel art game sprite, Maodie the anthropomorphic orange tabby cat samurai, lying dramatically on his side in kabuki death scene pose, face tinted green, X-shaped eyes, tongue sticking out, small green pixel poison cloud above him, mushroom kasa hat rolled away beside him, wooden katana lying on ground, one paw reaching toward sky dramatically, pure white background, no shadow, isolated character.

--niji 7 --s 20 --c 3 --style raw --ar 1:1 --cref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --cw 100 --sref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --sw 100 --no text, watermark, realistic, 3D, gradient, anti-aliasing, smooth shading, shadow, background, scenery
```

---

### 😤 Proud — 升级

```
16-bit pixel art game sprite, Maodie the anthropomorphic orange tabby cat samurai, standing tall with arms crossed, chin slightly raised, eyes closed in satisfied expression, chest puffed out with pride, mushroom kasa hat perfectly straight, small pixel star sparkles around him, pure white background, no shadow, isolated character, front-facing.

--niji 7 --s 20 --c 3 --style raw --ar 1:1 --cref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --cw 100 --sref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --sw 100 --no text, watermark, realistic, 3D, gradient, anti-aliasing, smooth shading, shadow, background, scenery
```

---

### 🔍 Curious — 嗅探新样本

```
16-bit pixel art game sprite, Maodie the anthropomorphic orange tabby cat samurai, leaning forward with nose extended sniffing intensely, eyes wide and focused, one paw raised slightly, whiskers twitching forward, mushroom kasa hat tilting forward from leaning, expression extremely serious as if sniffing a deadly enemy, pure white background, no shadow, isolated character, front-facing.

--niji 7 --s 20 --c 3 --style raw --ar 1:1 --cref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --cw 100 --sref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --sw 100 --no text, watermark, realistic, 3D, gradient, anti-aliasing, smooth shading, shadow, background, scenery
```

---

### 😴 Sleeping — 休眠

```
16-bit pixel art game sprite, Maodie the anthropomorphic orange tabby cat samurai, sitting slumped with eyes closed, head drooping forward, small pixel ZZZ bubbles floating above, mushroom kasa hat drooping to one side, paws folded in lap, still wearing full samurai outfit, pure white background, no shadow, isolated character, front-facing.

--niji 7 --s 20 --c 3 --style raw --ar 1:1 --cref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --cw 100 --sref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --sw 100 --no text, watermark, realistic, 3D, gradient, anti-aliasing, smooth shading, shadow, background, scenery
```

---

### 😕 Confused — 误判可食（实际有毒）

```
16-bit pixel art game sprite, Maodie the anthropomorphic orange tabby cat samurai, head tilted to one side with puzzled expression, one eyebrow raised, one paw raised to cheek, small question mark pixel bubble above head, mushroom kasa hat tilting with the head tilt, looking genuinely baffled but trying to maintain dignity, pure white background, no shadow, isolated character, front-facing.

--niji 7 --s 20 --c 3 --style raw --ar 1:1 --cref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --cw 100 --sref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --sw 100 --no text, watermark, realistic, 3D, gradient, anti-aliasing, smooth shading, shadow, background, scenery
```

---

### 🎉 Celebrating — 旅程完成

```
16-bit pixel art game sprite, Maodie the anthropomorphic orange tabby cat samurai, both arms raised triumphantly in the air, rare full smile on face, eyes sparkling, mushroom kasa hat flying off above head, tail curled up in excitement, pixel confetti and stars exploding around him, still wearing full samurai outfit, pure white background, no shadow, isolated character, front-facing.

--niji 7 --s 20 --c 3 --style raw --ar 1:1 --cref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --cw 100 --sref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --sw 100 --no text, watermark, realistic, 3D, gradient, anti-aliasing, smooth shading, shadow, background, scenery
```

---

## 等级外观进化 Prompt

> 以下 prompt 作为**附加描述段**，追加在对应 mood prompt 的角色描述之后、参数之前。

### Level 0 — 无知小猫 (Clueless Kitten)

```
Make Maodie extra small with baby kitten proportions, oversized round head, enormous wide eyes relative to body. The mushroom kasa hat is way too big and keeps falling over one eye. The samurai gi is baggy and dragging on the ground. The wooden katana is comically large compared to his tiny body. He looks like a toddler who raided his father's wardrobe. Endearingly incompetent but deadly serious about it.

称号显示: "无知小猫"
```

### Level 1 — 好奇啃菇者 (Curious Nibbler)

```
Maodie is slightly larger, the kasa hat fits properly now. He carries a tiny magnifying glass in one paw, examining everything with forensic intensity. Slightly more alert and upright posture than level 0. The samurai gi fits better. Still small but growing into his role. A tiny satchel hangs at his side.

称号显示: "好奇啃菇者"
```

### Level 2 — 孢子侦察兵 (Spore Spotter)

```
Normal proportions now. Maodie carries a small field notebook tucked under one arm, filled with mushroom sketches. His expression is more focused and professionally stern. Posture is straighter and more disciplined. The gi is clean and properly tied. A headband peeks out from under the kasa hat.

称号显示: "孢子侦察兵"
```

### Level 3 — 菌褶检察官 (Gill Inspector)

```
Maodie wears a tiny white lab coat OVER his samurai gi -- an absurd combination of scientist and warrior. Expression is more serious and confident with slightly sharper eyes. The kasa hat sits neatly. A monocle hangs from a chain. He has the bearing of a samurai scholar. Professional yet ridiculous.

称号显示: "菌褶检察官"
```

### Level 4 — 菌道学徒 (Myco Apprentice)

```
Maodie looks slightly more muscular and battle-hardened. He carries a rolled-up mushroom identification scroll on his back like a warrior's battle plans. Expression is determined and weathered. The lab coat has a small badge with a mushroom emblem. A scar-shaped marking on one ear. He looks like a veteran samurai with a PhD.

称号显示: "菌道学徒"
```

### Level 5 — 蘑菇大师 (Fungal Master)

```
Full master warrior-scholar look. Maodie wears reading glasses perched on his nose, the lab coat is pristine white, a golden mushroom pin on the lapel. He holds a framed diploma in one paw. The wooden katana has been replaced with a gleaming katana with a mushroom-shaped tsuba guard. His kasa hat is decorated with golden trim. A faint scholarly pixel glow aura surrounds him. Expression is wise, serene, and impossibly dignified for a cat. He has achieved enlightenment through mushrooms.

称号显示: "蘑菇大师"
```

---

## 场景 & 背景 Prompt

> 场景背景使用 Niji 7，不需要 --cref（角色不在背景图里），只用 --sref 保持风格一致。

### 训练场背景 — 森林空地

```
16-bit pixel art game background, retro RPG style forest clearing at dawn. Tall pixel trees with dappled light, scattered mushrooms of various colors growing from mossy ground, a small wooden training post in the center. Fallen autumn leaves, fireflies. Peaceful but mysterious atmosphere. Flat 2D side-scroller perspective. Limited color palette. Pure flat background, no characters.

--niji 7 --ar 16:9 --s 50 --style raw --sref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --sw 80 --no text, realistic, 3D, anti-aliasing, characters, people --q 1
```

### 专家模式背景 — 蘑菇鉴定实验室

```
16-bit pixel art game background, a cozy mushroom research laboratory interior. Wooden shelves filled with glass jars containing preserved mushroom specimens, a large magnifying station, scrolls and books everywhere, a bubbling cauldron, dried mushrooms hanging from ceiling. Warm candlelight. A calligraphy scroll on wall. Side-scroller perspective. Pure flat background, no characters.

--niji 7 --ar 16:9 --s 50 --style raw --sref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --sw 80 --no text, realistic, 3D, anti-aliasing, characters, people --q 1
```

### 旅程地图背景 — 古老卷轴

```
16-bit pixel art treasure map on aged parchment, showing a winding path through forests mountains and mushroom villages. Pixel landmarks marked with tiny mushroom icons. A compass rose in one corner. Dotted trail with numbered stops. Old Japanese map style mixed with game world aesthetic. No characters.

--niji 7 --ar 3:2 --s 50 --style raw --sref https://media.discordapp.net/attachments/1492561065819771022/1492600633617023026/luci_08993_16-bit_pixel_art_character_sheet_retro_game_sprite_s_669acbeb-c320-4805-abfb-9a520607c4fc.png --sw 80 --no text, realistic, 3D, anti-aliasing, characters, people --q 1
```

---

## 配件 & 道具 Prompt

> 配件单独生成，确保可以作为独立图层叠加。使用 Niji 7，无需 --cref。

### 蘑菇斗笠 (Kasa Hat)

```
16-bit pixel art item sprite, a single mushroom-shaped samurai kasa hat. Red cap with white spots like amanita muscaria, with a chin strap. Simple clean pixel art, white background, game item icon style. Front and side view.

--niji 7 --ar 1:1 --s 20 --style raw --no text, realistic, 3D, anti-aliasing --q 1
```

### 腰带蘑菇纹样木刀

```
16-bit pixel art item sprite, a small wooden katana sword with a rope belt featuring a small carved mushroom symbol. Simple handle wrapping. Clean pixel art, white background, game weapon icon style.

--niji 7 --ar 1:1 --s 20 --style raw --no text, realistic, 3D, anti-aliasing --q 1
```

### 放大镜 (Level 1)

```
16-bit pixel art item sprite, a cute tiny magnifying glass with wooden handle, game item icon, white background, clean pixel art.

--niji 7 --ar 1:1 --s 20 --style raw --no text, realistic, 3D --q 1
```

### 田野笔记本 (Level 2)

```
16-bit pixel art item sprite, a small leather-bound field notebook with mushroom sketches visible on open page, game item icon, white background, clean pixel art.

--niji 7 --ar 1:1 --s 20 --style raw --no text, realistic, 3D --q 1
```

### 迷你白大褂 (Level 3)

```
16-bit pixel art item sprite, a tiny white lab coat sized for a cat, with a small mushroom emblem badge, game item icon, white background, clean pixel art.

--niji 7 --ar 1:1 --s 20 --style raw --no text, realistic, 3D --q 1
```

### 蘑菇鉴定卷轴 (Level 4)

```
16-bit pixel art item sprite, a rolled-up scroll with mushroom illustrations peeking out, tied with a red ribbon, game item icon, white background, clean pixel art.

--niji 7 --ar 1:1 --s 20 --style raw --no text, realistic, 3D --q 1
```

### 毕业证 + 金蘑菇胸针 (Level 5)

```
16-bit pixel art item sprites, a framed diploma with a wax seal and a gleaming golden mushroom pin brooch, game item icons, white background, clean pixel art.

--niji 7 --ar 1:1 --s 20 --style raw --no text, realistic, 3D --q 1
```

---

## UI 元素 & 特效 Prompt

### 蘑菇图标 — 可食用系列

```
16-bit pixel art icon set, 5 different edible mushroom varieties as cute game item icons, arranged in a row. Warm friendly colors, green check marks subtly incorporated. Clean pixel art, white background, game UI style.

--niji 7 --ar 5:1 --s 20 --style raw --no text, realistic, 3D --q 1
```

### 蘑菇图标 — 有毒系列

```
16-bit pixel art icon set, 5 different poisonous mushroom varieties as menacing game item icons, arranged in a row. Purple and sickly green tones, tiny skull-and-crossbone hints. Clean pixel art, white background, game UI style.

--niji 7 --ar 5:1 --s 20 --style raw --no text, realistic, 3D --q 1
```

### 毒雾特效

```
16-bit pixel art effect sprite, a green-purple poison cloud puff animation sequence, 4 frames arranged horizontally. Expanding toxic mist with small skull shapes. Transparent background style, game VFX sprite sheet.

--niji 7 --ar 4:1 --s 20 --style raw --no text, realistic, 3D --q 1
```

### 闪光特效

```
16-bit pixel art effect sprite, golden sparkle burst animation sequence, 4 frames arranged horizontally. Radiating pixel stars and light particles. Transparent background style, game VFX sprite sheet.

--niji 7 --ar 4:1 --s 20 --style raw --no text, realistic, 3D --q 1
```

### 对话气泡框

```
16-bit pixel art UI element, a speech bubble frame with a mushroom-shaped pointer tail. Cream-colored interior, dark pixel border, slight parchment texture. Game dialogue box style. Clean isolated asset.

--niji 7 --ar 3:2 --s 20 --style raw --no text, realistic, 3D --q 1
```

### 等级徽章模板

```
16-bit pixel art UI badge template, a circular emblem frame with crossed katana and mushroom motifs, space for level number in center. Gold and dark blue colors. Game achievement badge style.

--niji 7 --ar 1:1 --s 20 --style raw --no text, realistic, 3D --q 1
```

---

## Asset 清单总表

### 精灵图 (PNG, 透明背景)

| Asset ID | 文件路径 | 内容 | 尺寸 |
|---|---|---|---|
| `maodie-L0-idle` | `maodie/level0/idle.png` | 待机循环（4帧） | 800×200 |
| `maodie-L0-thinking` | `maodie/level0/thinking.png` | 思考循环（6帧） | 1200×200 |
| `maodie-L0-happy` | `maodie/level0/happy.png` | 正确胜利（4帧） | 800×200 |
| `maodie-L0-sick` | `maodie/level0/sick.png` | 中毒反应（5帧） | 1000×200 |
| `maodie-L0-proud` | `maodie/level0/proud.png` | 升级庆祝（4帧） | 800×200 |
| `maodie-L0-curious` | `maodie/level0/curious.png` | 嗅探（4帧） | 800×200 |
| `maodie-L0-sleeping` | `maodie/level0/sleeping.png` | 睡眠循环（3帧） | 600×200 |
| `maodie-L0-confused` | `maodie/level0/confused.png` | 困惑歪头（3帧） | 600×200 |
| `maodie-L0-celebrating` | `maodie/level0/celebrating.png` | 终极庆典（6帧） | 1200×200 |
| ... | `maodie/level1/` 至 `maodie/level5/` | 每等级相同 9 种情绪 | 同上 |

**总计: 9 情绪 × 6 等级 = 54 张精灵图**
**注意: 每张使用 --cref 的图 GPU 成本 ×2，实际消耗相当于 108 张普通图**

### 静态素材

| 素材 | 文件路径 | 用途 |
|---|---|---|
| 角色肖像（头像） | `maodie/portrait.png` | 顶栏头像、会话卡片 |
| 对话气泡框 | `ui/speech-bubble.svg` | 对话容器 |
| 等级徽章模板 | `ui/level-badge.svg` | 覆盖在角色头像上 |
| 蘑菇斗笠 | `maodie/accessories/kasa-hat.png` | 可合成到角色头部 |
| 放大镜 | `maodie/accessories/magnifying-glass.png` | Level 1 配件 |
| 田野笔记本 | `maodie/accessories/notebook.png` | Level 2 配件 |
| 白大褂 | `maodie/accessories/labcoat.png` | Level 3 配件 |
| 鉴定卷轴 | `maodie/accessories/scroll.png` | Level 4 配件 |
| 毕业证 + 金蘑菇胸针 | `maodie/accessories/diploma-pin.png` | Level 5 配件 |

### 背景 & 场景素材

| 素材 | 文件路径 | 用途 |
|---|---|---|
| 森林空地（训练） | `scenes/bg-forest.png` | 训练场景背景 |
| 实验桌（专家） | `scenes/bg-lab.png` | 专家模式背景 |
| 旅途地图（卷轴） | `scenes/bg-journey-map.png` | 旅途/进度页 |
| 蘑菇图标（可食用） | `mushrooms/icons/edible-{1-5}.svg` | UI 装饰 |
| 蘑菇图标（有毒） | `mushrooms/icons/poison-{1-5}.svg` | UI 装饰 |
| 毒雾特效 | `ui/effects/poison-cloud.png` | 中毒叠加特效 |
| 闪光特效 | `ui/effects/sparkle.png` | 正确预测庆祝 |
| 纸屑飘落 | `ui/effects/confetti.json` | 升级 Lottie 纸屑 |

---

## 对话系统模板

Maodie 通过对话气泡说话。语气始终像一个过于严肃的年轻武士。

### 打招呼（会话开始）

- **Level 0:** "吾乃 Maodie……虽然对蘑菇一无所知，但吾已立下武士之誓！请赐教！*紧张地握住木刀*"
- **Level 3:** "又一个样本。*调整白大褂* 待吾以菌褶检察之眼审视之。"
- **Level 5:** "欢迎回来。吾已鉴定 {n} 个样本，准确率 {acc}%。*扶了扶眼镜* 蘑菇之道，永无止境。"

### 预测（揭晓前）

- 有信心: "以吾之剑为誓——此菇 {verdict}。那 {feature} 绝不会骗过吾的眼睛。"
- 不确定: "唔……*汗流浃背* 吾判定为 {verdict}……但说实话，只有 {conf}% 的信心。父亲在上，保佑弟子。"
- 完全没底: "…… *盯着蘑菇三秒* ……{verdict}？*声音发抖*"

### 预测正确

- "哈！武士的直觉果然不会错！那个 {feature} 就是破绽！"
- "连续 {correct}/{total} 正确！菇道修行，大有进步！*克制住笑意*"
- "…… *嘴角微微上扬后立刻恢复严肃* 吾从未怀疑过。"

### 预测错误（假阴性 — 说可食，实际有毒）

- "*整个猫变绿* 什……什么！这是毒菇？！吾差点以身殉菇！！"
- "*夸张地倒在地上* 父亲……对不起……弟子还需继续修行…… *四脚朝天*"
- "*躺在地上* 记住了……{feature} 不可轻信……这是用生命换来的教训……"

### 预测错误（假阳性 — 说有毒，实际可食）

- "什么？这……这居然是安全的？*困惑地歪头* 吾太过谨慎了。"
- "唔，吾需要学习——并非所有 {cap_color} 的菇都是敌人。"
- "*尴尬地站直* 吾……吾只是在进行压力测试。绝对不是判断失误。"

### 升级

- 0→1: "吾终于能分辨蘑菇和石头了！这……这也算进步吧！*握紧放大镜*"
- 1→2: "获得田野笔记本！从今日起，每一朵蘑菇都将被记录在案！*认真翻开笔记本*"
- 2→3: "白……白大褂？！*颤抖着穿上* 吾现在既是武士，又是学者了！"
- 3→4: "吾已经能在梦里识别菌褶了。来吧，给吾最难的蘑菇！*展开鉴定卷轴*"
- 4→5: "吾……做到了。蘑菇大师。*摘下斗笠擦去眼角的泪水* 父亲，您看到了吗。*戴上眼镜* 所有的蘑菇都值了。"

---

## 动画状态机

### 训练模式（自动循环）

```
  [暂停 >2分钟]
       │
  ┌────▼────────┐   [按下开始]      ┌─────────────┐
  │   sleeping  │◄─────────────── │    idle     │◄─── [按下暂停]
  └────┬────────┘                  └──────┬──────┘
       │ [按下开始]                        │ [按下开始]
       └──────────────────────────────────┘
                                           │
                                    ┌──────▼──────┐
                          ┌─────────│   curious   │◄──────────────────┐
                          │         └──────┬──────┘   (循环重复)       │
                          │  0.3–1.2s      │                           │
                          │         ┌──────▼──────┐                   │
                          │         │   thinking  │ 0.5–2s (预测)     │
                          │         └──────┬──────┘                   │
                          │          自动揭晓                          │
                          │       ┌─────────────┐                     │
                          │       │  正确吗？   │                     │
                          │       └──┬───────┬──┘                     │
                          │       是 │       │ 否                     │
                          │    ┌─────▼──┐ ┌──▼──────────────────┐    │
                          │    │ happy  │ │ 真实标签=有毒？     │    │
                          │    └─────┬──┘ └──┬──────────────┬───┘    │
                          │          │      是│            否│       │
                          │          │    ┌────▼──┐    ┌─────▼────┐  │
                          │          │    │  sick │    │ confused │  │
                          │          │    └────┬──┘    └─────┬────┘  │
                          │          └─────────┴─────────────┘       │
                          │                    │ 检查经验值/准确率    │
                          │             ┌──────▼─────┐               │
                          │             │  升级了吗？ ├─ 是 ─► proud ─┤
                          │             └──────┬─────┘               │
                          │                    │ 否                   │
                          └────────────────────┴─────────────────────┘
```

> **训练循环情绪链:** `curious` → `thinking` → [`happy` / `sick` / `confused`] → (升级? → `proud`) → `curious` → …

---

## 生成优先级

| 优先级 | 情绪 | 等级 | 原因 |
|---|---|---|---|
| **P0** | `curious`, `thinking`, `happy`, `sick`, `confused` | 0 | 每个训练周期必看 |
| **P1** | `idle`, `proud` | 0 | 暂停状态 + 升级 |
| **P2** | `curious`, `thinking`, `happy`, `sick`, `confused` | 1, 2 | 前期升级 |
| **P3** | 全部 9 种 | 3, 4, 5 | 进阶用户 |
| **P4** | `sleeping`, `celebrating` | 0 | 低频边缘场景 |

**推荐首批生成: Level 0 全部 9 种情绪 = 9 张图**

---

## 附录：常见问题 & 注意事项

| 问题 | 解决方法 |
|---|---|
| 基准图 URL 失效 | 重新上传图片到 Discord，右键复制新链接，替换文档中所有 URL |
| 生成结果不够像素 | 降低 --s 值（试 10-15），加强 `chunky pixel blocks` 描述 |
| 角色变成四脚猫 | 在 --no 里确保有 `four-legged, crawling, animal pose` |
| 背景没有纯白 | 正文加 `pure white background #FFFFFF, no shadow, no ground shadow` |
| 一致性不够 | 提高 --ow 值（试 200），或检查基准图URL是否有效 |
| 中文字符无法渲染 | 改用图案描述代替文字，如 `mushroom symbol` 代替"菇"字 |

---

> **最终提醒：** 基准图 URL 已内嵌在所有 prompt 中，直接复制粘贴即可使用。如 Discord CDN 链接失效，重新上传基准图并全局替换 URL。祝蘑菇武士之旅顺利！🍄⚔️🐱
