# 🍄⚔️ Maodie — 蘑菇武士角色设计 Midjourney Prompt 完整手册

> **角色重塑：** 从"Professor Whiskers 教授猫"变为 **Maodie（猫蝶）** —— 一只严肃到令人发笑的年轻猫咪武士。他的父亲因误食毒蘑菇去世，他发誓要成为世界上最强的蘑菇武士，用武道守护所有人远离毒菇之害。

---

## 目录

1. [角色档案](#角色档案)
2. [Midjourney 参数速查表](#midjourney-参数速查表)
3. [一致性策略](#一致性策略)
4. [第一步：角色基准图生成](#第一步角色基准图生成)
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
| **标志配饰** | 头戴蘑菇形斗笠（红色伞帽白色斑点，像毒蝇伞），腰挎一柄写着「菇」字的木刀 |
| **性格** | 极度严肃认真，但因为太认真了反而很搞笑；动作夸张，经常用武士礼仪对待蘑菇；失败时极其戏剧化地倒地 |
| **视觉风格** | 游戏像素风（16-bit / 32-bit pixel art），粗像素轮廓，有限调色板，复古游戏感 |
| **尺寸** | ~200×200px 头像（响应式缩放） |
| **设计哲学** | 像素版的武士猫 × 搞笑严肃反差萌。想象一只猫穿着武士装备，用对待生死大敌的态度去嗅一朵蘑菇。越认真越好笑 |

---

## Midjourney 参数速查表

> 以下为本项目会用到的关键参数，基于 Midjourney V7（当前默认版本）及 Niji 7。

### 核心参数

| 参数 | 语法 | 说明 | 本项目推荐值 |
|---|---|---|---|
| **版本** | `--v 7` | 选择模型版本，V7 为当前默认 | `--v 7` 或 `--niji 7`（更适合像素/动漫风） |
| **宽高比** | `--ar W:H` | 控制输出图像比例 | `--ar 1:1`（头像/精灵图）、`--ar 4:1`（精灵序列帧） |
| **风格化** | `--stylize N` 或 `--s N` | 0-1000，值越高 MJ 的艺术自由度越高 | `--s 50`（低值以保持像素风格精确） |
| **混沌度** | `--chaos N` 或 `--c N` | 0-100，控制四张图之间的变化程度 | `--c 5`（低值确保一致性） |
| **怪异度** | `--weird N` 或 `--w N` | 0-3000，引入非常规美学 | `--w 0`（本项目不使用） |
| **质量** | `--quality N` 或 `--q N` | 渲染耗时，0.25/0.5/1/2 | `--q 1`（标准质量） |
| **排除** | `--no [元素]` | 排除不想出现的元素 | `--no text, watermark, realistic, 3D, gradient, shadow, anti-aliasing` |
| **重复** | `--repeat N` 或 `--r N` | 同一 prompt 运行 N 次 | `--r 4`（批量生成挑选最佳） |
| **种子** | `--seed N` | 0-4294967295，固定随机噪声起点 | 找到好图后锁定 seed 进行微调 |
| **Draft 模式** | `--draft` | V7专属，10倍快速，半价，适合快速迭代 | 初始探索阶段使用 |
| **Tile** | `--tile` | 生成可无缝拼接的图块 | 仅背景纹理使用 |
| **Raw 风格** | `--style raw` | 更贴近 prompt 字面意思，减少 MJ 自行美化 | `--style raw`（像素风建议使用） |

### 一致性参数（关键！）

| 参数 | 语法 | 说明 | 本项目用法 |
|---|---|---|---|
| **Omni Reference** | `--oref [URL]` | V7 专属！提供参考图让 MJ 复制角色/物体特征 | 生成基准图后，所有后续 prompt 都附加 `--oref [基准图URL]` |
| **Omni Weight** | `--ow N` | 0-1000（默认100），控制 oref 参考强度 | `--ow 150`（偏高以确保角色特征锁定） |
| **Style Reference** | `--sref [URL]` | 复制参考图的艺术风格/色调/美学 | 用一张满意的像素风参考图锁定全套风格 |
| **Style Weight** | `--sw N` | 控制 sref 强度 | `--sw 100` |
| **Character Reference** | `--cref [URL]` | V6 专属（V7 中用 oref 替代），复制角色面部/服装 | 如果使用 Niji 6 则用此参数 |
| **Character Weight** | `--cw N` | 0-100，控制 cref 匹配强度 | `--cw 100` |

### 图像参考

| 参数 | 语法 | 说明 |
|---|---|---|
| **Image Prompt** | 在 prompt 最前方放置图片 URL | 影响构图和内容 |
| **Image Weight** | `--iw N` | 0-2，控制图片 prompt 的影响力 |

---

## 一致性策略

确保 Maodie 在 54+ 张素材中保持视觉统一的完整流程：

### 第一阶段：建立基准
1. **生成角色基准图**（Character Sheet）— 中性姿势、正面、清晰背景
2. 找到满意的基准图后，**记录 seed 值**
3. 将此图的 URL 作为后续所有 prompt 的 `--oref` 参考

### 第二阶段：锁定风格
4. 将基准图同时设为 `--sref`，确保像素风格一致
5. 所有 prompt 末尾统一附加：`--oref [基准URL] --ow 150 --sref [风格URL] --sw 100`

### 第三阶段：参考照片（猫的严肃脸）
6. 上传你的严肃猫照片作为面部情绪参考：
   - 在 Web 界面拖入图片区域，或 Discord 中作为 image prompt
   - **照片参考链接预留位：** `[YOUR_SERIOUS_CAT_PHOTO_URL]`
   - 使用方式：将照片 URL 放在 prompt 最前面作为 image prompt，并设 `--iw 0.5`（轻度参考情绪，不覆盖像素风）

### 第四阶段：批量生产
7. 按优先级表逐步生成所有 mood × level 组合
8. 每生成一批，检查一致性，必要时微调 `--ow` 值

---

## 第一步：角色基准图生成

> ⚠️ 这是最关键的一步！后续所有素材的一致性都依赖于此图。

### Prompt — 角色 Sheet（正面全身，中性姿势）

```
[YOUR_SERIOUS_CAT_PHOTO_URL]
https://cdn.discordapp.com/attachments/1492561065819771022/1492569838227296408/d9645876ea04480faa6f825866c9d60b_s.png?ex=69dbcf7c&is=69da7dfc&hm=21a6f850e78c3a517444ea6b410a14d41f9bcc891654c52a7fb9a768bccc55ee
16-bit pixel art character sheet, retro game sprite style, limited color palette of warm oranges, cream white, deep red, and forest green. Visible individual pixels, no anti-aliasing, crisp pixel edges.

Subject: a young orange tabby cat samurai warrior named Maodie, standing in a serious neutral pose facing forward. The cat has an absurdly serious deadpan expression on a round fluffy face, tiny furrowed brows, intense focused eyes, small frown mouth -- comedically over-serious for a small cat.

Wearing: a red-and-white-spotted mushroom-shaped kasa hat (like amanita muscaria cap) on head, a miniature dark blue samurai gi robe tied with a belt, and a tiny wooden katana with the character 菇 carved on it strapped to the waist.

The character stands upright on hind legs in a dignified samurai stance, paws at sides, chest slightly puffed, looking like a tiny warrior taking himself way too seriously.

White background, transparent PNG style, front-facing character reference sheet, clean simple pixel art suitable for game sprite, no text overlay.

--niji 7 --s 30 --c 5 --ar 1:1 --no text, watermark, realistic, 3D, gradient, anti-aliasing, smooth shading
```

### Prompt — 角色 Sheet（多角度，可选）

```
[YOUR_SERIOUS_CAT_PHOTO_URL]

16-bit pixel art character turnaround sheet, front view, side view, back view, and 3/4 view of the same character. Retro game sprite style, limited color palette, visible individual pixels, crisp pixel edges, no anti-aliasing.

Subject: a young orange tabby cat samurai named Maodie with an absurdly serious expression, wearing a red-and-white amanita mushroom kasa hat, dark blue samurai gi, and a tiny wooden katana at waist with 菇 character.

White background, game asset reference sheet layout, clean pixel art.

--v 7 --ar 2:1 --s 30 --c 5 --style raw --no text, watermark, realistic, 3D, gradient, anti-aliasing --q 1 --r 4
```

---

## 角色状态 × 情绪 Prompt

> 所有 mood prompt 共享以下**前缀**（Base Prompt），请复制粘贴在每个 mood prompt 前面。

### 🔧 Base Prompt（全局前缀）

```
[YOUR_SERIOUS_CAT_PHOTO_URL]

16-bit pixel art game sprite, retro pixel style, limited color palette of warm oranges cream white deep red and forest green, visible individual pixels, no anti-aliasing, crisp pixel edges.

Character: Maodie, a young orange tabby cat samurai with a red-and-white amanita mushroom kasa hat, dark blue samurai gi robe, tiny wooden katana at waist.
```

### 🔧 全局后缀（Suffix）

```
White background, transparent PNG style, single character, clean pixel art game sprite, no text.

--oref [YOUR_BASE_CHARACTER_URL] --ow 150 --sref [YOUR_STYLE_REFERENCE_URL] --sw 100 --v 7 --ar 1:1 --s 30 --c 5 --style raw --no text, watermark, realistic, 3D, gradient, anti-aliasing, smooth shading --q 1
```

---

### 😐 Idle — 待机

```
{Base Prompt}

Maodie is sitting cross-legged in formal seiza position like a meditating samurai, paws resting on knees, eyes half-closed in stern meditation, tail gently swaying behind. The mushroom kasa hat sits perfectly straight. His expression is deadly serious as if guarding a sacred treasure, but he is just a small fluffy cat sitting still. A tiny 菇 wooden katana rests across his lap.

{Suffix}
```

### 🤔 Thinking — 思考中

```
{Base Prompt}

Maodie is in a dramatic thinking pose, one paw stroking his chin like a detective examining evidence, eyebrows intensely furrowed, one eye squinted. He is leaning forward studying something invisible with the focus of a master swordsman reading an opponent. A tiny thought bubble with a mushroom silhouette and question mark floats above. His kasa hat is slightly tilted from concentration. The seriousness is comically disproportionate.

{Suffix}
```

### 😄 Happy — 正确预测

```
{Base Prompt}

Maodie has jumped into the air in a restrained victory pose, but trying desperately to look cool about it. One paw is raised in a subtle fist pump while his face fights between a stern expression and an involuntary smile breaking through. His kasa hat is bouncing slightly off his head. Small pixel sparkle effects around him. His tail is straight up with excitement he is trying to hide. A samurai who just won a duel but must remain dignified.

{Suffix}
```

### 🤢 Sick — 误判可食（实际有毒）

```
{Base Prompt}

Maodie is lying dramatically on his side in an over-the-top death scene pose, like a kabuki actor performing a dramatic death. His face is tinted green with pixel X-shaped eyes and tongue sticking out. Tiny pixel mushroom crumbs near his mouth. A green pixel poison cloud puffs above him. His kasa hat has rolled away. His wooden katana lies beside him. One paw reaches toward the sky dramatically. The whole scene looks like a theatrical samurai death but it is just a small cat who ate the wrong mushroom. Maximum dramatic comedy.

{Suffix}
```

### 😤 Proud — 升级

```
{Base Prompt}

Maodie stands tall in a powerful samurai victory stance on hind legs, arms crossed over chest, chin raised, eyes closed with the most smug self-satisfied expression possible. His kasa hat is tilted at a cool angle. A golden pixel aura glows around him. His wooden katana is planted in the ground beside him like a conquering warrior's sword. He looks like he just saved the entire kingdom but actually he just learned to identify one more mushroom. Pixel confetti or cherry blossom petals falling.

{Suffix}
```

### 🔍 Curious — 嗅探新样本

```
{Base Prompt}

Maodie is crouched low in a ninja-like stalking pose, creeping forward with intense hunter eyes locked on target, ears fully perked, whiskers spread wide, nose twitching. His paw is reaching forward cautiously like disarming a bomb. His tail is low and tense. His kasa hat has slid slightly forward. He is approaching a mushroom with the tactical seriousness of a samurai approaching a dragon, but it is just a regular mushroom on the ground. Comedic intensity.

{Suffix}
```

### 😴 Sleeping — 休眠

```
{Base Prompt}

Maodie is curled into a ball in a surprisingly cute sleeping position, tail wrapped over his nose, eyes peacefully closed. Three pixel "Z" letters float above in increasing size. His kasa hat has slipped down covering one eye. His wooden katana is clutched in his paws like a teddy bear. Even in sleep his tiny brow is slightly furrowed as if having serious dreams about mushrooms. A tiny dream bubble shows a pixel mushroom.

{Suffix}
```

### 😵 Confused — 误判有毒（实际可食）

```
{Base Prompt}

Maodie has his head tilted at a sharp angle, one ear up one ear down, eyes slightly crossed with genuine bewilderment. One paw is scratching the back of his head. A large pixel "?" and "!" float beside him spinning. His kasa hat is crooked and askew. His wooden katana has been dropped on the ground. He looks like a samurai who just realized his ultimate technique failed against a daikon radish. Comical disbelief on a serious face.

{Suffix}
```

### 🎉 Celebrating — 旅程完成

```
{Base Prompt}

Maodie is dancing on hind legs in a wild victory dance, wearing a tiny pixel party hat ON TOP of his mushroom kasa hat -- double hat situation. Pixel confetti and tiny mushroom shapes rain down. His eyes are sparkling with tears of joy but still trying to maintain warrior dignity. He holds a tiny framed diploma in one paw and a golden mushroom trophy in the other. His wooden katana is strapped to his back. He looks like a samurai at his own surprise birthday party -- thrilled but unsure if celebrating is honorable.

{Suffix}
```

---

## 等级外观进化 Prompt

> 以下 prompt 作为**附加描述段**，追加在对应 mood prompt 的角色描述之后、Suffix 之前。

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

### 训练场背景 — 森林空地

```
16-bit pixel art game background, retro RPG style forest clearing at dawn. Tall pixel trees with dappled light, scattered mushrooms of various colors growing from mossy ground, a small wooden training post in the center. Fallen autumn leaves, fireflies. Peaceful but mysterious atmosphere. Flat 2D side-scroller perspective. Limited color palette.

--oref [YOUR_STYLE_REFERENCE_URL] --sref [YOUR_STYLE_REFERENCE_URL] --v 7 --ar 16:9 --s 50 --style raw --no text, realistic, 3D, anti-aliasing --q 1
```

### 专家模式背景 — 蘑菇鉴定实验室

```
16-bit pixel art game background, a cozy mushroom research laboratory interior. Wooden shelves filled with glass jars containing preserved mushroom specimens, a large magnifying station, scrolls and books everywhere, a bubbling cauldron, dried mushrooms hanging from ceiling. Warm candlelight. A calligraphy scroll on wall reads 菇道. Side-scroller perspective.

--oref [YOUR_STYLE_REFERENCE_URL] --sref [YOUR_STYLE_REFERENCE_URL] --v 7 --ar 16:9 --s 50 --style raw --no text, realistic, 3D, anti-aliasing --q 1
```

### 旅程地图背景 — 古老卷轴

```
16-bit pixel art treasure map on aged parchment, showing a winding path through forests mountains and mushroom villages. Pixel landmarks marked with tiny mushroom icons. A compass rose in one corner. Dotted trail with numbered stops. Old Japanese map style mixed with game world aesthetic.

--oref [YOUR_STYLE_REFERENCE_URL] --sref [YOUR_STYLE_REFERENCE_URL] --v 7 --ar 3:2 --s 50 --style raw --no text, realistic, 3D, anti-aliasing --q 1
```

---

## 配件 & 道具 Prompt

> 每个配件单独生成，确保可以作为独立图层叠加。

### 蘑菇斗笠 (Kasa Hat)

```
16-bit pixel art item sprite, a single mushroom-shaped samurai kasa hat. Red cap with white spots like amanita muscaria, with a chin strap. Simple clean pixel art, white background, game item icon style. Front and side view.

--v 7 --ar 1:1 --s 20 --style raw --no text, realistic, 3D, anti-aliasing --q 1
```

### 菇字木刀

```
16-bit pixel art item sprite, a small wooden katana sword with the Chinese character 菇 carved into the blade. Simple handle wrapping. Clean pixel art, white background, game weapon icon style.

--v 7 --ar 1:1 --s 20 --style raw --no text, realistic, 3D, anti-aliasing --q 1
```

### 放大镜 (Level 1)

```
16-bit pixel art item sprite, a cute tiny magnifying glass with wooden handle, game item icon, white background, clean pixel art.

--v 7 --ar 1:1 --s 20 --style raw --no text, realistic, 3D --q 1
```

### 田野笔记本 (Level 2)

```
16-bit pixel art item sprite, a small leather-bound field notebook with mushroom sketches visible on open page, game item icon, white background, clean pixel art.

--v 7 --ar 1:1 --s 20 --style raw --no text, realistic, 3D --q 1
```

### 迷你白大褂 (Level 3)

```
16-bit pixel art item sprite, a tiny white lab coat sized for a cat, with a small mushroom emblem badge, game item icon, white background, clean pixel art.

--v 7 --ar 1:1 --s 20 --style raw --no text, realistic, 3D --q 1
```

### 蘑菇鉴定卷轴 (Level 4)

```
16-bit pixel art item sprite, a rolled-up scroll with mushroom illustrations peeking out, tied with a red ribbon, game item icon, white background, clean pixel art.

--v 7 --ar 1:1 --s 20 --style raw --no text, realistic, 3D --q 1
```

### 毕业证 + 金蘑菇胸针 (Level 5)

```
16-bit pixel art item sprites, a framed diploma with a wax seal and a gleaming golden mushroom pin brooch, game item icons, white background, clean pixel art.

--v 7 --ar 1:1 --s 20 --style raw --no text, realistic, 3D --q 1
```

---

## UI 元素 & 特效 Prompt

### 蘑菇图标 — 可食用系列

```
16-bit pixel art icon set, 5 different edible mushroom varieties as cute game item icons, arranged in a row. Warm friendly colors, green check marks subtly incorporated. Clean pixel art, white background, game UI style.

--v 7 --ar 5:1 --s 20 --style raw --no text, realistic, 3D --q 1
```

### 蘑菇图标 — 有毒系列

```
16-bit pixel art icon set, 5 different poisonous mushroom varieties as menacing game item icons, arranged in a row. Purple and sickly green tones, tiny skull-and-crossbone hints. Clean pixel art, white background, game UI style.

--v 7 --ar 5:1 --s 20 --style raw --no text, realistic, 3D --q 1
```

### 毒雾特效

```
16-bit pixel art effect sprite, a green-purple poison cloud puff animation sequence, 4 frames arranged horizontally. Expanding toxic mist with small skull shapes. Transparent background style, game VFX sprite sheet.

--v 7 --ar 4:1 --s 20 --style raw --no text, realistic, 3D --q 1
```

### 闪光特效

```
16-bit pixel art effect sprite, golden sparkle burst animation sequence, 4 frames arranged horizontally. Radiating pixel stars and light particles. Transparent background style, game VFX sprite sheet.

--v 7 --ar 4:1 --s 20 --style raw --no text, realistic, 3D --q 1
```

### 对话气泡框

```
16-bit pixel art UI element, a speech bubble frame with a mushroom-shaped pointer tail. Cream-colored interior, dark pixel border, slight parchment texture. Game dialogue box style. Clean isolated asset.

--v 7 --ar 3:2 --s 20 --style raw --no text, realistic, 3D --q 1
```

### 等级徽章模板

```
16-bit pixel art UI badge template, a circular emblem frame with crossed katana and mushroom motifs, space for level number in center. Gold and dark blue colors. Game achievement badge style.

--v 7 --ar 1:1 --s 20 --style raw --no text, realistic, 3D --q 1
```

---

## Asset 清单总表

### 精灵图 (PNG, 透明背景)

每个情绪 × 每个等级 = 一张精灵图。按等级分组为精灵序列帧。

| Asset ID | 文件路径 | 内容 | 尺寸 |
|---|---|---|---|
| `maodie-L0-idle` | `maodie/level0/idle.png` | 待机循环（4帧） | 800×200 (4×200) |
| `maodie-L0-thinking` | `maodie/level0/thinking.png` | 思考循环（6帧） | 1200×200 (6×200) |
| `maodie-L0-happy` | `maodie/level0/happy.png` | 正确胜利（4帧） | 800×200 |
| `maodie-L0-sick` | `maodie/level0/sick.png` | 中毒反应（5帧） | 1000×200 |
| `maodie-L0-proud` | `maodie/level0/proud.png` | 升级庆祝（4帧） | 800×200 |
| `maodie-L0-curious` | `maodie/level0/curious.png` | 嗅探（4帧） | 800×200 |
| `maodie-L0-sleeping` | `maodie/level0/sleeping.png` | 睡眠循环（3帧） | 600×200 |
| `maodie-L0-confused` | `maodie/level0/confused.png` | 困惑歪头（3帧） | 600×200 |
| `maodie-L0-celebrating` | `maodie/level0/celebrating.png` | 终极庆典（6帧） | 1200×200 |
| ... | `maodie/level1/` 至 `maodie/level5/` | 每等级相同 9 种情绪 | 同上 |

**总计: 9 情绪 × 6 等级 = 54 张精灵图**

### 替代方案: Lottie 动画（推荐）

| Asset ID | 文件路径 | 描述 |
|---|---|---|
| `maodie-L{n}-{mood}.json` | `maodie/lottie/level{n}/{mood}.json` | 每种情绪的 Lottie 动画 |

**共 54 个 Lottie 文件**

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

### 音效（可选增强）

| 音效 | 文件路径 | 触发条件 |
|---|---|---|
| 喵（好奇） | `sounds/meow-curious.mp3` | 新蘑菇出现 |
| 喵（开心） | `sounds/meow-happy.mp3` | 预测正确 |
| 喵（难受） | `sounds/meow-sick.mp3` | 误食毒菇 |
| 呼噜 | `sounds/purr.mp3` | 待机环境音 |
| 升级号角 | `sounds/levelup.mp3` | 角色升级 |
| 嚼蘑菇 | `sounds/munch.mp3` | 角色"吃"蘑菇 |
| 拔刀音效 | `sounds/katana-draw.mp3` | 进入新阶段 |
| 翻卷轴 | `sounds/scroll-unfurl.mp3` | 浏览旅途地图 |

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

### 选择下一个样本（自动训练循环）

- 随机: "让吾随缘选一朵……武士不畏未知！*闭眼伸手抓*"
- 不确定度: "吾偏要选最没把握的那一朵——真正的武士迎难而上！"
- 委员会: "吾心中的三位长老意见不一……必须亲自验证！"
- BALD: "信息论指出——这一朵的学习价值最高。*推了推眼镜* 这就是科学。"

### 自动训练中（间歇显示）

- "正在审查第 #{n} 号蘑菇样本……*严肃地嗅*"
- "已鉴定 {labeled} 朵蘑菇。吾的菇感越来越强了。"
- "准确率 {accuracy}% — {level_comment}"（例如: "对一只小猫来说还不错！" / "吾已今非昔比！"）

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
> 每轮结束回到 `curious`（不是 `idle`）。`idle` 仅在训练暂停时显示。

### 专家模式（用户上传照片）

```
  idle ──► [上传照片] ──► curious ──► thinking ──► happy / sick / confused ──► idle
```

---

## 生成优先级

| 优先级 | 情绪 | 等级 | 原因 |
|---|---|---|---|
| **P0** | `curious`, `thinking`, `happy`, `sick`, `confused` | 0 | 每个训练周期必看 |
| **P1** | `idle`, `proud` | 0 | 暂停状态 + 升级 |
| **P2** | `curious`, `thinking`, `happy`, `sick`, `confused` | 1, 2 | 前期升级 |
| **P3** | 全部 9 种 | 3, 4, 5 | 进阶用户 |
| **P4** | `sleeping`, `celebrating` | 0 | 低频边缘场景 |

**推荐首批生成: Level 0 全部 9 种情绪 = 9 张精灵图**

---

## 附录：Midjourney 工作流快速指南

### 完整生成流程

1. **探索阶段** — 使用 `--draft` 快速低成本出图，找到满意的角色方向
2. **基准锁定** — 用正式 prompt 生成角色 Sheet，锁定 `--seed`
3. **参考设定** — 将基准图 URL 设为 `--oref` + `--sref`
4. **批量生产** — 按优先级表，逐个 mood + level 组合生成
5. **质量检查** — 每批检查一致性，微调 `--ow`/`--sw`
6. **精修** — 使用 Midjourney Editor 修补不满意的细节区域

### 参考图片占位

| 用途 | 占位符 | 说明 |
|---|---|---|
| 严肃猫脸照片 | `[YOUR_SERIOUS_CAT_PHOTO_URL]` | 你的猫猫严肃脸参考照，放在 prompt 最前面 |
| 角色基准图 | `[YOUR_BASE_CHARACTER_URL]` | 第一步生成满意后的基准图 URL |
| 风格参考图 | `[YOUR_STYLE_REFERENCE_URL]` | 满意的像素风格图 URL（可与基准图相同） |

### Niji 7 替代方案

如果发现 V7 的像素风效果不够理想，可切换到 Niji 7（专为动漫/插画优化）：
- 将 `--v 7` 替换为 `--niji 7`
- Niji 7 对清晰线条和平面风格有更好的支持
- 注意：Niji 7 使用 `--cref` 而非 `--oref` 进行角色一致性控制

---

> **最终提醒：** 所有 `[YOUR_..._URL]` 占位符必须在实际使用时替换为真实的图片 URL。先完成"第一步：角色基准图生成"，获得满意的基准图后，再将其 URL 填入所有后续 prompt 中。祝你的蘑菇武士之旅顺利！🍄⚔️🐱
