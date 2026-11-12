# Cat Agent Character Design — "Professor Whiskers"

## Character Profile

| Attribute | Value |
|---|---|
| Name | Professor Whiskers (user can rename) |
| Species | Orange tabby cat with a tiny mushroom-shaped beret |
| Personality | Curious, slightly clumsy, earnest, dramatic when wrong |
| Visual Style | Flat illustration / chibi anime style, thick outlines, pastel palette |
| Size on Screen | ~200x200px avatar (scales responsively) |

**Design philosophy:** The cat should feel like a Duolingo owl crossed with a Studio Ghibli side character — cute enough to root for, expressive enough to convey learning progress, and dramatic enough to make mistakes entertaining.

---

## Cat States & Moods

The cat has **7 primary moods**, each with a distinct pose, expression, and optional animation loop.

| State | Trigger | Expression | Body Language | Duration |
|---|---|---|---|---|
| `idle` | Training paused / home page | Neutral, relaxed eyes | Sitting, tail gently swaying | Loop |
| `thinking` | Auto-loop: model predicting | Squinting, one eyebrow raised | Leaning forward, paw on chin | 2-4s |
| `happy` | Correct prediction | Big smile, sparkle eyes | Jumping, paws up | 1.5s → curious |
| `sick` | False negative (predicted edible → was poisonous) | Green face, X eyes, tongue out | Lying on side, mushroom crumbs | 2s → curious |
| `proud` | Level up | Smug grin, closed eyes | Standing tall, chest puffed, beret tilted | 2s → curious |
| `curious` | Auto-loop: picking next sample from pool | Wide eyes, ears perked | Sniffing forward, whiskers twitching | 0.3–1.2s → thinking |
| `sleeping` | Training paused > 2 min without interaction | Closed eyes, Zzz bubbles | Curled up, tail over nose | Loop until Start pressed |
| `confused` | False positive (predicted poisonous → was edible) OR low-confidence prediction | Tilted head, question mark | Scratching head with paw | 1.5s → curious |
| `celebrating` | Reached Level 5 / Expert journey complete | Party hat, confetti | Dancing on hind legs | 3s loop |

> **Training loop mood cycle:** `curious` → `thinking` → [`happy` / `sick` / `confused`] → (level-up? → `proud`) → `curious` → …
> The loop returns to `curious` (not `idle`) after each round. `idle` is only shown when training is paused.

---

## Level Appearances

The cat's appearance evolves as it levels up — visual progression reinforces the learning journey.

| Level | Title | Appearance Change | Accessory |
|---|---|---|---|
| 0 | Clueless Kitten | Small, wide-eyed, oversized beret falls over eyes | None |
| 1 | Curious Nibbler | Slightly bigger, beret fits better | Magnifying glass |
| 2 | Spore Spotter | Normal size, alert posture | Field notebook |
| 3 | Gill Inspector | Confident stance, sharper eyes | Lab coat (tiny) |
| 4 | Myco Apprentice | Slightly buff, serious expression | Mushroom identification chart (rolled up) |
| 5 | Fungal Master | Full professor look, wise eyes, neat beret | Diploma + golden mushroom pin |

---

## Art Asset Manifest

### Sprite Sheets (PNG, transparent background)

Each mood × each level = a sprite. For efficiency, group by level as sprite sheets.

| Asset ID | File | Contents | Dimensions |
|---|---|---|---|
| `cat-L0-idle` | `cat/level0/idle.png` | Idle loop (4 frames) | 800x200 (4×200) |
| `cat-L0-thinking` | `cat/level0/thinking.png` | Thinking loop (6 frames) | 1200x200 (6×200) |
| `cat-L0-happy` | `cat/level0/happy.png` | Happy jump (4 frames) | 800x200 |
| `cat-L0-sick` | `cat/level0/sick.png` | Poisoned reaction (5 frames) | 1000x200 |
| `cat-L0-proud` | `cat/level0/proud.png` | Level-up celebration (4 frames) | 800x200 |
| `cat-L0-curious` | `cat/level0/curious.png` | Sniffing (4 frames) | 800x200 |
| `cat-L0-sleeping` | `cat/level0/sleeping.png` | Sleep loop (3 frames) | 600x200 |
| `cat-L0-confused` | `cat/level0/confused.png` | Head tilt (3 frames) | 600x200 |
| `cat-L0-celebrating` | `cat/level0/celebrating.png` | Dance (6 frames) | 1200x200 |
| ... | `cat/level1/` through `cat/level5/` | Same 9 moods per level | Same dimensions |

**Total: 9 moods × 6 levels = 54 sprite sheets**

### Alternative: Lottie Animations (Recommended)

Instead of sprite sheets, use **Lottie JSON** animations for smoother, scalable, smaller-file animations.

| Asset ID | File | Description |
|---|---|---|
| `cat-L{n}-{mood}.json` | `cat/lottie/level{n}/{mood}.json` | Each mood as a Lottie animation |

**54 Lottie files total** — can be generated from After Effects or using AI tools like LottieFiles AI.

### Static Assets

| Asset | File | Use |
|---|---|---|
| Cat portrait (profile) | `cat/portrait.png` | Header avatar, session card |
| Speech bubble frame | `ui/speech-bubble.svg` | Dialogue container |
| Level badge template | `ui/level-badge.svg` | Overlay on cat avatar |
| Mushroom beret | `cat/accessories/beret.png` | Composited on cat head |
| Magnifying glass | `cat/accessories/magnifying-glass.png` | Level 1 accessory |
| Field notebook | `cat/accessories/notebook.png` | Level 2 accessory |
| Lab coat | `cat/accessories/labcoat.png` | Level 3 accessory |
| ID chart scroll | `cat/accessories/chart-scroll.png` | Level 4 accessory |
| Diploma + pin | `cat/accessories/diploma.png` | Level 5 accessory |

### Background & Scene Assets

| Asset | File | Use |
|---|---|---|
| Forest clearing (training bg) | `mushrooms/bg-forest.png` | Training arena background |
| Lab table (expert bg) | `mushrooms/bg-lab.png` | Expert mode background |
| Journey map (parchment) | `mushrooms/bg-journey-map.png` | Journey/progress page |
| Mushroom icons (edible set) | `mushrooms/icons/edible-{1-5}.svg` | UI decorations |
| Mushroom icons (poison set) | `mushrooms/icons/poison-{1-5}.svg` | UI decorations |
| Poison cloud effect | `ui/effects/poison-cloud.png` | Overlay when cat eats poison |
| Sparkle effect | `ui/effects/sparkle.png` | Correct prediction celebration |
| Confetti | `ui/effects/confetti.json` | Lottie confetti for level-up |

### Sound Effects (Optional Enhancement)

| Sound | File | Trigger |
|---|---|---|
| Meow (curious) | `sounds/meow-curious.mp3` | New mushroom presented |
| Meow (happy) | `sounds/meow-happy.mp3` | Correct prediction |
| Meow (sick) | `sounds/meow-sick.mp3` | False negative |
| Purr | `sounds/purr.mp3` | Idle state ambient |
| Level-up fanfare | `sounds/levelup.mp3` | Cat levels up |
| Munch | `sounds/munch.mp3` | Cat "eats" mushroom |
| Page turn | `sounds/page-turn.mp3` | Navigating journey map |

---

## AI Art Generation Prompts

Use these prompts with image generators (Midjourney, DALL-E 3, Stable Diffusion) to create consistent assets.

### Base Character Prompt (use as prefix for all)
```
Style: flat vector illustration, chibi anime proportions, thick black outlines,
pastel color palette (warm oranges, soft creams, muted greens).
Character: an adorable orange tabby cat with green eyes, wearing a tiny
mushroom-shaped beret (red cap with white spots). The cat has expressive
cartoon eyes, visible whiskers, and a fluffy tail. White background,
transparent PNG style. No text. Clean, simple, suitable for web app sprite.
```

### Per-Mood Prompts

**Idle:**
```
{base prompt} The cat is sitting relaxed, tail gently curved beside it, looking
forward with a calm neutral expression. Paws neatly together. The mushroom beret
sits slightly tilted on its head.
```

**Thinking:**
```
{base prompt} The cat is leaning forward with one paw on its chin, squinting
thoughtfully at something. One eyebrow raised. The pose suggests deep
concentration. A small thought bubble with a question mark floats above.
```

**Happy:**
```
{base prompt} The cat is jumping with joy, both front paws raised in the air,
eyes closed in a big smile, mouth open in a happy meow. Small sparkles around
the cat. The beret is bouncing slightly off its head.
```

**Sick (Poisoned):**
```
{base prompt} The cat is lying on its side with a green-tinted face, X-shaped
eyes (cartoon dizzy), tongue sticking out. Small mushroom crumbs near its mouth.
A green poison cloud hovers above. The beret has fallen off and lies nearby.
Expression is comically dramatic, not scary.
```

**Proud (Level Up):**
```
{base prompt} The cat is standing tall on its hind legs, chest puffed out,
eyes closed with a smug satisfied grin. The mushroom beret is tilted at a cool
angle. A golden glow/aura surrounds the cat. One paw is on its hip.
```

**Curious:**
```
{base prompt} The cat is crouched low, leaning forward with ears perked up
and eyes wide open. Its nose is sniffing forward, whiskers spread wide.
The tail is raised with a hook at the tip. The expression is excited discovery.
```

**Sleeping:**
```
{base prompt} The cat is curled into a ball, tail wrapped over its nose, eyes
peacefully closed. Three "Z" letters float above in increasing size. The
mushroom beret has slipped down over one ear. Soft, cozy feeling.
```

**Confused:**
```
{base prompt} The cat has its head tilted to one side at a 30-degree angle,
one ear up and one down, eyes slightly crossed. One paw is raised scratching
the back of its head. A large "?" floats beside it. The beret is askew.
```

**Celebrating (Journey Complete):**
```
{base prompt} The cat is dancing on its hind legs with a tiny party hat ON TOP
of its mushroom beret. Confetti falls around it. Eyes are sparkling with joy.
It's holding a tiny diploma in one paw and a golden mushroom trophy in the other.
```

### Per-Level Appearance Prompts (append to mood prompts)

**Level 0 — Clueless Kitten:**
```
Make the cat smaller and rounder (kitten proportions), with extra-large eyes
relative to head. The mushroom beret is oversized and keeps falling over one eye.
The cat looks endearingly incompetent.
```

**Level 1 — Curious Nibbler:**
```
The cat is slightly larger, the beret fits properly now. It carries a small
magnifying glass in one paw. Slightly more alert posture than level 0.
```

**Level 2 — Spore Spotter:**
```
Normal proportions. The cat carries a small field notebook tucked under one arm.
Its expression is more focused and attentive. Posture is straighter.
```

**Level 3 — Gill Inspector:**
```
The cat wears a tiny white lab coat over its fur. Expression is more serious
and confident. Slightly sharper eyes. The beret sits neatly. Professional vibe.
```

**Level 4 — Myco Apprentice:**
```
The cat looks slightly more muscular/confident. It carries a rolled-up mushroom
identification chart. Expression is determined. The lab coat has a small badge.
```

**Level 5 — Fungal Master:**
```
Full professor look — the cat wears reading glasses perched on its nose, the
lab coat is pristine, a golden mushroom pin on the lapel. It holds a framed
diploma. Expression is wise and serene. The beret is perfectly positioned.
A faint scholarly glow around the character.
```

---

## Cat Dialogue System

The cat speaks through a speech bubble. Dialogue is generated from templates + optional Gemini enhancement.

### Template Categories

**Greeting (session start):**
- Level 0: "Mew! I'm {name}... I don't know anything about mushrooms yet, but I'm VERY hungry. Teach me?"
- Level 3: "Ah, another specimen! Let me get my lab coat. I've been getting quite good at this."
- Level 5: "Welcome back. I've analyzed over {n} specimens with {acc}% accuracy. Shall we continue?"

**Prediction (before reveal):**
- Confident: "I'm fairly sure this one is {verdict}. The {feature} reminds me of previous samples."
- Uncertain: "Hmm... this is tricky. I'll say {verdict} but honestly I'm guessing. My confidence is only {conf}%."
- Very uncertain: "I have absolutely no idea. *sniffs nervously* ...{verdict}?"

**Correct prediction:**
- "Yes! I knew it! That {feature} was the giveaway!"
- "Another one right! I'm on a streak! ({correct}/{total})"
- "*purrs smugly* My training is paying off."

**Wrong prediction (false positive — said poison, was edible):**
- "Oh! It was actually safe? I was being too cautious. Better safe than sorry though!"
- "Hmm, I need to learn that not everything scary-looking is dangerous."

**Wrong prediction (false negative — said edible, was poisonous):**
- "*turns green* Oh no... that was POISONOUS?! I almost ate that!"
- "BLEGH! *dramatically falls over* Note to self: {feature} can be deceiving!"
- "*lying on side* I need to study more specimens before I trust that pattern again..."

**Wrong prediction (false positive — said poisonous, was edible):**
- "Oh! It was actually safe? I was being too cautious."
- "Hmm, I need to learn that not every {cap_color} cap is dangerous."
- "*embarrassed* I panicked. That was perfectly edible."

**Level up:**
- 0→1: "I can tell mushrooms apart from rocks now! That's... progress!"
- 1→2: "I got a notebook! Time to take proper notes on these specimens."
- 2→3: "They gave me a LAB COAT! I must be getting serious about this."
- 3→4: "I can identify gill patterns in my sleep now. Bring on the tough ones!"
- 4→5: "I... I did it. I'm a Fungal Master. *adjusts reading glasses* All those mushrooms were worth it."

**Picking next specimen (auto-training loop, per strategy):**
- Random: "Let me just grab a random mushroom from the forest..."
- Uncertainty: "I'll study the one I'm least sure about — it'll teach me the most."
- Committee: "My inner committee can't agree on this one. Let's settle it!"
- BALD: "Information theory says THIS one will teach me the most. *adjusts glasses*"

**Auto-training running (ambient, shown between rounds):**
- "Studying specimen #{n}..."
- "I've analyzed {labeled} mushrooms so far. Getting sharper!"
- "{accuracy}% accuracy — {level_comment}"  (e.g. "Not bad for a kitten!" / "I'm really improving!")

---

## Animation State Machine

### Training Mode (Auto-loop)

```
  [Pause >2min]
       │
  ┌────▼────────┐   [Start pressed]     ┌─────────────┐
  │   sleeping  │◄────────────────── ── │    idle     │◄─── [Pause pressed]
  └────┬────────┘                       └──────┬──────┘
       │ [Start pressed]                       │ [Start pressed]
       └───────────────────────────────────────┘
                                               │
                                        ┌──────▼──────┐
                              ┌─────────│   curious   │◄──────────────────┐
                              │         └──────┬──────┘   (loop repeats)  │
                              │  0.3–1.2s      │                           │
                              │         ┌──────▼──────┐                   │
                              │         │   thinking  │ 0.5–2s (predict)  │
                              │         └──────┬──────┘                   │
                              │          auto-reveal                       │
                              │       ┌─────────────┐                     │
                              │       │ wasCorrect? │                     │
                              │       └──┬───────┬──┘                     │
                              │       yes│       │no                      │
                              │    ┌─────▼──┐ ┌──▼──────────────────┐    │
                              │    │ happy  │ │ trueLabel=poisonous? │    │
                              │    └─────┬──┘ └──┬──────────────┬───┘    │
                              │          │      yes│            no│       │
                              │          │    ┌────▼──┐    ┌─────▼────┐  │
                              │          │    │  sick │    │ confused │  │
                              │          │    └────┬──┘    └─────┬────┘  │
                              │          └─────────┴─────────────┘       │
                              │                    │ check XP / accuracy  │
                              │             ┌──────▼─────┐               │
                              │             │ level up?  ├─ yes ─► proud ─┤
                              │             └──────┬─────┘               │
                              │                    │ no                   │
                              └────────────────────┴─────────────────────┘
```

### Expert Mode (User uploads photo — unchanged)

```
  idle ──► [upload photo] ──► curious ──► thinking ──► happy / sick / confused ──► idle
```

### Asset Generation Priority

| Priority | Moods | Levels | Why |
|---|---|---|---|
| **P0** | `curious`, `thinking`, `happy`, `sick`, `confused` | 0 | Visible every training cycle |
| **P1** | `idle`, `proud` | 0 | Paused state + level-up |
| **P2** | `curious`, `thinking`, `happy`, `sick`, `confused` | 1, 2 | Early leveling |
| **P3** | All 9 moods | 3, 4, 5 | Advanced users |
| **P4** | `sleeping`, `celebrating` | 0 | Low-frequency edge cases |

**Recommended initial batch: Level 0 all 9 moods = 9 sprites**
