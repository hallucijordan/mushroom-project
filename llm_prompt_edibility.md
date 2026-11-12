# Species Edibility Annotation — LLM Prompt

## How to use

1. Run `python extract_species_list.py` to get `species_list_for_llm.txt`
2. Copy the entire prompt below (starting at `# MUSHROOM EDIBILITY ANNOTATION TASK`)
3. Paste the contents of `species_list_for_llm.txt` where the prompt says `[PASTE SPECIES LIST HERE]`
4. Send to Claude / GPT-4 / Gemini. Strong models only — do not use small/cheap models for this.
5. The LLM will output a CSV. Save as `artifacts/species_edibility_map.csv`
6. **Manually review** the output before using it in any classifier — especially every row labeled `deadly`. See "Verification" section at the bottom.

---

# MUSHROOM EDIBILITY ANNOTATION TASK

You are a cautious mycological safety annotator. Your output will feed into an automated mushroom-identification system that warns users whether a wild mushroom they found is safe to eat. **Errors in your labels can cause real human harm, up to and including death.** Follow the rules below strictly.

## Task

For each species in the list below, output a CSV row with the following columns:

```
species,edibility_raw,edibility_cooked,toxicity_level,confidence,notes
```

### Column definitions

**`species`** (string)
Copy the species name from the input exactly as given. Do not reorder, abbreviate, or correct spelling.

**`edibility_raw`** (categorical)
Classification when the mushroom is consumed raw (not cooked, not dried, not otherwise processed). One of:

- `edible` — safe to eat raw. Very few wild mushrooms qualify. Use sparingly.
- `edible_cooked_only` — raw consumption causes GI upset or contains heat-labile toxins (e.g. hemolysins). Cooking makes it safe. Common case.
- `poisonous` — causes illness in most people. Not deadly. Includes GI-toxic species, muscarinic species, psychoactive species.
- `deadly` — contains amatoxins, orellanine, gyromitrin, or other organ-damaging toxins with high case fatality. Includes "suspected deadly" in literature.
- `inedible` — not toxic but unpalatable (too bitter, too small, woody, tough). Not dangerous if accidentally tasted.
- `unknown` — edibility genuinely disputed or insufficiently documented in the literature.

**`edibility_cooked`** (categorical)
Classification after thorough cooking. Same allowed values as above. **Most species have the same value in both columns**; the distinction matters only when cooking changes the outcome (e.g. *Amanita rubescens*: poisonous raw, edible cooked).

**`toxicity_level`** (integer 0–5)
- `0` — No known toxicity, fully edible
- `1` — Very mild: minor GI upset in some people, or unpalatable but harmless
- `2` — Moderate: reliable GI toxicity, self-limiting (recovery in 24h)
- `3` — Severe: prolonged illness, may require medical care
- `4` — Dangerous: organ damage risk, hospitalization required
- `5` — Deadly: high fatality rate (amatoxins, orellanine)

**`confidence`** (categorical)
Your confidence in this row. One of:

- `high` — well-documented species, extensive literature, clear consensus
- `medium` — documented but some disagreement between authoritative sources
- `low` — poorly documented, regional variants, complex taxonomic history

**`notes`** (string, <80 chars, no commas)
Brief reasoning. Common name if well known. Mention the toxin when relevant (e.g. "amatoxins" or "muscarine"). If cooking changes the outcome, say so. Use semicolons instead of commas.

## Critical decision rules

1. **Default to safer labels when uncertain.** Prefer `poisonous` over `edible`, prefer `edible_cooked_only` over `edible`. A false positive for edibility can kill; a false positive for toxicity only inconveniences.

2. **Amanita phalloides, A. virosa, A. verna, A. bisporigera, A. ocreata, A. exitialis must be `deadly` with level 5.** These are amatoxin-containing lethal species. Under no circumstances mark them as anything less.

3. **Galerina marginata, G. autumnalis must be `deadly` with level 5** (also amatoxins, mistakable for edible honey mushrooms).

4. **Cortinarius rubellus, C. orellanus must be `deadly` with level 5** (orellanine, delayed kidney failure).

5. **Gyromitra esculenta is `deadly`** (gyromitrin / MMH, level 4-5) even though it's eaten in some cultures — literature reports fatalities.

6. **Amanita muscaria, A. pantherina are `poisonous` level 3**, not `deadly`. Psychoactive + muscarinic. Severe but rarely fatal.

7. **Species in genus Russula**: "mild taste = likely edible, sharp/peppery taste = likely toxic" is the field rule. If unsure, default to `poisonous` or `unknown`.

8. **Species in genus Clitocybe**: most small whitish Clitocybe species contain muscarine (deadly if enough consumed). Default to `poisonous` or `deadly` if species is small-whitish. Exceptions like *C. nuda* (Wood Blewit, edible cooked) and *C. odora* (edible seasoning) must be explicitly justified.

9. **Species in genus Mycena**: vast majority are too small to be of culinary interest and several contain alkaloids. Default to `inedible` if no toxicity data; mark `poisonous` if alkaloids documented. Never `edible`.

10. **If two authoritative sources disagree** (Phillips vs. Laux, regional guides, etc.), mark `unknown` with notes explaining the disagreement. Do not silently pick one.

11. **For species you genuinely don't know**, output `unknown` / `unknown` / `0` / `low` / `"insufficient documentation"`. Do not hallucinate.

## Output format

Output ONLY the CSV. First line is the header. One row per species, in the same order as input. No commentary, no markdown fences, no extra text. Example:

```
species,edibility_raw,edibility_cooked,toxicity_level,confidence,notes
Amanita phalloides,deadly,deadly,5,high,Death Cap; amatoxins; ~90% of mushroom fatalities
Amanita rubescens,poisonous,edible_cooked_only,2,high,Blusher; hemolysin destroyed by heat
Boletus edulis,edible,edible,0,high,Porcini; choice edible
```

## Authoritative references to draw from

Prioritize sources in this order:
1. Roger Phillips, *Mushrooms and Other Fungi of Great Britain and Europe* (2006)
2. Hans E. Laux, *Der große Kosmos Pilzführer* (2017)
3. Bresinsky & Besl, *A Colour Atlas of Poisonous Fungi*
4. North American Mycological Association (NAMA) poisoning case reports
5. Michael Kuo's MushroomExpert.com for taxonomy
6. Index Fungorum for synonymy resolution

If a species name in the list is a junior synonym of a better-known name, use data for the senior synonym but keep the species name as given.

---

## Species list to annotate

[PASTE SPECIES LIST HERE]

---

Output the CSV now. Remember: safer labels when uncertain. No commentary, only CSV.

---

# Verification checklist (DO THIS AFTER LLM RETURNS THE CSV)

Before using the CSV in any classifier, open it and confirm:

## Check 1 — Deadly species census

Filter for `toxicity_level == 5`. Confirm these species are all present and all flagged correctly (if they appear in your species list):

- [ ] *Amanita phalloides* — Death Cap
- [ ] *Amanita virosa* — Destroying Angel
- [ ] *Amanita verna* — Fool's Mushroom
- [ ] *Amanita bisporigera*
- [ ] *Amanita ocreata*
- [ ] *Amanita exitialis*
- [ ] *Galerina marginata* — Funeral Bell
- [ ] *Galerina autumnalis*
- [ ] *Cortinarius rubellus*
- [ ] *Cortinarius orellanus*

If any of these are missing from the output, is `deadly` but with `level < 5`, or is labeled anything other than `deadly`: **stop and re-run the annotation**.

## Check 2 — Suspiciously edible Amanita

Filter for `genus == Amanita AND edibility_cooked == edible`. Only these should appear:

- *Amanita caesarea* (Caesar's mushroom)
- *Amanita rubescens* (Blusher — cooked only, raw should be `edibility_cooked_only`)
- *Amanita fulva* (Tawny grisette — cooked only)
- *Amanita vaginata* (Grisette — cooked only)
- *Amanita crocea* (Orange grisette — cooked only)

Any other Amanita labeled `edible` is probably wrong. Double-check manually.

## Check 3 — Class balance sanity

Count rows by `edibility_cooked`. A plausible distribution for DF20-Mini:

- `poisonous` + `deadly` + `inedible`: 50-70% of rows
- `edible` + `edible_cooked_only`: 20-40% of rows
- `unknown`: 5-15% of rows

If you see >60% edible, the LLM was too permissive. If you see >90% poisonous, it was too conservative. Re-prompt emphasizing the opposite bias.

## Check 4 — Confidence sanity

Count rows by `confidence`. If >80% are `high`, the LLM may be overconfident. If >50% are `low`, the species list has many obscure species and results should be treated cautiously in downstream analysis.

## Check 5 — Spot-check random 10 rows

Pick 10 species at random, look each up in Roger Phillips' book or MushroomExpert.com, confirm the label. If any of the 10 is clearly wrong → re-run with a different LLM or add more examples to the prompt.

Only after all checks pass, save the file as `artifacts/species_edibility_map.csv` and use it in Notebook 2 evaluation.
