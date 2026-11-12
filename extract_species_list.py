"""
Extract unique species from DF20-Mini metadata CSVs and emit a list ready
for LLM annotation.

Usage:
    python extract_species_list.py

Output:
    species_list.txt       — plain text, one "genus | species" per line,
                             grouped by genus, sorted alphabetically
    species_list_for_llm.txt — same list but formatted for direct paste into
                               the LLM prompt (numbered, no header)
"""
from pathlib import Path
import pandas as pd

# ---------- Adjust these paths if your layout differs ----------
METADATA_DIR   = Path('data/DF20M-metadata')
METADATA_TRAIN = METADATA_DIR / 'DF20M-train_metadata_PROD.csv'
METADATA_TEST  = METADATA_DIR / 'DF20M-public_test_metadata_PROD.csv'
OUTPUT_DIR     = Path('.')
# ----------------------------------------------------------------

# Load both CSVs
meta_train = pd.read_csv(METADATA_TRAIN)
meta_test  = pd.read_csv(METADATA_TEST)
meta = pd.concat([meta_train, meta_test], ignore_index=True)

# Use the 'species' column (standard binomial), same choice as in Notebook 1.
# If not present, fall back to scientificName.
species_col = 'species' if 'species' in meta.columns else 'scientificName'

# Per-species: genus + image count
df = (
    meta.dropna(subset=[species_col, 'genus'])
        .groupby([species_col, 'genus'])
        .size()
        .reset_index(name='n_images')
        .rename(columns={species_col: 'species'})
        .sort_values(['genus', 'species'])
        .reset_index(drop=True)
)

print(f"Total unique species: {len(df)}")
print(f"Genera: {df['genus'].nunique()}")
print(f"\nCount per genus:")
print(df['genus'].value_counts())

# ---- Output 1: human-readable grouped list ----
out_full = OUTPUT_DIR / 'species_list.txt'
with open(out_full, 'w') as f:
    f.write(f"# DF20-Mini species list\n")
    f.write(f"# Total: {len(df)} species across {df['genus'].nunique()} genera\n\n")
    for g in sorted(df['genus'].unique()):
        sub = df[df['genus'] == g]
        f.write(f"## {g} ({len(sub)} species)\n")
        for _, row in sub.iterrows():
            f.write(f"{row['species']}  ({row['n_images']} images)\n")
        f.write("\n")
print(f"\nWrote {out_full}")

# ---- Output 2: LLM-ready numbered list ----
out_llm = OUTPUT_DIR / 'species_list_for_llm.txt'
with open(out_llm, 'w') as f:
    for i, row in df.iterrows():
        f.write(f"{i+1}. {row['species']}\n")
print(f"Wrote {out_llm}")

print(f"\nFirst 5 species (preview):")
print(df.head().to_string(index=False))
