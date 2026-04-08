"""
Clean and process raw Camera dei Deputati voting data.
Run: python src/clean.py

Input:  data/raw/camera_votes_full.csv
Output: data/processed/votes_clean.parquet
        data/processed/vote_matrix.parquet
        data/processed/deputies.parquet
        data/processed/similarity_matrix.parquet
"""

import pandas as pd
import numpy as np
import time

from src.config import RAW_DIR, PROCESSED_DIR

RAW_FILE = RAW_DIR / 'camera_votes_full.csv'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load ──────────────────────────────────────────────────

print(f'Loading {RAW_FILE} ...')
t0 = time.time()
df = pd.read_csv(
    RAW_FILE,
    dtype={
        'legislature': 'int16',
        'session': 'int16',
        'vote_num': 'int16',
        'vote_type': 'category',
        'deputy_name': 'str',
        'group': 'category',
        'vote': 'category',
    }
)
print(f'Loaded {len(df):,} rows in {time.time()-t0:.1f}s')

# ── 2. Deduplicate ───────────────────────────────────────────

n_before = len(df)
df = df.drop_duplicates(subset=['session', 'vote_num', 'deputy_name'])
print(f'Dropped {n_before - len(df):,} duplicate rows')

# ── 3. Parse dates ───────────────────────────────────────────

df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
df['year_month'] = df['date'].dt.to_period('M')
df['year'] = df['date'].dt.year
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# ── 4. Encode votes ──────────────────────────────────────────

VOTE_MAP = {
    'Favorevole': 1,
    'Contrario': -1,
    'Astensione': 0,
    'Non ha partecipato': 9,
    'In missione': 7,
    'Presidente di turno': 8,
    'Ha votato': 6,
}

df['vote_code'] = df['vote'].map(VOTE_MAP)

unmapped = df[df['vote_code'].isna()]['vote'].unique()
if len(unmapped) > 0:
    print(f'WARNING: Unmapped vote values: {unmapped}')
else:
    print('All votes mapped successfully')

df['vote_code'] = df['vote_code'].fillna(99).astype('int8')

# Boolean flags
df['voted'] = df['vote_code'].isin([1, -1, 0])
df['absent'] = df['vote_code'] == 9
df['on_mission'] = df['vote_code'] == 7
df['presiding'] = df['vote_code'] == 8
df['secret_ballot'] = df['vote_code'] == 6

# Sanity check
check = df[['voted', 'absent', 'on_mission', 'presiding', 'secret_ballot']].sum(axis=1)
unexpected = (check != 1).sum()
if unexpected > 0:
    print(f'WARNING: {unexpected:,} rows not in exactly one category')
else:
    print(f'All {len(df):,} rows classified into exactly one category')

# ── 5. Vote ID ───────────────────────────────────────────────

df['vote_id'] = df['session'].astype(str) + '_' + df['vote_num'].astype(str)

# ── 6. Consolidate group names ───────────────────────────────

GROUP_MAP = {
    'AIV-RE': 'AZIONE-IV',
    'APERRE': 'AZIONE-IV',
    'IVICRE': 'AZIONE-IV',
    'M-AVS': 'AVS',
    'M-NM': 'NM-M-C',
    'FDI': 'FDI',
    'PD-IDP': 'PD-IDP',
    'LEGA': 'LEGA',
    'M5S': 'M5S',
    'FI-PPE': 'FI-PPE',
    'AVS': 'AVS',
    'NM-M-C': 'NM-M-C',
}

df['group_clean'] = df['group'].map(GROUP_MAP).fillna(df['group'].astype(str))
print(f"Groups: {df['group'].nunique()} original -> {df['group_clean'].nunique()} consolidated")

# ── 7. Classify vote types ───────────────────────────────────

vote_type_str = df['vote_type'].astype(str).str.lower()
desc_str = df['description'].astype(str).str.lower()

df['is_final_vote'] = vote_type_str.str.contains('finale', na=False)
df['is_confidence'] = vote_type_str.str.contains('fiducia', na=False)
df['is_amendment'] = (
    ~df['is_final_vote'] &
    ~df['is_confidence'] &
    vote_type_str.str.contains('nominale|appello', na=False)
)
df['is_substantive'] = df['is_final_vote'] | df['is_confidence'] | df['is_amendment']
df['is_odg'] = desc_str.str.contains('ordine del giorno', na=False)

n_substantive = df[df['is_substantive']]['vote_id'].nunique()
n_total = df['vote_id'].nunique()
print(f"Substantive votes: {n_substantive:,} / {n_total:,} ({n_substantive/n_total*100:.1f}%)")

# ── 8. Save cleaned long-format data ─────────────────────────

out_path = PROCESSED_DIR / 'votes_clean.parquet'
df.to_parquet(out_path, index=False)
print(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

# ── 9. Build vote matrix ─────────────────────────────────────

print('Building vote matrix...')
df_voted = df[df['voted']][['deputy_name', 'vote_id', 'vote_code']].copy()

vote_matrix = df_voted.pivot_table(
    index='deputy_name',
    columns='vote_id',
    values='vote_code',
    aggfunc='first'
)
print(f'Vote matrix: {vote_matrix.shape[0]} deputies x {vote_matrix.shape[1]} votes')

out_path = PROCESSED_DIR / 'vote_matrix.parquet'
vote_matrix.to_parquet(out_path)
print(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

# ── 10. Compute similarity matrix ────────────────────────────

print('Computing similarity matrix...')
similarity = vote_matrix.T.corr(method='pearson')

out_path = PROCESSED_DIR / 'similarity_matrix.parquet'
similarity.to_parquet(out_path)
print(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

# ── 11. Save deputy lookup ───────────────────────────────────

deputy_group = (
    df.groupby('deputy_name')['group_clean']
    .agg(lambda x: x.value_counts().index[0])
)
deputies_df = deputy_group.reset_index().rename(columns={'group_clean': 'group'})
out_path = PROCESSED_DIR / 'deputies.parquet'
deputies_df.to_parquet(out_path, index=False)
print(f"Saved {out_path}")

# ── Summary ──────────────────────────────────────────────────

print()
print('=' * 50)
print('CLEANING COMPLETE')
print('=' * 50)
print(f"  Total rows:       {len(df):>12,}")
print(f"  Unique votes:     {df['vote_id'].nunique():>12,}")
print(f"  Unique deputies:  {df['deputy_name'].nunique():>12,}")
print(f"  Unique groups:    {df['group_clean'].nunique():>12,}")
print()
print('  Files:')
for f in sorted(PROCESSED_DIR.iterdir()):
    print(f'    {f.name}: {f.stat().st_size / 1e6:.1f} MB')
print()
print('  Next: open notebooks/explore.ipynb')
