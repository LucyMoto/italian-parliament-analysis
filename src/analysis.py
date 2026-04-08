"""
Analysis pipeline: dimensionality reduction, clustering, and export.
Run: python src/analysis.py

Input:  data/processed/*.parquet
Output: outputs/deputy_graph.json  (feeds the dashboard)
        outputs/analysis_summary.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time

# ── Config ────────────────────────────────────────────────────
try:
    from config import PROCESSED_DIR, OUTPUTS_DIR
except ImportError:
    PROCESSED_DIR = Path('data/processed')
    OUTPUTS_DIR = Path('outputs')

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Major groups for focused analysis
MAJOR_GROUPS = ['FDI', 'PD-IDP', 'LEGA', 'M5S', 'FI-PPE', 'AVS', 'NM-M-C', 'AZIONE-IV']

PARTY_COLORS = {
    'FDI': '#003DA5',
    'PD-IDP': '#E2001A',
    'LEGA': '#008C45',
    'M5S': '#FFD700',
    'FI-PPE': '#0077CC',
    'AVS': '#6B8E23',
    'NM-M-C': '#9B59B6',
    'AZIONE-IV': '#FF6B35',
    'M-ALT': '#888888',
    'M-MIN': '#AAAAAA',
    'M-+EUR': '#666666',
    '------': '#CCCCCC',
}

PARTY_LABELS = {
    'FDI': 'Fratelli d\'Italia',
    'PD-IDP': 'Partito Democratico',
    'LEGA': 'Lega',
    'M5S': 'Movimento 5 Stelle',
    'FI-PPE': 'Forza Italia',
    'AVS': 'Alleanza Verdi e Sinistra',
    'NM-M-C': 'Noi Moderati',
    'AZIONE-IV': 'Azione / Italia Viva',
    'M-ALT': 'Misto - Altro',
    'M-MIN': 'Misto - Minoranze linguistiche',
    'M-+EUR': 'Misto - +Europa',
    '------': 'Misto',
}

COALITION_GROUPS = {'FDI', 'LEGA', 'FI-PPE', 'NM-M-C'}
OPPOSITION_GROUPS = {'PD-IDP', 'M5S', 'AVS'}


# ── 1. Load data ──────────────────────────────────────────────

print('Loading data...')
votes_df = pd.read_parquet(PROCESSED_DIR / 'votes_clean.parquet')
vote_matrix = pd.read_parquet(PROCESSED_DIR / 'vote_matrix.parquet')
similarity = pd.read_parquet(PROCESSED_DIR / 'similarity_matrix.parquet')
deputies_df = pd.read_parquet(PROCESSED_DIR / 'deputies.parquet')

deputy_group = dict(zip(deputies_df['deputy_name'], deputies_df['group']))
all_deputies = vote_matrix.index.tolist()

print(f'  {len(all_deputies)} deputies')
print(f'  {vote_matrix.shape[1]} votes')


# ── 2. Compute deputy stats ──────────────────────────────────

print('Computing deputy stats...')

# Attendance and absence rates
dep_stats = (
    votes_df.groupby(['deputy_name', 'group_clean'])
    .agg(
        total=('vote_code', 'count'),
        voted=('voted', 'sum'),
        absent=('absent', 'sum'),
        on_mission=('on_mission', 'sum'),
    )
    .reset_index()
)
dep_stats['attendance_pct'] = (dep_stats['voted'] / dep_stats['total'] * 100).round(1)
dep_stats['absent_pct'] = (dep_stats['absent'] / dep_stats['total'] * 100).round(1)

# Use the group with most votes for each deputy (handles switchers)
dep_stats = dep_stats.sort_values('total', ascending=False).drop_duplicates('deputy_name', keep='first')

# Defection rate
df_voted = votes_df[votes_df['voted']].copy()
party_majority = (
    df_voted.groupby(['group_clean', 'vote_id'])['vote_code']
    .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
    .reset_index()
    .rename(columns={'vote_code': 'party_line'})
)
df_def = df_voted.merge(party_majority, on=['group_clean', 'vote_id'], how='left')
df_def['defected'] = df_def['vote_code'] != df_def['party_line']

dep_defection = (
    df_def.groupby('deputy_name')
    .agg(votes_cast=('defected', 'count'), defections=('defected', 'sum'))
    .reset_index()
)
dep_defection['defection_pct'] = (
    dep_defection['defections'] / dep_defection['votes_cast'] * 100
).round(2)

# Merge stats
dep_stats = dep_stats.merge(
    dep_defection[['deputy_name', 'votes_cast', 'defections', 'defection_pct']],
    on='deputy_name', how='left'
)


# ── 3. UMAP dimensionality reduction ─────────────────────────

print('Running UMAP...')
t0 = time.time()

from umap import UMAP

# Fill NaN in vote matrix with 0 (didn't participate = neutral)
vm_filled = vote_matrix.fillna(0).values

reducer = UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='correlation',
    random_state=42,
)
coords = reducer.fit_transform(vm_filled)

print(f'  UMAP done in {time.time()-t0:.1f}s')


# ── 4. Clustering ─────────────────────────────────────────────

print('Clustering with DBSCAN...')

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

coords_scaled = StandardScaler().fit_transform(coords)
clustering = DBSCAN(eps=0.5, min_samples=5).fit(coords_scaled)
labels = clustering.labels_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f'  Found {n_clusters} clusters, {n_noise} noise points')


# ── 5. Build edges (top connections per deputy) ───────────────

print('Building edge list...')

# For the network graph, we don't want all 413*412/2 edges
# Keep only strong connections (top N most similar per deputy)
TOP_N_EDGES = 5
SIM_THRESHOLD = 0.7  # minimum similarity to draw an edge

edges = []
seen = set()

for i, dep1 in enumerate(all_deputies):
    sims = similarity.loc[dep1].drop(dep1).sort_values(ascending=False)
    
    # Top N most similar
    for dep2, sim_val in sims.head(TOP_N_EDGES).items():
        if sim_val < SIM_THRESHOLD:
            continue
        edge_key = tuple(sorted([dep1, dep2]))
        if edge_key not in seen:
            seen.add(edge_key)
            edges.append({
                'source': dep1,
                'target': dep2,
                'weight': round(float(sim_val), 3),
            })

# Also add strong cross-party edges (interesting connections)
for i, dep1 in enumerate(all_deputies):
    g1 = deputy_group.get(dep1, '')
    sims = similarity.loc[dep1].drop(dep1)
    
    for dep2, sim_val in sims.items():
        g2 = deputy_group.get(dep2, '')
        if g1 != g2 and sim_val > 0.95:
            edge_key = tuple(sorted([dep1, dep2]))
            if edge_key not in seen:
                seen.add(edge_key)
                edges.append({
                    'source': dep1,
                    'target': dep2,
                    'weight': round(float(sim_val), 3),
                })

print(f'  {len(edges)} edges')


# ── 6. Export JSON ────────────────────────────────────────────

print('Exporting JSON...')

nodes = []
for i, dep_name in enumerate(all_deputies):
    group = deputy_group.get(dep_name, 'Unknown')
    stats_row = dep_stats[dep_stats['deputy_name'] == dep_name]
    
    node = {
        'id': dep_name,
        'group': group,
        'group_full': PARTY_LABELS.get(group, group),
        'color': PARTY_COLORS.get(group, '#999999'),
        'side': 'coalition' if group in COALITION_GROUPS else 'opposition' if group in OPPOSITION_GROUPS else 'other',
        'x': round(float(coords[i, 0]), 4),
        'y': round(float(coords[i, 1]), 4),
        'cluster': int(labels[i]),
        'attendance_pct': float(stats_row['attendance_pct'].iloc[0]) if len(stats_row) > 0 else 0,
        'absent_pct': float(stats_row['absent_pct'].iloc[0]) if len(stats_row) > 0 else 0,
        'defection_pct': float(stats_row['defection_pct'].iloc[0]) if len(stats_row) > 0 else 0,
        'votes_cast': int(stats_row['votes_cast'].iloc[0]) if len(stats_row) > 0 else 0,
        'defections': int(stats_row['defections'].iloc[0]) if len(stats_row) > 0 else 0,
    }
    nodes.append(node)

# Party summary
party_summary = []
for group in MAJOR_GROUPS:
    group_nodes = [n for n in nodes if n['group'] == group]
    if not group_nodes:
        continue
    party_summary.append({
        'group': group,
        'group_full': PARTY_LABELS.get(group, group),
        'color': PARTY_COLORS.get(group, '#999'),
        'side': 'coalition' if group in COALITION_GROUPS else 'opposition' if group in OPPOSITION_GROUPS else 'other',
        'n_deputies': len(group_nodes),
        'avg_attendance': round(np.mean([n['attendance_pct'] for n in group_nodes]), 1),
        'avg_defection': round(np.mean([n['defection_pct'] for n in group_nodes]), 2),
    })

graph_data = {
    'nodes': nodes,
    'edges': edges,
    'parties': party_summary,
    'clusters': {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
    },
    'meta': {
        'n_deputies': len(nodes),
        'n_edges': len(edges),
        'n_votes': int(vote_matrix.shape[1]),
        'date_range': f"{votes_df['date'].min().strftime('%d/%m/%Y')} — {votes_df['date'].max().strftime('%d/%m/%Y')}",
    }
}


# Convert numpy types for JSON serialization
def convert_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def clean_for_json(data):
    if isinstance(data, dict):
        return {k: clean_for_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [clean_for_json(v) for v in data]
    return convert_types(data)

graph_data = clean_for_json(graph_data)

out_path = OUTPUTS_DIR / 'deputy_graph.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(graph_data, f, ensure_ascii=False, indent=2, default=convert_types)

print(f'  Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)')


# ── 7. Summary ────────────────────────────────────────────────

summary_lines = [
    '=' * 50,
    'ANALYSIS COMPLETE',
    '=' * 50,
    '',
    f'Deputies:     {len(nodes)}',
    f'Edges:        {len(edges)}',
    f'Clusters:     {n_clusters} (+{n_noise} noise)',
    f'Votes:        {vote_matrix.shape[1]}',
    '',
    'UMAP coordinates computed (correlation metric)',
    'DBSCAN clustering applied',
    '',
    'Cluster composition:',
]

for c in range(n_clusters):
    c_nodes = [n for n in nodes if n['cluster'] == c]
    groups = {}
    for n in c_nodes:
        groups[n['group']] = groups.get(n['group'], 0) + 1
    groups_str = ', '.join(f"{g}: {c}" for g, c in sorted(groups.items(), key=lambda x: -x[1]))
    summary_lines.append(f'  Cluster {c} ({len(c_nodes)} deputies): {groups_str}')

noise_nodes = [n for n in nodes if n['cluster'] == -1]
if noise_nodes:
    groups = {}
    for n in noise_nodes:
        groups[n['group']] = groups.get(n['group'], 0) + 1
    groups_str = ', '.join(f"{g}: {c}" for g, c in sorted(groups.items(), key=lambda x: -x[1]))
    summary_lines.append(f'  Noise ({len(noise_nodes)} deputies): {groups_str}')

summary_text = '\n'.join(summary_lines)
print(summary_text)

summary_path = OUTPUTS_DIR / 'analysis_summary.txt'
with open(summary_path, 'w') as f:
    f.write(summary_text)

print(f'\nNext: python src/dashboard.py')
