#!/usr/bin/env python3
"""
interactive_umap.py
====================
Generates an interactive Plotly HTML UMAP visualization of the
747-molecule chemical space (REQ-5).

Outputs
-------
digital_discovery_manuscript/figures/interactive_umap.html

Usage
-----
    source /home/tchapet/VirtualEnv/bin/activate
    python code/interactive_umap.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / 'CALCULATIONS-MADE' / 'data_processing'
SMILES_F = ROOT / 'SMILES_molecules.csv'
OUT_FIG  = ROOT / 'digital_discovery_manuscript' / 'figures'
OUT_FIG.mkdir(exist_ok=True)

# ── Feature set for UMAP embedding ───────────────────────────────────────────
FEATURES_UMAP = [
    'S1_energy_eV', 'T1_energy_eV', 'Delta_E_ST_eV',
    'S1_S_he', 'T1_S_he', 'Delta_S_he',
    'S1_CT_number', 'T1_CT_number',
    'S1_Lambda_D', 'S1_Lambda_A', 'T1_Lambda_D', 'T1_Lambda_A',
    'S1_Delta_r', 'T1_Delta_r',
    'Char_diff_squared', 'S_NTO_sum',
    'S1_osc_strength', 'HOMO_LUMO_gap_eV',
]

# ── Pareto-optimal candidates ─────────────────────────────────────────────────
PARETO_CANDIDATES = {
    'DMAC-DPS': {'color': '#FFD700', 'symbol': 'star', 'size': 18,
                 'note': 'Soret band (425 nm), Ω=0.99'},
    'PXZ-NAI':  {'color': '#FF6B35', 'symbol': 'star', 'size': 18,
                 'note': 'Q_y band (629 nm)'},
    'BACN':     {'color': '#9B59B6', 'symbol': 'star', 'size': 18,
                 'note': 'High spectral overlap'},
}

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading feature matrix...')
df = pd.read_csv(DATA_DIR / 'combined_features_747mol_full_ct.csv')
df = df[(df['environment'] == 'gas') & (df['method'] == 'stda')].copy()
df = df.dropna(subset=FEATURES_UMAP)
print(f'  Dataset: {len(df)} molecules')

# Load SMILES
smiles_df = pd.read_csv(SMILES_F)
smiles_df.columns = ['molecule', 'SMILES']
df = df.merge(smiles_df, on='molecule', how='left')

# ── UMAP embedding ────────────────────────────────────────────────────────────
print('Computing UMAP embedding...')
X = df[FEATURES_UMAP].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    metric='euclidean', random_state=42)
embedding = reducer.fit_transform(X_scaled)
df['UMAP_1'] = embedding[:, 0]
df['UMAP_2'] = embedding[:, 1]
print('  UMAP embedding complete')

# ── Classify molecules ────────────────────────────────────────────────────────
df['is_pareto'] = df['molecule'].isin(PARETO_CANDIDATES.keys())
df['is_tadf']   = df['Delta_E_ST_eV'] < 0.2
df['category']  = 'Other'
df.loc[df['is_tadf'], 'category'] = 'TADF candidate (ΔE_ST < 0.2 eV)'
df.loc[df['is_pareto'], 'category'] = 'Pareto-optimal'

# ── Build hover text ──────────────────────────────────────────────────────────
df['hover'] = (
    '<b>' + df['molecule'] + '</b><br>' +
    'ΔE_ST = ' + df['Delta_E_ST_eV'].round(3).astype(str) + ' eV<br>' +
    'S_he(T₁) = ' + df['T1_S_he'].round(3).astype(str) + '<br>' +
    'S_he(S₁) = ' + df['S1_S_he'].round(3).astype(str) + '<br>' +
    'E(S₁) = ' + df['S1_energy_eV'].round(3).astype(str) + ' eV<br>' +
    'E(T₁) = ' + df['T1_energy_eV'].round(3).astype(str) + ' eV<br>' +
    'SMILES: ' + df['SMILES'].fillna('N/A').str[:40] + '...'
)

# ── Build Plotly figure ───────────────────────────────────────────────────────
print('Building interactive Plotly figure...')

fig = go.Figure()

# Layer 1: All non-Pareto molecules, colored by ΔE_ST
df_bg = df[~df['is_pareto']].copy()
fig.add_trace(go.Scatter(
    x=df_bg['UMAP_1'], y=df_bg['UMAP_2'],
    mode='markers',
    marker=dict(
        size=6,
        color=df_bg['Delta_E_ST_eV'],
        colorscale='Viridis_r',
        cmin=0.0, cmax=1.0,
        colorbar=dict(
            title=dict(text='ΔE_ST (eV)', side='right'),
            thickness=12, len=0.7,
            tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ),
        opacity=0.75,
        line=dict(width=0),
    ),
    text=df_bg['hover'],
    hovertemplate='%{text}<extra></extra>',
    name='All molecules',
    showlegend=True,
))

# Layer 2: Pareto-optimal candidates (gold stars)
for mol_name, props in PARETO_CANDIDATES.items():
    row = df[df['molecule'] == mol_name]
    if len(row) == 0:
        continue
    fig.add_trace(go.Scatter(
        x=row['UMAP_1'], y=row['UMAP_2'],
        mode='markers+text',
        marker=dict(
            size=props['size'],
            color=props['color'],
            symbol=props['symbol'],
            line=dict(color='black', width=1.5),
        ),
        text=[mol_name],
        textposition='top center',
        textfont=dict(size=11, color='black', family='Arial Black'),
        hovertemplate=(
            f'<b>{mol_name}</b><br>'
            f'{props["note"]}<br>'
            f'ΔE_ST = {row["Delta_E_ST_eV"].values[0]:.3f} eV<br>'
            f'S_he(T₁) = {row["T1_S_he"].values[0]:.3f}'
            '<extra></extra>'
        ),
        name=f'★ {mol_name}',
        showlegend=True,
    ))

# ── Layout ────────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text='<b>TADF Chemical Space — UMAP Projection</b><br>'
             '<sup>747 donor–acceptor molecules | colored by ΔE_ST | hover for details</sup>',
        x=0.5, xanchor='center', font=dict(size=16)
    ),
    xaxis=dict(title='UMAP dimension 1', showgrid=True, gridcolor='#EEEEEE',
               zeroline=False),
    yaxis=dict(title='UMAP dimension 2', showgrid=True, gridcolor='#EEEEEE',
               zeroline=False),
    plot_bgcolor='white',
    paper_bgcolor='white',
    legend=dict(
        title='<b>Legend</b>',
        x=0.01, y=0.99, xanchor='left', yanchor='top',
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#CCCCCC', borderwidth=1,
        font=dict(size=11),
    ),
    width=900, height=650,
    margin=dict(l=60, r=60, t=100, b=60),
    font=dict(family='Arial, sans-serif', size=12),
    hoverlabel=dict(bgcolor='white', font_size=12, font_family='Arial'),
)

# ── Add annotation box ────────────────────────────────────────────────────────
fig.add_annotation(
    text=(
        '<b>Key metrics:</b><br>'
        '• 747 molecules screened<br>'
        '• 27.7× computational reduction<br>'
        '• 45.5% Pareto efficiency gain<br>'
        '• ★ = Pareto-optimal candidates<br>'
        '<br>'
        '<i>Data: Zenodo 10.5281/zenodo.14241084</i>'
    ),
    xref='paper', yref='paper',
    x=0.99, y=0.01,
    xanchor='right', yanchor='bottom',
    showarrow=False,
    font=dict(size=10, color='#444444'),
    bgcolor='rgba(245,245,245,0.9)',
    bordercolor='#CCCCCC',
    borderwidth=1,
    borderpad=8,
    align='left',
)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = OUT_FIG / 'interactive_umap.html'
fig.write_html(
    str(out_path),
    include_plotlyjs='cdn',   # smaller file; requires internet to view
    full_html=True,
    config={'displayModeBar': True, 'scrollZoom': True,
            'toImageButtonOptions': {'format': 'png', 'filename': 'tadf_umap',
                                     'height': 800, 'width': 1200, 'scale': 2}},
)
print(f'  Saved: {out_path}')
print(f'  File size: {out_path.stat().st_size / 1024:.0f} KB')

print('\n✅ REQ-5b complete: interactive UMAP visualization generated')
