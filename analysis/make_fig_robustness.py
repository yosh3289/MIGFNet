#!/usr/bin/env python
"""Figure: 7-scenario robustness (5-fold mean Ranking Score), Bare vs MIGF-A2,
two panels (nnU-Net, Mamba) showing the backbone-conditional pattern under degradation."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'workspace', 'workspace', 'stats_recompute')
SCEN = ['ideal','missing_t2w','missing_hbv','missing_adc','artifact_t2w','artifact_hbv','artifact_adc']
LBL  = ['Ideal','miss\nT2W','miss\nHBV','miss\nADC','artif\nT2W','artif\nHBV','artif\nADC']

def load(pat):
    fs = glob.glob(os.path.join(BASE, '**', 'vast_5fold', 'results', pat), recursive=True)
    seen = {}
    for f in fs: seen.setdefault(os.path.basename(f), f)
    acc = {s: [] for s in SCEN}
    for f in seen.values():
        sc = json.load(open(f)).get('scenarios', {})
        for s in SCEN:
            v = sc.get(s, {}).get('ranking_score')
            if v is not None: acc[s].append(v)
    return np.array([np.mean(acc[s]) for s in SCEN])

bn = load('bench_nnunet_fold*');      mn = load('ablation_nnunet_A2_*')
bm = load('bench_mamba_fold*');       mm = load('migf_mamba_A2_fold*')

C_BARE, C_MIGF = '#7570b3', '#1b9e77'
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})
ang = np.linspace(0, 2*np.pi, len(SCEN), endpoint=False)
ang = np.concatenate([ang, ang[:1]])

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), subplot_kw=dict(polar=True))
for ax, (bare, migf, title) in zip(axes, [(bn, mn, '(a) nnU-Net'), (bm, mm, '(b) Mamba')]):
    for vals, c, lab in [(bare, C_BARE, 'Bare'), (migf, C_MIGF, 'MIGF (A2)')]:
        v = np.concatenate([vals, vals[:1]])
        ax.plot(ang, v, '-o', color=c, lw=1.6, ms=3.5, label=lab)
        ax.fill(ang, v, color=c, alpha=0.12)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(LBL, fontsize=7.5)
    ax.set_ylim(0.40, 0.78); ax.set_yticks([0.5, 0.6, 0.7])
    ax.set_yticklabels(['0.5','0.6','0.7'], fontsize=7)
    ax.set_title(title, fontsize=10, pad=14)
    ax.grid(alpha=0.4)

axes[0].legend(loc='upper center', bbox_to_anchor=(1.12, -0.08), ncol=2, frameon=False, fontsize=9)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'figures', 'Figure_robustness.pdf')
fig.savefig(out, bbox_inches='tight', dpi=300)
print('saved', out)
for nm, b, m in [('nnUNet', bn, mn), ('Mamba', bm, mm)]:
    print(nm, 'MIGF-Bare delta:', np.round(m-b, 3))
