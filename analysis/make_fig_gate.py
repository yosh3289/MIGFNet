#!/usr/bin/env python
"""Figure: gate mechanism -- static prior (nnU-Net) vs per-sample routing (Mamba).
(a) entry-stage modality prior weights; (b) across-case variability of gate
weights (the static-vs-routing evidence); (c) counterfactual mean-gate delta.
Run with mar conda env."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
W = json.load(open(os.path.join(HERE, 'gate_weights_local.json')))           # entry weights + across-case SD
G = json.load(open(os.path.join(HERE, '..', '..', 'workspace', 'workspace',
    'gate_analysis_archive', 'workspace', 'vast_5fold', 'audit_dumps',
    'gate_static_prior_analysis.json')))['rows']                              # counterfactual deltas

def wcol(fam, key): return np.array([x[key] for x in W if x['family'] == fam])
def gcol(fam, key): return np.array([x[key] for x in G if x['family'] == fam])

C_NN, C_MB = '#2c7fb8', '#d95f0e'
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})
rng = np.random.default_rng(0)
fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.7))

# ---- (a) modality prior weights ----
a = ax[0]
nn_w = wcol('nnunet', 'entry_mean').mean(0)        # [3]
mb_w = wcol('mamba',  'entry_mean').mean(0)
nn_e = wcol('nnunet', 'entry_casesd').mean(0)      # across-case SD as error
mb_e = wcol('mamba',  'entry_casesd').mean(0)
x = np.arange(3); wbar = 0.36
a.bar(x-wbar/2, nn_w, wbar, yerr=nn_e, color=C_NN, label='nnU-Net', capsize=2, error_kw=dict(lw=1))
a.bar(x+wbar/2, mb_w, wbar, yerr=mb_e, color=C_MB, label='Mamba', capsize=2, error_kw=dict(lw=1))
a.set_xticks(x); a.set_xticklabels(['T2W','HBV','ADC'])
a.set_ylabel('Entry-stage gate weight'); a.set_ylim(0, 0.62)
a.set_title('(a) Modality prior', fontsize=9.5)
a.legend(fontsize=7.5, frameon=False, loc='upper right')

# ---- (b) across-case variability (static vs routing) ----
b = ax[1]
nn_sd = wcol('nnunet', 'entry_casesd').mean(1)     # per-ckpt overall across-case SD
mb_sd = wcol('mamba',  'entry_casesd').mean(1)
b.scatter(0+rng.uniform(-0.08,0.08,len(nn_sd)), nn_sd, s=30, c=C_NN, edgecolor='k', lw=0.4, zorder=3)
b.scatter(1+rng.uniform(-0.08,0.08,len(mb_sd)), mb_sd, s=30, c=C_MB, edgecolor='k', lw=0.4, zorder=3)
for xx, v, c in [(0, nn_sd, C_NN), (1, mb_sd, C_MB)]:
    b.hlines(v.mean(), xx-0.18, xx+0.18, color=c, lw=2, zorder=4)
b.set_xticks([0,1]); b.set_xticklabels(['nnU-Net','Mamba'])
b.set_ylabel('Gate across-case SD'); b.set_xlim(-0.45, 1.45); b.set_ylim(0, 0.058)
b.set_title('(b) Per-sample variability', fontsize=9.5)
b.annotate('', xy=(0.5, mb_sd.mean()), xytext=(0.5, nn_sd.mean()),
           arrowprops=dict(arrowstyle='<->', color='0.4', lw=1))
b.text(0.56, (nn_sd.mean()+mb_sd.mean())/2, r'$\approx11\times$', fontsize=9, color='0.3')
b.text(0.0, nn_sd.mean()+0.004, 'static', fontsize=7.5, ha='center', color=C_NN)
b.text(1.0, mb_sd.mean()+0.006, 'routing', fontsize=7.5, ha='center', color=C_MB)

# ---- (c) counterfactual mean-gate delta ----
c = ax[2]
nn_d = gcol('nnunet', 'delta_meangate_ranking_score')
mb_d = gcol('mamba',  'delta_meangate_ranking_score')
c.axhline(0, color='0.6', lw=0.8, zorder=1)
c.scatter(0+rng.uniform(-0.08,0.08,len(nn_d)), nn_d, s=30, c=C_NN, edgecolor='k', lw=0.4, zorder=3)
c.scatter(1+rng.uniform(-0.08,0.08,len(mb_d)), mb_d, s=30, c=C_MB, edgecolor='k', lw=0.4, zorder=3)
for xx, v, cc in [(0, nn_d, C_NN), (1, mb_d, C_MB)]:
    c.hlines(v.mean(), xx-0.18, xx+0.18, color=cc, lw=2, zorder=4)
c.set_xticks([0,1]); c.set_xticklabels(['nnU-Net','Mamba'])
c.set_ylabel(r'$\Delta$Ranking Score'); c.set_xlim(-0.45, 1.45)
c.set_title('(c) Mean-gate counterfactual', fontsize=9.5)
c.text(0.0, max(nn_d)+0.012, '~0', fontsize=8, ha='center', color=C_NN)
c.text(1.0, max(mb_d)+0.004, 'degrades', fontsize=7.5, ha='center', color=C_MB)

fig.tight_layout()
out = os.path.join(HERE, 'figures', 'Figure_gate_mechanism.pdf')
fig.savefig(out, bbox_inches='tight', dpi=300)
print('saved', out)
print(f"nnUNet across-case SD {nn_sd.mean():.4f}+-{nn_sd.std(ddof=1):.4f} | Mamba {mb_sd.mean():.4f}+-{mb_sd.std(ddof=1):.4f} | {mb_sd.mean()/nn_sd.mean():.1f}x")
