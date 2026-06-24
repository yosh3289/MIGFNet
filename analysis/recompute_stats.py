#!/usr/bin/env python
"""
Recompute exact statistics for Paper 1 CMPB from raw per-(fold,seed) JSONs.
No scipy (broken in env): exact Student-t p-values via incomplete beta;
Wilcoxon signed-rank (exact for small n, normal approx for n>=12);
95% CI on mean paired difference via inverse-t (bisection).
"""
import json, math, glob, os
from statistics import mean, stdev
from itertools import combinations

RESULTS = glob.glob('**/vast_5fold/results/*.json', recursive=True)
# dedup by basename: prefer any path; both archives have disjoint (fold,seed) so no real dup
_seen = {}
for _f in RESULTS:
    _seen.setdefault(os.path.basename(_f), _f)
RESULTS = list(_seen.values())

# ---------- incomplete beta -> Student t two-sided p ----------
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-12, 1e-30
    qab, qap, qam = a+b, a+1.0, a-1.0
    c = 1.0
    d = 1.0 - qab*x/qap
    if abs(d) < FPMIN: d = FPMIN
    d = 1.0/d; h = d
    for m in range(1, MAXIT+1):
        m2 = 2*m
        aa = m*(b-m)*x/((qam+m2)*(a+m2))
        d = 1.0+aa*d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0+aa/c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0/d; h *= d*c
        aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
        d = 1.0+aa*d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0+aa/c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0/d; de = d*c; h *= de
        if abs(de-1.0) < EPS: break
    return h

def betai(a, b, x):
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)
    bt = math.exp(lbeta + a*math.log(x) + b*math.log(1.0-x))
    if x < (a+1.0)/(a+b+2.0):
        return bt*_betacf(a, b, x)/a
    else:
        return 1.0 - bt*_betacf(b, a, 1.0-x)/b

def t_two_sided_p(t, df):
    if df <= 0: return float('nan')
    if math.isinf(t): return 0.0
    x = df/(df + t*t)
    return betai(df/2.0, 0.5, x)   # two-sided p

def t_cdf(t, df):
    p2 = t_two_sided_p(t, df)
    return 1.0 - p2/2.0 if t > 0 else p2/2.0

def t_ppf(q, df):
    lo, hi = -100.0, 100.0
    for _ in range(200):
        mid = (lo+hi)/2.0
        if t_cdf(mid, df) < q: lo = mid
        else: hi = mid
    return (lo+hi)/2.0

# ---------- Wilcoxon signed-rank ----------
def wilcoxon(diffs):
    d = [x for x in diffs if x != 0.0]
    n = len(d)
    if n == 0: return None
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0]*n
    i = 0
    while i < n:
        j = i
        while j+1 < n and abs(d[order[j+1]]) == abs(d[order[i]]): j += 1
        r = (i+j)/2.0 + 1.0
        for k in range(i, j+1): ranks[order[k]] = r
        i = j+1
    Wpos = sum(ranks[i] for i in range(n) if d[i] > 0)
    Wneg = sum(ranks[i] for i in range(n) if d[i] < 0)
    W = min(Wpos, Wneg)
    if n <= 12:  # exact enumeration over sign assignments
        from itertools import product
        absr = [abs(rr) for rr in ranks]
        count = 0; total = 2**n
        for signs in product([0,1], repeat=n):
            s = sum(absr[i] for i in range(n) if signs[i])
            if s <= W: count += 1
        p = min(1.0, 2.0*count/total)
        return {'W': W, 'n': n, 'p': p, 'method': 'exact'}
    else:
        mu = n*(n+1)/4.0
        sig = math.sqrt(n*(n+1)*(2*n+1)/24.0)
        z = (W - mu + 0.5)/sig  # continuity correction
        from math import erf
        p = 2*(0.5*(1+erf(-abs(z)/math.sqrt(2))))
        return {'W': W, 'n': n, 'p': min(1.0,p), 'z': z, 'method': 'normal-approx'}

# ---------- paired t + CI ----------
def paired_stats(diffs):
    n = len(diffs)
    if n < 2: return None
    m = mean(diffs); sd = stdev(diffs); se = sd/math.sqrt(n)
    if se == 0:
        return {'n': n, 'mean': m, 'sd': sd, 't': float('inf'), 'p': 0.0, 'ci': (m, m)}
    t = m/se; df = n-1
    p = t_two_sided_p(t, df)
    tc = t_ppf(0.975, df)
    return {'n': n, 'mean': m, 'sd': sd, 'se': se, 't': t, 'df': df, 'p': p,
            'ci': (m - tc*se, m + tc*se)}

# ---------- load data ----------
# family -> config -> (fold,seed) -> ideal metrics
def famkey(fname):
    b = os.path.basename(fname)
    if b.startswith('prostate158'): return None
    import re
    m = re.search(r'fold(\d)_seed(\d+)', b)
    if not m: return None
    fold, seed = int(m.group(1)), int(m.group(2))
    # map to (backbone, config)
    if b.startswith('bench_nnunet'): bb,cfg = 'nnunet','Bare'
    elif b.startswith('bench_mamba'): bb,cfg = 'mamba','Bare'
    elif b.startswith('bench_monai_unet'): bb,cfg = 'unet','Bare'
    elif b.startswith('migf_mamba_A2'): bb,cfg = 'mamba','A2'
    elif b.startswith('ablation_nnunet_'):
        cfg = b.split('ablation_nnunet_')[1].split('_')[0]; bb='nnunet'
    elif b.startswith('ablation_mamba_'):
        cfg = b.split('ablation_mamba_')[1].split('_')[0]; bb='mamba'
    elif b.startswith('ablation_unet_'):
        cfg = b.split('ablation_unet_')[1].split('_')[0]; bb='unet'
    else: return None
    return bb, cfg, (fold, seed)

data = {}
for f in RESULTS:
    k = famkey(f)
    if not k: continue
    bb, cfg, fs = k
    try: j = json.load(open(f))
    except: continue
    ideal = j.get('scenarios', {}).get('ideal')
    if not ideal: continue
    data.setdefault(bb, {}).setdefault(cfg, {})[fs] = {
        'ranking_score': ideal.get('ranking_score'),
        'case_specificity': ideal.get('case_specificity'),
        'case_sensitivity': ideal.get('case_sensitivity'),
        'dice': ideal.get('dice'),
        'num_positive': ideal.get('num_positive'),
    }

CONFIGS = ['Bare','A1','A2','A3','A4','A5','A6','Full']
# G/D/M membership
GDM = {'Bare':(0,0,0),'A1':(0,1,1),'A2':(1,0,1),'A3':(1,1,0),
       'A4':(1,0,0),'A5':(0,1,0),'A6':(0,0,1),'Full':(1,1,1)}

def cell_mean_sd(bb, cfg, metric):
    runs = data.get(bb,{}).get(cfg,{})
    vals = [v[metric] for v in runs.values() if v.get(metric) is not None]
    if not vals: return None
    return (mean(vals), stdev(vals) if len(vals)>1 else 0.0, len(vals))

def paired_vs_bare(bb, cfg, metric):
    a = data.get(bb,{}).get('Bare',{})
    b = data.get(bb,{}).get(cfg,{})
    common = sorted(set(a.keys()) & set(b.keys()))
    diffs = [b[k][metric]-a[k][metric] for k in common
             if a[k].get(metric) is not None and b[k].get(metric) is not None]
    return diffs

print("="*78)
print("VERIFIED 8-CELL ABLATION (ideal scenario, ddof=1)")
print("="*78)
for bb in ['nnunet','mamba']:
    print(f"\n### {bb.upper()} ###")
    print(f"{'cfg':<6}{'n':>4}{'Score m+/-sd':>18}{'CaseSpec m+/-sd':>20}")
    for cfg in CONFIGS:
        s = cell_mean_sd(bb,cfg,'ranking_score')
        c = cell_mean_sd(bb,cfg,'case_specificity')
        if s is None: continue
        print(f"{cfg:<6}{s[2]:>4}{s[0]:>10.4f}+/-{s[1]:.4f}  {c[0]:>10.4f}+/-{c[1]:.4f}")

print("\n"+"="*78)
print("PAIRED config-vs-Bare (ranking_score): exact t-test + Wilcoxon + 95% CI")
print("="*78)
for bb in ['nnunet','mamba']:
    print(f"\n### {bb.upper()} ###")
    print(f"{'cfg':<6}{'n':>4}{'meanD':>10}{'95% CI':>22}{'t-p':>10}{'Wilcox-p':>11}")
    for cfg in CONFIGS:
        if cfg=='Bare': continue
        d = paired_vs_bare(bb,cfg,'ranking_score')
        if len(d)<2: continue
        ps = paired_stats(d); w = wilcoxon(d)
        ci = f"[{ps['ci'][0]:+.4f},{ps['ci'][1]:+.4f}]"
        wp = f"{w['p']:.4f}" if w else "--"
        print(f"{cfg:<6}{ps['n']:>4}{ps['mean']:>+10.4f}{ci:>22}{ps['p']:>10.4f}{wp:>11}({w['method'][:4]})")

print("\n"+"="*78)
print("PAIRED config-vs-Bare (case_specificity): exact t + Wilcoxon + 95% CI")
print("="*78)
for bb in ['nnunet','mamba']:
    print(f"\n### {bb.upper()} ###")
    print(f"{'cfg':<6}{'n':>4}{'meanD':>10}{'95% CI':>22}{'t-p':>10}{'Wilcox-p':>11}")
    for cfg in CONFIGS:
        if cfg=='Bare': continue
        d = paired_vs_bare(bb,cfg,'case_specificity')
        if len(d)<2: continue
        ps = paired_stats(d); w = wilcoxon(d)
        ci = f"[{ps['ci'][0]:+.4f},{ps['ci'][1]:+.4f}]"
        wp = f"{w['p']:.4f}" if w else "--"
        print(f"{cfg:<6}{ps['n']:>4}{ps['mean']:>+10.4f}{ci:>22}{ps['p']:>10.4f}{wp:>11}({w['method'][:4]})")

print("\n"+"="*78)
print("MARGINAL EFFECTS (mean of configs WITH component - mean WITHOUT), ranking_score")
print("="*78)
for bb in ['nnunet','mamba']:
    print(f"\n### {bb.upper()} ###")
    for ci_, comp in enumerate(['G','D','M']):
        with_cfgs = [c for c in CONFIGS if GDM[c][ci_]==1]
        wo_cfgs   = [c for c in CONFIGS if GDM[c][ci_]==0]
        wv = [cell_mean_sd(bb,c,'ranking_score')[0] for c in with_cfgs if cell_mean_sd(bb,c,'ranking_score')]
        ov = [cell_mean_sd(bb,c,'ranking_score')[0] for c in wo_cfgs if cell_mean_sd(bb,c,'ranking_score')]
        print(f"  {comp}: {mean(wv)-mean(ov):+.4f}  (with {len(wv)} cfgs vs without {len(ov)})")

print("\nDONE.")
