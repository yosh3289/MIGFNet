#!/usr/bin/env python
"""Local re-extraction of MIGF modality-quality weights (alpha) for the 30 A2
checkpoints, to recover the per-modality static-prior breakdown for the figure.
Mirrors gate_analysis.py: hooks ModalityQualityEstimator(3D), collects per-case
[N,3] softmax weights on each model's validation fold, saves mean + across-case
std per modality. Run with the `mar` conda env (torch cu128 + picai_baseline).
"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np
import torch
warnings.filterwarnings('ignore')

MAR = Path(r".")
OUT = Path(r"./gate_weights_local.json")
sys.path.insert(0, str(MAR)); sys.path.insert(0, str(MAR / "scripts"))
import yaml
from torch.utils.data import DataLoader
from src.data.dataset import PICIADataset
from src.models.adaptive_native import build_adaptive_native
from src.models.adaptive_mpnet import build_adaptive_mpnet

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def jobs():
    js = []
    for fold in range(5):
        for seed in (42, 123, 789):
            js.append(dict(family='nnunet', fold=fold, seed=seed,
                ckpt=MAR/f"outputs/checkpoints/ablation_nnunet/A2/fold{fold}_seed{seed}/adaptive_nnunet/best_model.pth",
                cfg=MAR/"configs/ablation_nnunet_A2_no_deepsup.yaml"))
            js.append(dict(family='mamba', fold=fold, seed=seed,
                ckpt=MAR/f"outputs/checkpoints/ablation_mamba/A2/fold{fold}_seed{seed}/MIGFNet-Mamba/best_model.pth",
                cfg=MAR/"configs/ablation_mamba_no_deepsup.yaml"))
    return js

def build_model(family, cfg, ckpt):
    model = (build_adaptive_native("adaptive_nnunet", cfg) if family=='nnunet'
             else build_adaptive_mpnet(cfg)).to(dev)
    sd = torch.load(ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(sd["model_state_dict"]); model.eval()
    return model

CACHE = MAR / "data" / "cache"
def val_ids(fold):
    from picai_baseline.splits.picai import valid_splits
    return valid_splits[fold]["subject_list"]

def loader(cfg, fold, batch_size=8):
    """Direct-cache val loader (bypasses raw-image discovery): yields batches of
    {t2w,hbv,adc} each [B,1,32,128,128] from cache .pt files for this fold's val split."""
    ids = [s for s in val_ids(fold) if (CACHE/f"{s}.pt").exists()]
    batches = []
    cur = {"t2w": [], "hbv": [], "adc": []}
    def flush():
        if cur["t2w"]:
            batches.append({k: torch.stack(v) for k, v in cur.items()})
            for k in cur: cur[k] = []
    for sid in ids:
        d = torch.load(CACHE/f"{sid}.pt", map_location="cpu", weights_only=False)
        for k in ("t2w", "hbv", "adc"):
            cur[k].append(d[k].float().unsqueeze(0))  # [1,32,128,128]
        if len(cur["t2w"]) == batch_size:
            flush()
    flush()
    return batches

def is_qe(m): return m.__class__.__name__ in {"ModalityQualityEstimator3D","ModalityQualityEstimator"}

def collect(model, ld):
    caps = {}
    hs = []
    for name, mod in model.named_modules():
        if is_qe(mod):
            caps[name] = []
            def hook(_m,_i,out,key=name):
                a = out.detach().float().cpu()
                if a.ndim==3 and a.shape[-1]==1: a = a[...,0]
                caps[key].append(a.numpy())
            hs.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        for b in ld:
            t2w,hbv,adc = (b["t2w"].to(dev), b["hbv"].to(dev), b["adc"].to(dev))
            with torch.amp.autocast('cuda'):
                _ = model([t2w,hbv,adc])
    for h in hs: h.remove()
    return {k:(np.concatenate(v,0) if v else np.zeros((0,3),np.float32)) for k,v in caps.items()}

results = []
for i, j in enumerate(jobs()):
    if not Path(j['ckpt']).exists():
        print(f"[{i+1}/30] MISSING {j['ckpt']}"); continue
    cfg = yaml.safe_load(open(j['cfg']))
    model = build_model(j['family'], cfg, j['ckpt'])
    ld = loader(cfg, j['fold'])
    sw = collect(model, ld)
    stage_names = [k for k in sw if sw[k].size]
    if not stage_names:
        print(f"[{i+1}/30] {j['family']} f{j['fold']}s{j['seed']}: NO QE captured"); continue
    # entry stage = first QE (modality-weighting stage, matches original Fig 4b)
    entry = sw[stage_names[0]]                       # [N,3]
    # across-stage aggregate for ratio (as in paper)
    case_sds = [sw[k].std(0) for k in stage_names]
    mod_sds  = [sw[k].std(1).mean() for k in stage_names]
    ratio = float(np.mean([c.mean() for c in case_sds]) / max(np.mean(mod_sds), 1e-8))
    ew, esd = entry.mean(0), entry.std(0)           # entry-stage mean & across-case std
    results.append(dict(family=j['family'], fold=j['fold'], seed=j['seed'], ratio=ratio,
        entry_mean=[float(x) for x in ew], entry_casesd=[float(x) for x in esd],
        stage_means={k: [float(x) for x in sw[k].mean(0)] for k in stage_names},
        stage_casesd={k: [float(x) for x in sw[k].std(0)] for k in stage_names}))
    print(f"[{i+1}/30] {j['family']} f{j['fold']}s{j['seed']}: "
          f"entry w=[{ew[0]:.3f},{ew[1]:.3f},{ew[2]:.3f}] casesd=[{esd[0]:.3f},{esd[1]:.3f},{esd[2]:.3f}] ratio={ratio:.4f}")
    del model; torch.cuda.empty_cache()

json.dump(results, open(OUT,'w'), indent=1)
print("\nsaved", OUT, f"({len(results)} ckpts)")
# quick aggregate
for fam in ('nnunet','mamba'):
    r=[x for x in results if x['family']==fam]
    if not r: continue
    mw=np.array([x['entry_mean'] for x in r])
    sd=np.array([x['entry_casesd'] for x in r])
    rr=np.array([x['ratio'] for x in r])
    print(f"{fam}: entry weights T2W/HBV/ADC = {mw.mean(0).round(3)}, across-case sd = {sd.mean(0).round(3)}, ratio {rr.mean():.4f}+-{rr.std(ddof=1):.4f}")
