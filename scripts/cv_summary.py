"""Collects test Dice / AUPRC / l1-ratio over the 5 folds and prints mean +- std.
Usage (repo root):  python scripts/cv_summary.py"""
import glob, os, pickle, numpy as np
R = os.path.join(os.environ.get('LOG_DIR', '.'), 'logs', 'runs')
runs = {  # model -> contrast -> list of run names, fold 0 first
    'iterative': {'t2': [f'iterative_t2_f{k}' for k in range(5)],
                  't1': [f'iterative_t1_f{k}' for k in range(5)]},
    'one-step':  {'t2': [f'onestep_t2_f{k}' for k in range(5)],
                  't1': [f'onestep_t1_f{k}' for k in range(5)]},
}
def load(name):
    p = sorted(glob.glob(f'{R}/{name}/*/*_preds_dict.pkl'), key=os.path.getmtime)
    return pickle.load(open(p[-1], 'rb'))['test'] if p else None
for model, by_mode in runs.items():
    print(f'\n=== {model}')
    for mode, names in by_mode.items():
        res = {}
        for n in names:
            d = load(n)
            if d is None: print(f'  missing: {n}'); continue
            for s, e in d.items():
                if 'IXI' in s: continue
                r = res.setdefault(s.split('.')[-1], {'Dice': [], 'AUPRC': [], 'l1': []})
                r['Dice'].append(100 * e['DicePerVolMean']); r['AUPRC'].append(100 * e['AUPRCPerVolMean'])
                r['l1'].append(e['l1recoErrorUnhealthyMean'] / e['l1recoErrorHealthyMean'])
        for s, r in res.items():
            f = lambda v: f'{np.mean(v):6.2f} +- {np.std(v):4.2f}'
            print(f'  {s:8s} n={len(r["Dice"])}  Dice {f(r["Dice"])}   AUPRC {f(r["AUPRC"])}   l1 {f(r["l1"])}   '
                  f'per-fold Dice {[round(x, 2) for x in r["Dice"]]}')
