"""Qualitative figure: representative slices (75th percentile of localisation score) per test set.
Usage (repo root, after sourcing pc_environment.env):
    python scripts/make_fig_qual.py [percentile] [onestep_t2_run] [onestep_t1_run]
Run names default to those created by scripts/cross_validate.sh (fold 0)."""
import sys, glob, os, numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = os.path.join(os.environ['LOG_DIR'], 'logs', 'runs')   # run folders written by run.py
OUT = os.environ.get('FIG_DIR', '.')
PCT = float(sys.argv[1]) if len(sys.argv) > 1 else 75
RUN_T2 = sys.argv[2] if len(sys.argv) > 2 else 'onestep_t2_f0'
RUN_T1 = sys.argv[3] if len(sys.argv) > 3 else 'onestep_t1_f0'

def segments(nonwhite):
    idx = np.where(nonwhite)[0]; cuts = np.where(np.diff(idx) > 3)[0]
    return [(s, e + 1) for s, e in zip(np.r_[idx[0], idx[cuts + 1]], np.r_[idx[cuts], idx[-1]])]
def panels(path):
    a = np.asarray(Image.open(path).convert('RGB')); nw = (a < 250).any(2)
    r0, r1 = segments(nw.any(1))[0]
    return [a[r0:r1, c0:c1] for c0, c1 in segments(nw.any(0))][:4]
def score(p):
    m = p[2].astype(float).mean(2); g = p[3].mean(2) > 128
    if g.mean() < 0.01: return None                       # need a visible lesion
    return (m * g).sum() / (m.sum() + 1e-6)                # share of anomaly-map mass inside the lesion

sets = [('BraTS21', RUN_T2, lambda n: n.startswith('BraTS')),
        ('MSLUB',   RUN_T2, lambda n: n.startswith('patient')),
        ('ATLAS',   RUN_T1, lambda n: n.startswith('sub')),
        ('WMH',     RUN_T1, lambda n: not n.startswith(('BraTS', 'patient', 'sub', 'IXI')))]
rows = []
for name, run, test in sets:
    files = [f for f in glob.glob(f'{R}/{run}/*/grid/*_Grid.png') if test(os.path.basename(f))]
    sc = []
    for f in files:
        try:
            p = panels(f); s = score(p)
            if s is not None and len(p) == 4: sc.append((s, f))
        except Exception: pass
    sc.sort(); vals = np.array([s for s, _ in sc])
    target = np.percentile(vals, PCT); order = np.argsort(np.abs(vals - target))
    seen, n = set(), 0
    for i in order:
        s, f = sc[i]; sid = os.path.basename(f).rsplit('_', 2)[0]
        if sid in seen: continue
        seen.add(sid); rows.append((name, panels(f))); n += 1
        print(f'{name:8s} score {s:.2f} (p{PCT:.0f}={target:.2f}, n={len(sc)})  {os.path.basename(f)}')
        if n == 2: break

fig, ax = plt.subplots(len(rows), 4, figsize=(8, 2.05 * len(rows)))
for i, (name, p) in enumerate(rows):
    for j in range(4):
        ax[i, j].imshow(p[j]); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        for sp in ax[i, j].spines.values(): sp.set_visible(False)
    ax[i, 0].set_ylabel(name, fontsize=12)
for j, t in enumerate(['Input', 'Reconstruction', 'Anomaly map', 'Ground truth']):
    ax[0, j].set_title(t, fontsize=12)
plt.tight_layout(pad=0.3); plt.savefig(f'{OUT}/fig_qualitative.png', dpi=250); print('saved fig_qualitative.png')
