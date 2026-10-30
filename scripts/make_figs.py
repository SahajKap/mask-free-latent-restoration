"""Builds the two data figures for the paper.
  fig_qualitative.png : for each test set, the slices with the largest lesions (input | reconstruction | anomaly map | ground truth)
  fig_synthetic.png   : healthy IXI slices with randomly placed synthetic anomalies (input | mask | corrupted)
Usage (repo root):  python scripts/make_figs.py"""
import sys, glob, os, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '.')
R = '/workspace/logs/logs/runs'; OUT = '/workspace/logs'

def lesion_area(path):
    a = np.asarray(Image.open(path).convert('L')); w = a.shape[1]
    return (a[:, 3 * w // 4:] > 128).sum()
def pick(run, test, k):
    allf = glob.glob(f'{R}/{run}/*/grid/*_Grid.png')
    files = [f for f in allf if test(os.path.basename(f))]
    print(f'{run}: {len(allf)} grid images, {len(files)} match')
    files = sorted(files, key=lesion_area, reverse=True)
    out, seen = [], set()
    for f in files:                       # one slice per subject
        sid = os.path.basename(f).rsplit('_', 2)[0]
        if sid not in seen: out.append(f); seen.add(sid)
        if len(out) == k: break
    return out

sets = [('BraTS21 (T2)', 'os_t2', lambda n: n.startswith('BraTS')),
        ('MSLUB (T2)',   'os_t2', lambda n: n.startswith('patient')),
        ('ATLAS (T1)',   'os_t1', lambda n: n.startswith('sub')),
        ('WMH (T1)',     'os_t1', lambda n: not n.startswith(('BraTS', 'patient', 'sub', 'IXI')))]
rows, labels = [], []
for name, run, test in sets:
    for f in pick(run, test, 2):
        rows.append(Image.open(f).convert('RGB')); labels.append(name)
        print(name, os.path.basename(f))
W = max(r.width for r in rows); pad = 170
canvas = Image.new('RGB', (W + pad, sum(r.height for r in rows) + 60), 'white')
d = ImageDraw.Draw(canvas)
try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 26)
except Exception: font = ImageFont.load_default()
for i, t in enumerate(['Input', 'Reconstruction', 'Anomaly map', 'Ground truth']):
    d.text((pad + i * W // 4 + W // 8 - 70, 15), t, fill='black', font=font)
y = 60
for r, l in zip(rows, labels):
    canvas.paste(r, (pad, y)); d.text((10, y + r.height // 2 - 15), l, fill='black', font=font); y += r.height
canvas.save(f'{OUT}/fig_qualitative.png'); print('saved fig_qualitative.png')

# synthetic anomalies
import hydra
from hydra import compose, initialize
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.models.MaskFreeLDM import MaskFreeLDM
with initialize(config_path='../configs'):
    cfg = compose('config', overrides=['experiment=maskfree/final', 'datamodule.cfg.mode=t2', 'datamodule.cfg.num_workers=0',
                                       'datamodule.cfg.sample_set=True', 'model.cfg.init_ckpt=null'])
cfg.datamodule._target_ = 'src.datamodules.Datamodules_train.IXI'
dm = hydra.utils.instantiate(cfg.datamodule, fold=0); dm.setup()
m = MaskFreeLDM(cfg.model.cfg).cuda().eval()
torch.manual_seed(1); np.random.seed(1)
b = next(iter(dm.val_dataloader()))
x = b['vol']['data'].squeeze(-1).clamp(0, 1).cuda()
mk = b['mask']['data'].squeeze(-1).float().cuda()
keep = (x > 0.05).float().mean((1, 2, 3)) > 0.25                       # slices with plenty of brain
x, mk = x[keep][:6], mk[keep][:6]
with torch.no_grad():
    brain = m._brain(x, mk); M = m.make_mask(x, brain)
    xt = m.create_anomaly(x, M, m.sample_delta(x), brain)
    a = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(*cfg.model.cfg.amp)
    xc = (x + a * (xt - x)).clamp(0, 1)
n = x.shape[0]; fig, ax = plt.subplots(3, n, figsize=(2 * n, 6.3))
for j in range(n):
    for i, (t, cmap) in enumerate([(x, 'gray'), (M, 'magma'), (xc, 'gray')]):
        ax[i, j].imshow(t[j, 0].cpu().numpy(), cmap=cmap, vmin=0, vmax=1); ax[i, j].axis('off')
for i, t in enumerate(['Healthy slice', 'Mask', 'Corrupted']):
    ax[i, 0].text(-8, 48, t, rotation=90, va='center', ha='right', fontsize=11)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_synthetic.png', dpi=200); print('saved fig_synthetic.png')