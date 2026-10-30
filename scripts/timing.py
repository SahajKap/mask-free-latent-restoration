"""Reconstruction time per volume (50 slices) of trained models, as in the paper.
Usage (repo root, GPU):
    python scripts/timing.py <onestep_ckpt> <iterative_ckpt> <latent_scale> [ddpm_ckpt]
The optional DDPM checkpoint is a model trained with the cDDPM repository (experiment=cDDPM/DDPM)."""
import os, sys, time, torch
sys.path.insert(0, '.')
from hydra import compose, initialize_config_dir
from src.models.MaskFreeLDM import MaskFreeLDM

onestep, iterative, scale = sys.argv[1], sys.argv[2], sys.argv[3]
ddpm = sys.argv[4] if len(sys.argv) > 4 else None
os.environ.setdefault('INIT_CKPT', onestep); os.environ.setdefault('LATENT_SCALE', scale)

def timeit(fn, n=10):
    for _ in range(2): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n

x = torch.rand(50, 1, 96, 96, device='cuda')   # one volume = 50 slices
with initialize_config_dir(config_dir=os.path.abspath('configs'), version_base=None):
    for name, exp, ck in [('one-step', 'onestep', onestep), ('iterative', 'iterative', iterative)]:
        cfg = compose('config', overrides=[f'experiment=maskfree/{exp}', 'model.cfg.init_ckpt=null', f'model.cfg.latent_scale={scale}'])
        m = MaskFreeLDM(cfg.model.cfg).cuda().eval()
        m.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)['state_dict'], strict=False)
        with torch.no_grad(): s = timeit(lambda: m.reconstruct(x))
        print(f'{name:10s} {s:.3f} s / volume')
    if ddpm:
        from src.models.DDPM_2D import DDPM_2D
        from src.utils.generate_noise import gen_noise
        cfg = compose('config', overrides=['experiment=cDDPM/DDPM'])
        d = DDPM_2D(cfg.model.cfg).cuda().eval()
        d.load_state_dict(torch.load(ddpm, map_location='cpu', weights_only=False)['state_dict'], strict=False)
        T = cfg.model.cfg.get('test_timesteps', 500)
        def f():
            noise = gen_noise(d.cfg, x.shape).to('cuda') if d.cfg.get('noisetype') is not None else None
            return d.diffusion(x, cond=None, t=T - 1, noise=noise)
        with torch.no_grad(): s = timeit(f, n=3)
        print(f'DDPM (t={T}) {s:.3f} s / volume')
