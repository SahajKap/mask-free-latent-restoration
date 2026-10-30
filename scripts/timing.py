import sys, time, glob, torch
sys.path.insert(0, '.')
from hydra import compose, initialize
R = '/workspace/logs/logs/runs'
ck = lambda n: sorted(glob.glob(f'{R}/{n}/*/checkpoints/epoch-*.ckpt'))[-1]
def timeit(fn, n=10):
    for _ in range(2): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n
x = torch.rand(50, 1, 96, 96, device='cuda')   # one volume = 50 slices
from src.models.MaskFreeLDM import MaskFreeLDM
with initialize(config_path='../configs'):
    for name, run, extra in [('ours_iterative', 'ab_noinput', []),
                             ('ours_onestep',   'ab_onestep', ['model.cfg.t0_ensemble=[1]', 'model.cfg.ddim_steps=1'])]:
        cfg = compose('config', overrides=['experiment=maskfree/final', 'model.cfg.init_ckpt=null', 'model.cfg.latent_scale=1.458'] + extra)
        m = MaskFreeLDM(cfg.model.cfg).cuda().eval()
        m.load_state_dict(torch.load(ck(run), map_location='cpu', weights_only=False)['state_dict'], strict=False)
        with torch.no_grad(): s = timeit(lambda: m.reconstruct(x))
        print(f'{name:16s} {s:.3f} s / volume')
    try:
        from src.models.DDPM_2D import DDPM_2D
        from src.utils.generate_noise import gen_noise
        cfg = compose('config', overrides=['experiment=cDDPM/DDPM'])
        d = DDPM_2D(cfg.model.cfg).cuda().eval()
        d.load_state_dict(torch.load(ck('ddpm_t2'), map_location='cpu', weights_only=False)['state_dict'], strict=False)
        T = cfg.model.cfg.get('test_timesteps', 500)
        def f():
            noise = gen_noise(d.cfg, x.shape).to('cuda') if d.cfg.get('noisetype') is not None else None
            return d.diffusion(x, cond=None, t=T - 1, noise=noise)
        with torch.no_grad(): s = timeit(f, n=3)
        print(f'DDPM (t={T})       {s:.3f} s / volume')
    except Exception as e:
        print('DDPM timing failed:', repr(e)[:300])
