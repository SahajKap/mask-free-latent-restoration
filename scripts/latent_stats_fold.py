"""1/std of VAE latent means on IXI train slices of a given fold.
Usage (repo root):  python scripts/latent_stats_fold.py <vae_ckpt> <t1|t2> <fold>"""
import os, sys, torch, hydra
from hydra import compose, initialize
sys.path.insert(0, '.')
from src.models.MaskFreeLDM import MaskFreeLDM

ckpt, mode, fold = sys.argv[1], sys.argv[2], int(sys.argv[3])
from hydra import initialize_config_dir
with initialize_config_dir(config_dir=os.path.abspath('configs')):
    cfg = compose('config', overrides=['experiment=maskfree/vae', f'datamodule.cfg.mode={mode}',
                                       'datamodule.cfg.num_workers=0', 'datamodule.cfg.sample_set=True'])
cfg.datamodule._target_ = 'src.datamodules.Datamodules_train.IXI'
dm = hydra.utils.instantiate(cfg.datamodule, fold=fold); dm.setup()
model = MaskFreeLDM(cfg.model.cfg).cuda().eval()
model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['state_dict'], strict=False)
zs = []
with torch.no_grad():
    for i, b in enumerate(dm.train_dataloader()):
        zs.append(model.vae.encode(b['vol']['data'].squeeze(-1).cuda())[0].cpu())
        if i == 50: break
z = torch.cat(zs)
print(f'latent mean {z.mean():.4f}  std {z.std():.4f}  ->  latent_scale: {1/z.std():.4f}')
