"""
MaskFreeLDM
Learned anomaly generator + conditional latent diffusion for unsupervised
anomaly detection in brain MRI, implemented inside the cDDPM framework of
Behrendt et al. so that data loading, preprocessing, splits, threshold
selection and metrics are identical to the published benchmark.

One class, three training stages selected by cfg.stage:
    vae        train the VAE
    generator  train the anomaly generator      (VAE frozen)
    diffusion  train the conditional diffusion  (VAE + generator frozen)
Each later stage loads the previous stage's checkpoint via cfg.init_ckpt.
Test-time evaluation is meaningful only for the diffusion stage.

Input slices are 96x96 (imageDim 192x192, rescaleFactor 2), intensities in
[0, 1] after the framework's percentile rescaling. Latent is (4, 24, 24).
"""
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchio as tio
from pytorch_lightning import LightningModule

from src.utils.utils_eval import _test_step, _test_end, get_eval_dictionary


# ═══════════════════════════════ helpers ═══════════════════════════════

def gn(c):
    return nn.GroupNorm(8, c)


def gaussian_blur(x, sigma):
    """Separable Gaussian blur, no kornia dependency."""
    ks = int(2 * math.ceil(3 * sigma) + 1)
    ax = torch.arange(ks, device=x.device, dtype=x.dtype) - ks // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (ks // 2, ks // 2, 0, 0), mode="replicate"),
                 g.view(1, 1, 1, ks).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, ks // 2, ks // 2), mode="replicate"),
                 g.view(1, 1, ks, 1).repeat(c, 1, 1, 1), groups=c)
    return x


def erode(mask, iters):
    for _ in range(iters):
        mask = -F.max_pool2d(-mask, 3, 1, 1)
    return mask


# ═══════════════════════════════ VAE ═══════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(gn(c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1),
                                   gn(c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.block(x)


class Down(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv = nn.Conv2d(i, o, 4, 2, 1)
        self.res = ResBlock(o)

    def forward(self, x):
        return self.res(self.conv(x))


class Up(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv = nn.ConvTranspose2d(i, o, 4, 2, 1)
        self.res = ResBlock(o)

    def forward(self, x):
        return self.res(self.conv(x))


class VAE(nn.Module):
    """(1, 96, 96) -> (z_ch, 24, 24) -> (1, 96, 96)"""

    def __init__(self, z_ch=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),
                                 Down(32, 64), Down(64, 128),
                                 ResBlock(128), ResBlock(128))
        self.to_mu = nn.Conv2d(128, z_ch, 1)
        self.to_logvar = nn.Conv2d(128, z_ch, 1)
        self.dec = nn.Sequential(nn.Conv2d(z_ch, 128, 1), ResBlock(128), ResBlock(128),
                                 Up(128, 64), Up(64, 32),
                                 gn(32), nn.SiLU(), nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return self.to_mu(h), self.to_logvar(h).clamp(-30, 20)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar


# ═══════════════════════════ anomaly generator ═══════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), gn(o), nn.SiLU(),
                               nn.Conv2d(o, o, 3, padding=1), gn(o), nn.SiLU())

    def forward(self, x):
        return self.b(x)


class AnomalyGenerator(nn.Module):
    """(1, 96, 96) healthy slice -> soft mask in [0, 1]"""

    def __init__(self):
        super().__init__()
        self.e1, self.e2 = ConvBlock(1, 32), ConvBlock(32, 64)
        self.e3, self.b = ConvBlock(64, 128), ConvBlock(128, 128)
        self.d3, self.d2, self.d1 = ConvBlock(256, 64), ConvBlock(128, 32), ConvBlock(64, 32)
        self.out = nn.Conv2d(32, 1, 1)
        nn.init.constant_(self.out.bias, -2.0)  # start sparse

    def field(self, x):
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        s1 = self.e1(x)                         # 96
        s2 = self.e2(F.max_pool2d(s1, 2))       # 48
        s3 = self.e3(F.max_pool2d(s2, 2))       # 24
        b = self.b(F.max_pool2d(s3, 2))         # 12
        x = self.d3(torch.cat([up(b), s3], 1))
        x = self.d2(torch.cat([up(x), s2], 1))
        x = self.d1(torch.cat([up(x), s1], 1))
        return self.out(x)

    def forward(self, x):
        return torch.sigmoid(self.field(x))


# ═══════════════════════ conditional diffusion UNet ═══════════════════════

class TimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4))

    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None].float() * f[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


class ResT(nn.Module):
    def __init__(self, i, o, td):
        super().__init__()
        self.n1, self.c1 = gn(i), nn.Conv2d(i, o, 3, padding=1)
        self.tp = nn.Linear(td, o)
        self.n2, self.c2 = gn(o), nn.Conv2d(o, o, 3, padding=1)
        self.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity()

    def forward(self, x, t):
        h = self.c1(F.silu(self.n1(x))) + self.tp(F.silu(t))[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.n = gn(c)
        self.a = nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.n(x).flatten(2).transpose(1, 2)
        h, _ = self.a(h, h, h)
        return x + h.transpose(1, 2).reshape(B, C, H, W)


class DownT(nn.Module):
    def __init__(self, i, o, td, attn):
        super().__init__()
        self.r = ResT(i, o, td)
        self.at = Attn(o) if attn else nn.Identity()
        self.d = nn.Conv2d(o, o, 4, 2, 1)

    def forward(self, x, t):
        x = self.at(self.r(x, t))
        return self.d(x), x


class UpT(nn.Module):
    def __init__(self, i, s, o, td, attn):
        super().__init__()
        self.u = nn.ConvTranspose2d(i, i, 4, 2, 1)
        self.r = ResT(i + s, o, td)
        self.at = Attn(o) if attn else nn.Identity()

    def forward(self, x, skip, t):
        return self.at(self.r(torch.cat([self.u(x), skip], 1), t))


class CondUNet(nn.Module):
    """cat(z_t, z_cond): (2*z_ch, 24, 24) -> predicted noise (z_ch, 24, 24)"""

    def __init__(self, z_ch=4, base=64, tdim=256):
        super().__init__()
        td = tdim * 4
        self.temb = TimeEmb(tdim)
        self.inc = nn.Conv2d(2 * z_ch, base, 3, padding=1)
        self.d1 = DownT(base, base * 2, td, False)          # 24 -> 12
        self.d2 = DownT(base * 2, base * 4, td, True)       # 12 -> 6
        self.d3 = DownT(base * 4, base * 4, td, True)       # 6  -> 3
        self.m1, self.ma = ResT(base * 4, base * 4, td), Attn(base * 4)
        self.m2 = ResT(base * 4, base * 4, td)
        self.u3 = UpT(base * 4, base * 4, base * 4, td, True)   # 3  -> 6
        self.u2 = UpT(base * 4, base * 4, base * 2, td, True)   # 6  -> 12
        self.u1 = UpT(base * 2, base * 2, base, td, False)      # 12 -> 24
        self.out = nn.Sequential(gn(base), nn.SiLU(), nn.Conv2d(base, z_ch, 3, padding=1))

    def forward(self, z_t, cond, t):
        te = self.temb(t)
        x = self.inc(torch.cat([z_t, cond], 1))
        x, s1 = self.d1(x, te)
        x, s2 = self.d2(x, te)
        x, s3 = self.d3(x, te)
        x = self.m2(self.ma(self.m1(x, te)), te)
        x = self.u3(x, s3, te)
        x = self.u2(x, s2, te)
        x = self.u1(x, s1, te)
        return self.out(x)


# ═══════════════════════════ Lightning module ═══════════════════════════

class MaskFreeLDM(LightningModule):
    def __init__(self, cfg, prefix=None):
        super().__init__()
        self.cfg = cfg
        self.prefix = prefix

        z = cfg.get("latent_ch", 4)
        self.vae = VAE(z)
        self.gen = AnomalyGenerator()
        self.unet = CondUNet(z, cfg.get("unet_base", 64), cfg.get("time_dim", 256))

        self.T = cfg.get("timesteps", 1000)
        betas = torch.linspace(cfg.get("beta_start", 1e-4), cfg.get("beta_end", 0.02), self.T)
        self.register_buffer("alpha_bar", torch.cumprod(1.0 - betas, 0))

        if cfg.get("init_ckpt"):
            sd = torch.load(cfg.init_ckpt, map_location="cpu", weights_only=False)["state_dict"]
            missing, unexpected = self.load_state_dict(sd, strict=False)
            print(f"[MaskFreeLDM] loaded {cfg.init_ckpt} | missing={len(missing)} unexpected={len(unexpected)}")

        # only the active stage's module is trainable
        for m in (self.vae, self.gen, self.unet):
            m.requires_grad_(m is self._active())

        self.save_hyperparameters()

    # ── staging ─────────────────────────────────────────────────────────
    @property
    def stage_name(self):
        return self.cfg.get("stage", "diffusion")

    def _active(self):
        return {"vae": self.vae, "generator": self.gen, "diffusion": self.unet}[self.stage_name]

    def configure_optimizers(self):
        return optim.Adam(self._active().parameters(), lr=self.cfg.get("stage_lr", None) or self.cfg.lr)

    def update_prefix(self, prefix):
        self.prefix = prefix

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # torchio>=1.0 Images can't be rebuilt by Lightning's mover; hand it plain dicts
        def plain(o):
            if isinstance(o, dict):
                return {k: plain(v) for k, v in o.items()}
            if isinstance(o, list):
                return [plain(v) for v in o]
            return o
        return super().transfer_batch_to_device(plain(batch), device, dataloader_idx)

    # ── corruption ──────────────────────────────────────────────────────
    def _brain(self, x, mask=None):
        b = mask if mask is not None else (x > 0.01).float()
        return erode((b > 0).float(), self.cfg.get("brain_erode", 4))

    def sample_delta(self, x):
        B, _, H, W = x.shape
        r = max(H // 16, 2)
        organic = F.interpolate(torch.randn(B, 1, r, r, device=x.device), size=(H, W),
                                mode="bicubic", align_corners=False)
        organic = organic * torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.1, 0.3)
        shift = torch.empty(B, 1, 1, 1, device=x.device).uniform_(-0.4, 0.4).expand_as(x)
        foreign = torch.roll(x, 1, 0) - x                       # tissue from another subject
        c = torch.randint(0, 3, (B, 1, 1, 1), device=x.device).expand_as(x)
        return torch.where(c == 0, organic, torch.where(c == 1, shift, foreign))

    @staticmethod
    def _standardise(Z, brain):
        n = brain.sum((1, 2, 3), keepdim=True).clamp(min=1.0)
        mu = (Z * brain).sum((1, 2, 3), keepdim=True) / n
        sd = (((Z - mu) ** 2) * brain).sum((1, 2, 3), keepdim=True).div(n).add(1e-6).sqrt()
        return (Z - mu) / sd

    def make_mask(self, x, brain, source=None):
        """Soft mask covering exactly a random fraction of the brain (area_range), placed where a
        smooth field is highest. learned: field = generator output (+ place_noise * random field);
        random: field = random smooth noise only (ablation)."""
        source = source or self.cfg.get("mask_source", "learned")
        B, _, H, W = x.shape
        g = self.cfg.get("field_res", 12)
        noise = F.interpolate(torch.randn(B, 1, g, g, device=x.device), size=(H, W), mode="bicubic", align_corners=False)
        noise = self._standardise(noise, brain)
        if source == "random":
            Z = noise
        else:
            L = F.interpolate(F.adaptive_avg_pool2d(self.gen.field(x).float(), g), size=(H, W), mode="bicubic", align_corners=False)
            if self.cfg.get("field_flip", True):   # hemispheric symmetry: use the learned map on either side
                flip = torch.rand(B, 1, 1, 1, device=x.device) < 0.5
                L = torch.where(flip, L.flip(-1), L)
            Z = self._standardise(L, brain) + self.cfg.get("place_noise", 1.0) * noise
        a = torch.empty(B, device=x.device).uniform_(*self.cfg.get("area_range", [0.01, 0.08]))
        n = brain.flatten(1).sum(1)
        k = (a * n).long().clamp(min=1)
        Zb = Z.detach().flatten(1).masked_fill(brain.flatten(1) < 0.5, float("-inf"))
        tau = Zb.sort(1, descending=True).values.gather(1, (k - 1).unsqueeze(1)).view(B, 1, 1, 1)
        tau = torch.nan_to_num(tau, neginf=-1e4)
        return torch.sigmoid((Z - tau) / self.cfg.get("mask_temp", 0.1)) * brain

    def create_anomaly(self, x, M, delta, brain):
        sigma = float(np.random.uniform(*self.cfg.get("mask_sigma", [0.75, 2.0])))
        Ms = gaussian_blur(M * brain, sigma)
        return (x * (1 - Ms) + (x + delta) * Ms).clamp(0, 1)

    @torch.no_grad()
    def _conditioning(self, x, brain):
        M = self.make_mask(x, brain)
        x_fake = self.create_anomaly(x, M, self.sample_delta(x), brain)
        amp = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(*self.cfg.get("amp", [3.0, 6.0]))
        x_strong = (x + amp * (x_fake - x)).clamp(0, 1)
        return self.vae.encode(x_strong)[0]                    # mean latent, no sampling noise

    # ── per-stage losses ────────────────────────────────────────────────
    def _batch(self, batch):
        x = batch["vol"][tio.DATA].squeeze(-1).clamp(0, 1)
        m = batch["mask"][tio.DATA].squeeze(-1).float() if "mask" in batch else None
        return x, self._brain(x, m)

    def _gen_loss(self, x, brain):
        M = self.make_mask(x, brain)
        x_fake = self.create_anomaly(x, M, self.sample_delta(x), brain)
        mu_c = self.vae.encode(x)[0].detach()
        mu_f = self.vae.encode(x_fake)[0]
        area = (M * brain).sum((1, 2, 3)) / (brain.sum((1, 2, 3)) + 1.0)   # fraction of the brain
        err = area - self.cfg.get("target_area", 0.05)
        d = self.cfg.get("size_delta", 0.005)
        l_size = torch.where(err.abs() < d, 0.5 * err ** 2 / d, err.abs() - 0.5 * d).mean()
        l_tv = (M[:, :, 1:] - M[:, :, :-1]).abs().mean() + (M[..., 1:] - M[..., :-1]).abs().mean()
        l_div = -F.mse_loss(mu_f, mu_c) / (mu_c.pow(2).mean() + 1e-8)
        l_bin = ((4 * M * (1 - M) * brain).sum((1, 2, 3)) / (brain.sum((1, 2, 3)) + 1.0)).mean()
        loss = l_div   # area and shape are fixed by make_mask; size/tv/bin only logged
        return loss, {"size": l_size, "tv": l_tv, "div": l_div, "bin": l_bin, "area": area.mean()}

    def _diff_loss(self, x, brain):
        s = self.cfg.get("latent_scale", 1.0)
        with torch.no_grad():
            z0 = self.vae.encode(x)[0]
            zc = self._conditioning(x, brain)
            # with prob p_clean the condition is the clean latent, so healthy input maps to itself
            keep = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.cfg.get("p_clean_cond", 0.2)).float()
            zc = keep * z0 + (1 - keep) * zc
        z0, zc = z0 * s, zc * s
        zin = zc if self.cfg.get("corrupt_input", False) else z0
        drop = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.cfg.get("p_drop_cond", 0.0)).float()
        zc = zc * (1 - drop)
        t = torch.randint(0, int(self.cfg.get("t_max", self.T)), (x.shape[0],), device=x.device)  # t_max=1: one-step network (ablation)
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        eps = torch.randn_like(z0)
        zt = ab.sqrt() * zin + (1 - ab).sqrt() * eps
        target = z0 if self.cfg.get("pred_x0", False) else eps
        return F.mse_loss(self.unet(zt, zc, t), target), {}

    def _stage_loss(self, batch):
        x, brain = self._batch(batch)
        if self.stage_name == "vae":
            rec, mu, lv = self.vae(x)
            mse = F.mse_loss(rec, x)
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            return mse + self.cfg.get("beta_kl", 1e-3) * kl, {"mse": mse, "kl": kl}
        if self.stage_name == "generator":
            return self._gen_loss(x, brain)
        return self._diff_loss(x, brain)

    def training_step(self, batch, batch_idx: int):
        loss, parts = self._stage_loss(batch)
        bs = batch["vol"][tio.DATA].shape[0]
        self.log(f"{self.prefix}train/Loss", loss, on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
        for k, v in parts.items():
            self.log(f"{self.prefix}train/{k}", v, on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, _ = self._stage_loss(batch)
        bs = batch["vol"][tio.DATA].shape[0]
        # log under both names; the checkpoint callback monitors one of them
        self.log(f"{self.prefix}val/Loss_comb", loss, on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
        self.log(f"{self.prefix}val/loss", loss, on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
        return {"loss": loss}

    # ── inference ───────────────────────────────────────────────────────
    @torch.no_grad()
    def ddim_from(self, z_in, cond, t0, steps):
        """Noise z_in to step t0, then deterministic DDIM back to 0."""
        ab = self.alpha_bar
        ts = torch.linspace(t0 - 1, 0, steps, device=z_in.device).round().long()
        z = ab[ts[0]].sqrt() * z_in + (1 - ab[ts[0]]).sqrt() * torch.randn_like(z_in)
        clip = self.cfg.get("z_clip", None)
        for i, t in enumerate(ts):
            a = ab[t]
            tt = t.expand(z.shape[0])
            w = self.cfg.get("guidance_w", 1.0)
            out = self.unet(z, cond, tt)
            if w != 1.0:
                o_null = self.unet(z, torch.zeros_like(cond), tt)
                out = o_null + w * (out - o_null)
            if self.cfg.get("pred_x0", False):
                z0 = out
                eps = (z - a.sqrt() * z0) / (1 - a).sqrt()
            else:
                eps = out
                z0 = (z - (1 - a).sqrt() * eps) / a.sqrt()
            if clip:
                z0 = z0.clamp(-clip, clip)
            if i < len(ts) - 1:
                an = ab[ts[i + 1]]
                z = an.sqrt() * z0 + (1 - an).sqrt() * eps
            else:
                z = z0
        return z

    @torch.no_grad()
    def reconstruct(self, x):
        s = self.cfg.get("latent_scale", 1.0)
        mu = self.vae.encode(x.clamp(0, 1))[0] * s
        if self.cfg.get("test_cond", "input") == "corrupt":      # old behaviour, kept for ablation
            cond = self._conditioning(x, self._brain(x)) * s
        else:                                                    # condition on the test image itself
            cond = mu
        recos = []
        for t0 in self.cfg.get("t0_ensemble", [250, 500, 750]):
            z = self.ddim_from(mu, cond, int(t0), int(self.cfg.get("ddim_steps", 25)))
            recos.append(self.vae.decode(z / s))
        reco = torch.stack(recos).mean(0)
        if self.cfg.get("cancel_vae_error", False):
            # makes |x - reco'| = |D(E(x)) - reco|, removing the VAE's own error from the residual
            reco = x - self.vae.decode(mu / s) + reco
        return reco

    def on_test_start(self):
        self.eval_dict = get_eval_dictionary()
        self.inds = []
        self.latentSpace_slice = []
        self.new_size = [160, 190, 160]
        self.diffs_list = []
        self.seg_list = []
        if not hasattr(self, "threshold"):
            self.threshold = {}

    def test_step(self, batch: Any, batch_idx: int):
        torch.manual_seed(self.cfg.get("test_seed", 0) + batch_idx)   # deterministic per volume
        self.dataset = batch["Dataset"]
        input = batch["vol"][tio.DATA]
        data_orig = batch["vol_orig"][tio.DATA]
        data_seg = batch["seg_orig"][tio.DATA] if batch["seg_available"] else torch.zeros_like(data_orig)
        data_mask = batch["mask_orig"][tio.DATA]
        ID = batch["ID"]
        self.stage = batch["stage"]
        label = batch["label"]

        if self.cfg.get("num_eval_slices", input.size(4)) != input.size(4):
            n = self.cfg.get("num_eval_slices", input.size(4))
            s0 = int((input.size(4) - n) / 2)
            input = input[..., s0:s0 + n]
            data_orig = data_orig[..., s0:s0 + n]
            data_seg = data_seg[..., s0:s0 + n]
            data_mask = data_mask[..., s0:s0 + n]

        assert input.shape[0] == 1, "Batch size must be 1"
        x = input.squeeze(0).permute(3, 0, 1, 2)        # [B,C,H,W,D] -> [D,C,H,W]

        reco = self.reconstruct(x)

        final_volume = reco.squeeze(1).permute(1, 2, 0)  # -> H x W x D
        final_volume = final_volume.unsqueeze(0).unsqueeze(0)

        # keys the shared eval code expects
        self.eval_dict["latentSpace"].append(torch.zeros(1))
        self.eval_dict["AnomalyScoreRegPerVol"].append(0.0)

        _test_step(self, final_volume, data_orig, data_seg, data_mask, batch_idx, ID, label)

    def on_test_end(self):
        _test_end(self)
