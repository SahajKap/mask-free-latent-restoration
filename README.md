# Mask-Free Unsupervised Anomaly Detection in Brain MRI via One-Step Conditional Latent Restoration

Code for the paper by Rahul Nijhawan, Sahaj Kapoor, Mitul Pabri, Aarushi Rawal and Yashika Bhalla
(Thapar Institute of Engineering and Technology, Patiala, India).

A conditional network in the latent space of a VAE is trained, on healthy scans only, to map slices corrupted by
randomly placed synthetic anomalies back to their healthy latent code. No lesion masks are used at any stage. At test
time the network is applied once to the latent of the test slice, and the anomaly map is measured against the VAE's
own reconstruction (VAE-error cancellation). The same network trained as an iterative latent diffusion model is
included for comparison.

## Results (five-fold cross-validation, mean ± std)

| Model | BraTS21 Dice | BraTS21 AUPRC | MSLUB Dice | MSLUB AUPRC | ATLAS Dice | ATLAS AUPRC | WMH Dice | WMH AUPRC |
|---|---|---|---|---|---|---|---|---|
| One-step | 51.14 ± 1.14 | 51.20 ± 1.39 | 14.90 ± 0.28 | 13.02 ± 0.34 | 19.55 ± 0.41 | 22.67 ± 0.40 | 8.57 ± 0.85 | 7.98 ± 0.43 |
| Iterative | 45.60 ± 4.38 | 47.40 ± 5.35 | 5.34 ± 3.70 | 6.67 ± 2.71 | 17.95 ± 0.57 | 20.13 ± 1.00 | 6.54 ± 0.86 | 6.90 ± 0.50 |

Reconstruction time per volume (50 slices, one H100 MIG slice): one-step 0.017 s, iterative 0.432 s.

## Relation to the cDDPM benchmark

The method is implemented inside the benchmark code of Behrendt et al.
(*Guided reconstruction with conditioned diffusion models for unsupervised anomaly detection in brain MRIs*,
Computers in Biology and Medicine 186, 109660, 2025; https://github.com/FinnBehrendt/Conditioned-Diffusion-Models-UAD),
so data loading, splits, post-processing and metrics are theirs. This repository does not redistribute their code.
It contains only our own files and a patch (`compat.patch`) with our changes to theirs. Please also cite their work.

## 1. Setup

```bash
git clone https://github.com/SahajKap/mask-free-latent-restoration.git maskfree
git clone $(cat maskfree/UPSTREAM_URL.txt) cddpm && cd cddpm
git checkout $(cat ../maskfree/UPSTREAM_COMMIT.txt)
git apply ../maskfree/compat.patch
cp -r ../maskfree/src ../maskfree/configs ../maskfree/scripts .
pip install -r ../maskfree/requirements.txt
```

All commands below are run from inside `cddpm/`.
Tested with Python 3.10, PyTorch 2.6 (CUDA 12.4), PyTorch Lightning 2.6, TorchIO 1.2, Hydra 1.3 and MONAI.

## 2. Data

We do not redistribute data. Download the preprocessed IXI, BraTS21, MSLUB, ATLAS v2.0 and WMH data and the split
files as described in the cDDPM repository, then create `pc_environment.env`:

```
DATA_DIR=/path/to/folder/containing/Data
LOG_DIR=/path/to/logs
```

and load it in every shell:

```bash
set -a; source pc_environment.env; set +a; export WANDB_MODE=offline
```

The ATLAS v2.0 and WMH split-file paths are set in `configs/datamodule/IXI.yaml` (added by `compat.patch`);
adjust them if your split files are named differently.

## 3. Five-fold cross-validation (as in the paper)

```bash
bash scripts/cross_validate.sh t2      # T2 models: BraTS21, MSLUB
bash scripts/cross_validate.sh t1      # T1 models: ATLAS v2.0, WMH
python scripts/cv_summary.py           # mean ± std over folds (Dice, AUPRC, l1-ratio)
```

For each fold this trains a VAE (150 epochs), estimates its latent scale, and trains and tests the one-step and the
iterative model (2400 epochs each). A conditional model trains in under an hour on an H100 MIG slice.
The T1 runs also score the T1 scans of BraTS21 and MSLUB; the paper reports only ATLAS v2.0 and WMH for T1.

## 4. Training and evaluating a single model

```bash
python run.py experiment=maskfree/vae datamodule.cfg.mode=t2 name=vae_t2
export INIT_CKPT=<path to the VAE checkpoint>
export LATENT_SCALE=$(python scripts/latent_stats_fold.py $INIT_CKPT t2 0 | grep -oP 'latent_scale: \K[0-9.]+')
python run.py experiment=maskfree/onestep datamodule.cfg.mode=t2 name=onestep_t2
```

To evaluate an existing checkpoint without training, set `INIT_CKPT` to it, compute `LATENT_SCALE` from the same
file as above (conditional-model checkpoints contain their VAE), and add
`trainer.max_epochs=0 trainer.min_epochs=0 ckpt_path=null`.

Results are written to `<run dir>/<fold>_preds_dict.pkl`. The paper reports `DicePerVolMean` and `AUPRCPerVolMean`
of the `test` entries.

## 5. Pretrained models

The ten one-step models of the paper (five folds × T1/T2, VAE weights included) are attached to the
[v1.0 release](../../releases). File names are `fold{k}_onestep_{t1,t2}.pt`; use them as `INIT_CKPT` as in Section 4.

## 6. Ablations (paper Section 5.1)

Overrides of `experiment=maskfree/onestep`:

| Ablation | Override |
|---|---|
| without condition channel | train with `model.cfg.p_drop_cond=1.0`, evaluate with `model.cfg.guidance_w=0.0` |
| without VAE-error cancellation | `model.cfg.cancel_vae_error=False` (evaluation only) |
| without condition dropout | `model.cfg.p_drop_cond=0.0` |
| stronger amplification | `model.cfg.amp=[3.0,6.0]` |
| learned placement | train `experiment=maskfree/generator_ablation` first, then use its checkpoint as `INIT_CKPT` with `model.cfg.mask_source=learned` |
| learned + random placement | as above, plus `model.cfg.place_noise=3.0` |
| iterative model B | `experiment=maskfree/iterative model.cfg.corrupt_input=True` |

Ablations in the paper are evaluated on the BraTS21 validation split of fold 0, with the benchmark's threshold
selection.

## 7. Other scripts

- `scripts/timing.py`: reconstruction time per volume
- `scripts/make_figs.py`, `scripts/make_fig_qual.py`: figures

## Citation

If you use this code, please cite our paper (reference to be added on publication) and the cDDPM benchmark:

```bibtex
@article{behrendt2025cddpm,
  author  = {Behrendt, Finn and Bhattacharya, Debayan and Mieling, Robin and Maack, Lennart and Kr{\"u}ger, Julia and Opfer, Roland and Schlaefer, Alexander},
  title   = {Guided reconstruction with conditioned diffusion models for unsupervised anomaly detection in brain {MRIs}},
  journal = {Computers in Biology and Medicine},
  volume  = {186},
  pages   = {109660},
  year    = {2025},
  doi     = {10.1016/j.compbiomed.2025.109660}
}
```

## License

Our code is released under the MIT License (see `LICENSE`). The cDDPM benchmark code is not part of this repository
and remains under the terms of its authors.
