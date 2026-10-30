# Mask-Free Unsupervised Anomaly Detection in Brain MRI via One-Step Conditional Latent Restoration

Code for the paper by Nijhawan, Kapoor, Pabri, Rawal and Bhalla (Thapar Institute of Engineering and Technology).

A conditional network in the latent space of a VAE is trained, on healthy scans only, to map slices corrupted by
randomly placed synthetic anomalies back to their healthy latent code. At test time it is applied once to the latent
of the test slice, and the anomaly map is measured against the VAE's own reconstruction (VAE-error cancellation).
The same network trained as an iterative latent diffusion model is included for comparison.

The method is implemented inside the benchmark of Behrendt et al. (*Guided reconstruction with conditioned diffusion
models for unsupervised anomaly detection in brain MRIs*, Computers in Biology and Medicine 186, 2025), so data
loading, splits, post-processing and metrics are theirs. This repository contains only our additions and the
compatibility changes to their code. Please also cite their work and respect `LICENSE-upstream`.

## 1. Setup
```bash
# run from the folder that contains maskfree_release/
git clone $(cat maskfree_release/UPSTREAM_URL.txt) cddpm && cd cddpm
git checkout $(cat ../maskfree_release/UPSTREAM_COMMIT.txt)
git apply ../maskfree_release/compat.patch     # Lightning 2.x / TorchIO 1.x fixes, T1 test sets (ATLAS, WMH)
cp -r ../maskfree_release/src ../maskfree_release/configs ../maskfree_release/scripts .
pip install ema-pytorch                         # plus the upstream requirements
```
Tested with Python 3.10, PyTorch 2.6 (CUDA 12.4), PyTorch Lightning 2.6, TorchIO 1.2, Hydra 1.3, MONAI.

## 2. Data
Download the preprocessed IXI, BraTS21, MSLUB, ATLAS v2.0 and WMH data and split files as described upstream
(we do not redistribute data). Create `pc_environment.env` with `DATA_DIR=...` and `LOG_DIR=...`, then
`set -a; source pc_environment.env; set +a; export WANDB_MODE=offline`.

## 3. Five-fold cross-validation (as in the paper)
```bash
bash scripts/cross_validate.sh t2      # BraTS21, MSLUB
bash scripts/cross_validate.sh t1      # ATLAS, WMH
python scripts/cv_summary.py           # mean +- std over folds (Dice, AUPRC, l1-ratio)
```
Per fold this trains a VAE (150 epochs), estimates its latent scale, and trains and tests the one-step and the
iterative model (2400 epochs each). Each conditional model takes under an hour on an H100 MIG slice.
Note: the T1 runs also score the T1 scans of BraTS21/MSLUB; the paper reports only ATLAS and WMH for T1.

## 4. A single model
```bash
python run.py experiment=maskfree/vae datamodule.cfg.mode=t2 name=vae_t2
export INIT_CKPT=<VAE checkpoint>
export LATENT_SCALE=$(python scripts/latent_stats_fold.py $INIT_CKPT t2 0 | grep -oP 'latent_scale: \K[0-9.]+')
python run.py experiment=maskfree/onestep datamodule.cfg.mode=t2 name=onestep_t2
```
Evaluate an existing checkpoint only: set `INIT_CKPT` to it and add
`trainer.max_epochs=0 trainer.min_epochs=0 ckpt_path=null`.
Results: `<run dir>/<fold>_preds_dict.pkl`; the paper reports `DicePerVolMean` and `AUPRCPerVolMean` of the `test` entries.

## 5. Ablations (Table 5), as overrides of `experiment=maskfree/onestep`
| Ablation | Override |
|---|---|
| without condition channel | train `model.cfg.p_drop_cond=1.0`, evaluate with `model.cfg.guidance_w=0.0` |
| without VAE-error cancellation | `model.cfg.cancel_vae_error=False` (evaluation only) |
| without condition dropout | `model.cfg.p_drop_cond=0.0` |
| stronger amplification | `model.cfg.amp=[3.0,6.0]` |
| learned placement | train `experiment=maskfree/generator_ablation` first, then `model.cfg.mask_source=learned` with its checkpoint as `INIT_CKPT` (learned + random: also `model.cfg.place_noise=3.0`) |
| iterative model B | `experiment=maskfree/iterative model.cfg.corrupt_input=True` |

Ablations in the paper use the BraTS21 validation split of fold 0, with the benchmark's threshold selection.

## 6. Other scripts
`scripts/timing.py` (Table 3), `scripts/make_figs.py`, `scripts/make_fig_qual.py` (figures).
