#!/bin/bash
# Full 5-fold run for one contrast: per fold a VAE, its latent scale, the one-step and the iterative model.
# Usage (repo root, after sourcing pc_environment.env):  bash scripts/cross_validate.sh t2      (and t1)
set -e
MODE=$1; FOLDS=${FOLDS:-"0 1 2 3 4"}
R=$LOG_DIR/logs/runs
ck() { ls -t $R/$1/*/checkpoints/*.ckpt 2>/dev/null | grep -v last | head -1; }
C="datamodule.cfg.mode=$MODE"
for k in $FOLDS; do
  F="num_folds=$((k+1)) ++start_fold=$k"
  python run.py experiment=maskfree/vae $C $F name=vae_${MODE}_f$k
  export INIT_CKPT=$(ck vae_${MODE}_f$k)
  export LATENT_SCALE=$(python scripts/latent_stats_fold.py $INIT_CKPT $MODE $k | grep -oP 'latent_scale: \K[0-9.]+')
  python run.py experiment=maskfree/onestep   $C $F name=onestep_${MODE}_f$k
  python run.py experiment=maskfree/iterative $C $F name=iterative_${MODE}_f$k
done
