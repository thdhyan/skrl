#!/bin/bash
export WANDB_MODE=offline WANDB_SILENT=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
EX=$(git rev-parse --show-toplevel)/examples/contact_rich_uncertainty/cartpole_pomdp_belief_ppo.py
run() { python -c "import torch; torch.set_num_threads(1)" ; python $EX --headless --timesteps 30000 --seed $2 $3 --wandb --wandb-project skrl-cartpole-pomdp --log-dir runs/$1_s$2 > logs/$1_s$2.log 2>&1; }
mkdir -p logs
for seed in 1 2 3; do
  run full_obs $seed --full-obs &
  run masked $seed --baseline &
  run masked_belief $seed "" &
  wait
done
echo done
