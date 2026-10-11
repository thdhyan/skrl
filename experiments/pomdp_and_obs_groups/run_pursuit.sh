#!/bin/bash
# Observation-group ablation: pursuers learn (MAPPO) vs a frozen random evader, so the only variable is obs_mode.
export WANDB_MODE=offline WANDB_SILENT=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
EX=$(git rev-parse --show-toplevel)/examples/pursuit_evasion/torch_pursuit_evasion_selfplay.py
mkdir -p logs
run() {
  python $EX --grid 12 12 --n-pursuers 2 --capture-radius 0 --max-steps 50 --num-envs 32 \
    --learning-teams pursuers --obs-mode $1 --seed $2 --generations 6 --timesteps-per-generation 800 \
    --eval-episodes 32 --eval-max-opponents 1 --rollouts 32 --hidden 128 128 --device cpu \
    --checkpoint-dir runs/$1_s$2 --wandb --wandb-project skrl-pursuit-evasion --wandb-group obs_ablation_$1 \
    --wandb-name $1_s$2 > logs/$1_s$2.log 2>&1
}
for seed in 1 2; do
  run full_state $seed &
  run global_potential $seed &
  run local_potential $seed &
  wait
done
echo done
