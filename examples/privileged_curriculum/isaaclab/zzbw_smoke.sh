#!/usr/bin/env bash
# Isaac Lab smoke run of PrivilegedPPO (cartpole_privileged_smoke.py) on zz-bw, in the pe Isaac Lab SIF
# (Isaac Sim 6.1 + Isaac Lab 3.0) with this skrl fork on PYTHONPATH. Do not run Isaac Lab on the laptop.
#
# From the laptop:
#   rsync -az --delete -e "ssh -o ClearAllForwardings=yes" --exclude .venv --exclude .git --exclude runs \
#       --exclude __pycache__ ~/Projects/skrl/ zz-bw:Projects/skrl/
#   ssh zz-bw SKRL_GPU=1 ~/Projects/skrl/examples/privileged_curriculum/isaaclab/zzbw_smoke.sh run [script args]
#   ssh zz-bw ~/Projects/skrl/examples/privileged_curriculum/isaaclab/zzbw_smoke.sh frame <clip.mp4 under runs/>
#
# Outputs: /xtra/thakk100/skrl/runs (= runs/ in the container), log /xtra/thakk100/skrl/logs/smoke.log.
set -euo pipefail

SIF=${SKRL_SIF:-/xtra/thakk100/containers/pe-isaaclab.sif}
SKRL=$HOME/Projects/skrl
PE=$HOME/Projects/ebasa/pe
OUT=/xtra/thakk100/skrl
CACHE=/export/scratch/thakk100/pe-cache
PY=/workspace/isaaclab/_isaac_sim/python.sh
mkdir -p "$OUT/runs" "$OUT/logs" "$CACHE"/{home,kit-cache,kit-data,tmp} "$SKRL/runs"

mode=${1:-run}; shift || true
case "$mode" in
  run)
    inner=("$PY" examples/privileged_curriculum/isaaclab/cartpole_privileged_smoke.py
           --out runs/isaaclab_cartpole_privileged "$@") ;;
  frame)
    # frame <clip.mp4 under runs/> [out.png] [seconds]
    # python.sh changes directory: make the paths absolute in the container
    clip=/workspace/skrl/$1
    png=/workspace/skrl/${2:-${1%.mp4}.png}
    inner=("$PY" -c 'import imageio_ffmpeg, subprocess, sys; subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", sys.argv[3], "-i", sys.argv[1], "-frames:v", "1", sys.argv[2]], check=True)' "$clip" "$png" "${3:-2}") ;;
  *) echo "usage: $0 run [args...] | frame <mp4> [png] [sec]" >&2; exit 2 ;;
esac

cmd=(singularity exec --nv --containall --cleanenv
  --home "$CACHE/home:/root"
  --bind "$SKRL:/workspace/skrl"
  --bind "$OUT/runs:/workspace/skrl/runs"
  --bind "$PE:/workspace/pe"
  --bind "$CACHE/kit-cache:/isaac-sim/kit/cache"
  --bind "$CACHE/kit-data:/isaac-sim/kit/data"
  --bind "$CACHE/tmp:/tmp"
  --pwd /workspace/skrl
  --env CUDA_VISIBLE_DEVICES="${SKRL_GPU:-0}"
  --env OMNI_KIT_ACCEPT_EULA=YES --env ACCEPT_EULA=Y
  --env PYTHONPATH=/workspace/skrl
  --env PE_DEBUG_VIDEO=/workspace/pe/isaaclab_pe/common/debug_video.py
  "$SIF" "${inner[@]}")

if [ "$mode" = run ]; then
  tmux new-session -d -s skrl-smoke "$(printf '%q ' "${cmd[@]}") > $OUT/logs/smoke.log 2>&1"
  echo "started tmux session skrl-smoke, log: $OUT/logs/smoke.log"
else
  exec "${cmd[@]}"
fi
