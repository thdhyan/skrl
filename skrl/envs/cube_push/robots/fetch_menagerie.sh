#!/bin/bash
# Fetch the MuJoCo Menagerie models used as low-level embodiments (Apache-2.0).
# Sparse-checkout: only the two arm folders land in assets/menagerie/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/../assets/menagerie"
if [ -f "$DEST/franka_emika_panda/panda_nohand.xml" ] && [ -f "$DEST/universal_robots_ur5e/ur5e.xml" ]; then
  echo "menagerie arms already present in $DEST"
  exit 0
fi
rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$DEST"
cd "$DEST"
git sparse-checkout set franka_emika_panda universal_robots_ur5e
git checkout
ls franka_emika_panda/panda_nohand.xml universal_robots_ur5e/ur5e.xml
