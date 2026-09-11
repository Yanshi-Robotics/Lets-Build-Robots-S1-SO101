"""Download the SO-101 model the Cartesian programs load: one URDF, its meshes, the licence.

    python Cartesian/fetch_model.py --model-dir models/so101

Source: TheRobotStudio/SO-ARM100 on GitHub, pinned to one commit so that every student
gets the same file the solver and tests were written against (Apache-2.0, licence file
downloaded alongside). Files already present are left alone; run again to fill gaps.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO = "TheRobotStudio/SO-ARM100"
COMMIT = "7629d2ad9853d10fb903093a33ef6114099d97e5"
BASE_URL = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/"
URDF = "Simulation/SO101/so101_new_calib.urdf"
MESHES = (
    "base_motor_holder_so101_v1.stl", "base_so101_v2.stl", "motor_holder_so101_base_v1.stl",
    "motor_holder_so101_wrist_v1.stl", "moving_jaw_so101_v1.stl", "rotation_pitch_so101_v1.stl",
    "sts3215_03a_no_horn_v1.stl", "sts3215_03a_v1.stl", "under_arm_so101_v1.stl", "upper_arm_so101_v1.stl",
    "waveshare_mounting_plate_so101_v2.stl", "wrist_roll_follower_so101_v1.stl", "wrist_roll_pitch_so101_v2.stl",
)
# (path in the upstream repository, path under --model-dir)
FILES = [(URDF, "so101_new_calib.urdf"), ("LICENSE", "LICENSE")] + \
        [(f"Simulation/SO101/assets/{m}", f"assets/{m}") for m in MESHES]
TIMEOUT_S = 60


def fetch(model_dir: Path, output=print) -> int:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "assets").mkdir(exist_ok=True)
    downloaded = 0
    for remote, local in FILES:
        target = model_dir / local
        if target.exists() and target.stat().st_size > 0:
            continue
        url = BASE_URL + remote
        output(f"GET {url}")
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
            target.write_bytes(response.read())
        downloaded += 1
    output(f"{model_dir}: {len(FILES)} files present ({downloaded} downloaded), commit {COMMIT[:12]}")
    return downloaded


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model-dir", type=Path, required=True, help="where to put the model, e.g. models/so101")
    args = p.parse_args(argv)
    try:
        fetch(args.model_dir)
    except OSError as e:
        print(f"download failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
