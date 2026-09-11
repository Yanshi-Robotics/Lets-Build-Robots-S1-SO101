"""Download the Franka Emika Panda model used for the yaw animation in Lesson 8.

    python Cartesian/6dof-demo/panda/fetch_panda.py

Source: the plain-URDF copy that ships with PyBullet, in bulletphysics/bullet3 on GitHub,
pinned to one commit. It is Apache-2.0 (Franka Emika's franka_description); the licence
text is downloaded alongside and also committed here as LICENSE.txt. The URDF and the
meshes (about 9 MB of OBJ) are not committed: run this once and they land next to it.

The SO-101 cannot yaw its tool without moving it; the Panda, with seven joints, can. That
is the only reason a second arm appears in the course.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

REPO = "bulletphysics/bullet3"
COMMIT = "63c4d67e337017f9d8b298c900e9aabdb69296e7"
BASE_URL = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/examples/pybullet/gym/pybullet_data/franka_panda/"
FILES = ["panda.urdf", "LICENSE.txt"] + \
        [f"meshes/visual/{n}" for n in ("colors.png", "finger.mtl", "finger.obj", "hand.mtl", "hand.obj",
                                         "link1.mtl", "link1.obj", "link2.mtl", "link2.obj", "link3.mtl", "link3.obj",
                                         "link4.mtl", "link4.obj", "link5.mtl", "link5.obj", "link6.mtl", "link6.obj")] + \
        [f"meshes/collision/{n}" for n in ("finger.obj", "hand.obj", "link0.obj", "link1.obj", "link2.obj", "link3.obj",
                                            "link4.obj", "link5.obj", "link6.obj", "link7.obj")]
HERE = Path(__file__).resolve().parent
TIMEOUT_S = 60


def fetch(target_dir: Path = HERE, output=print) -> int:
    downloaded = 0
    for rel in FILES:
        target = target_dir / rel
        if target.exists() and target.stat().st_size > 0:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = BASE_URL + rel
        output(f"GET {url}")
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
            target.write_bytes(response.read())
        downloaded += 1
    output(f"{target_dir}: {len(FILES)} files present ({downloaded} downloaded), commit {COMMIT[:12]}")
    return downloaded


if __name__ == "__main__":
    try:
        fetch()
    except OSError as e:
        print(f"download failed: {e}", file=sys.stderr)
        sys.exit(1)
