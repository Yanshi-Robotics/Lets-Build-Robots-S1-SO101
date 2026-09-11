# Six degrees of freedom, one animation each

The animated figures in Lesson 8 that show a tool moving along or turning about one axis
at a time. Everything needed to make them again is here; the finished clips are committed
so the course site can copy them byte for byte.

| Clip | Shows |
|---|---|
| `dof-x.webp`, `dof-y.webp`, `dof-z.webp` | the SO-101 tool sliding ±4 cm along one base axis (base axes drawn at the origin) |
| `dof-pitch.webp`, `dof-roll.webp` | the SO-101 tool turning ±25° about its own y axis, ±40° about its approach axis, position held |
| `dof-yaw-panda.webp` | a Franka Panda turning its hand ±40° about the vertical axis with the hand's position held — the SO-101 has no joint for this |
| `jet-roll.webp`, `jet-pitch.webp`, `jet-yaw.webp` | a schematic aircraft: the three words come from flying |
| `gimbal-lock.webp` | three nested rings; the middle one turned to 90° lays the inner pivot on the outer axis, and one degree of freedom is gone |

Roll is drawn red, pitch green, yaw blue, the colours of the x, y, z axes they turn about.

## Making them

```sh
python Cartesian/6dof-demo/panda/fetch_panda.py                                   # once: the Panda model (Apache-2.0)
python Cartesian/6dof-demo/trajectories.py --model-dir models/so101               # joint angles per frame -> trajectories.json
blender --background --python Cartesian/6dof-demo/render.py -- --model-dir models/so101 --demo-dir Cartesian/6dof-demo
```

`trajectories.py` uses the course's own solver (`Cartesian/solver.py`) for the five SO-101
clips and placo directly for the Panda; every clip is a ping-pong that ends where it starts.
`render.py` runs inside Blender (5.2, Workbench renderer, CPU): it reads the URDFs and the
JSON, draws the arrows and captions, renders 15 fps frames with a transparent background and
lets `ffmpeg` pack each clip into an animated WebP plus a poster PNG. `manifest.json` records
the pinned model commits and the SHA-256 of every output; `--preview` renders three frames
per clip to check the framing.

## Licences

The SO-101 model is Apache-2.0 (TheRobotStudio/SO-ARM100, fetched by `Cartesian/fetch_model.py`).
The Panda model is Apache-2.0 (Franka Emika's description, taken from PyBullet's copy in
bulletphysics/bullet3; `panda/LICENSE.txt` and `panda/SOURCES.md`). The renders, the aircraft
and the rings are this course's own work and carry the repository licence. None of it is
hardware evidence: the clips play computed joint angles on models.
