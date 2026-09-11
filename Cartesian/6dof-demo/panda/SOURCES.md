# Where the Panda model comes from

| | |
|---|---|
| Robot | Franka Emika Panda, 7 revolute joints + parallel gripper |
| Source | `bulletphysics/bullet3`, `examples/pybullet/gym/pybullet_data/franka_panda/` |
| Commit | `63c4d67e337017f9d8b298c900e9aabdb69296e7` |
| Files | `panda.urdf`, `LICENSE.txt`, `meshes/visual/*` (OBJ + MTL + colors.png), `meshes/collision/*.obj` |
| Licence | Apache License 2.0 (`LICENSE.txt` in this directory, copied verbatim) |
| Fetched | 2026-09-11, by `fetch_panda.py` |

Notes: `meshes/visual/link7.obj` is an empty placeholder upstream, so the renderer uses
`meshes/collision/link7.obj` for that link (and `collision/link0.obj` for the base, which
has no visual mesh). Nothing in the model is modified; the animation in Lesson 8 only
plays joint angles computed by `../trajectories.py`.
