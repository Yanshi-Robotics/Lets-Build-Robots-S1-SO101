"""Render the six-degree-of-freedom animations for Lesson 8 (Blender, Workbench, CPU).

    blender --background --python Cartesian/6dof-demo/render.py -- \
        --model-dir models/so101 --demo-dir Cartesian/6dof-demo [--preview]

Reads `trajectories.json` (from trajectories.py) and writes, into --demo-dir, ten animated
WebP clips with a poster PNG each and a manifest.json:

    dof-x / dof-y / dof-z         the SO-101 tool sliding along one base axis
    dof-pitch / dof-roll          the SO-101 tool turning about its own y / approach axis
    dof-yaw-panda                 a Franka Panda turning its hand about the vertical axis
                                  with the hand's position fixed (the SO-101 has no joint for this)
    jet-roll / jet-pitch / jet-yaw a schematic aircraft: where the three words come from
    gimbal-lock                   three nested rings; the middle one turned to 90 degrees
                                  brings the outer and inner axes into line

Everything is a teaching render of the pinned models, not hardware evidence. Transparent
background, orthographic camera, 15 fps, each clip a ping-pong loop. --preview renders
three frames per clip for checking the framing.
"""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

WIDTH, HEIGHT = 640, 420
FPS = 15
SO101_URDF = "so101_new_calib.urdf"
SO101_COMMIT = "7629d2ad9853d10fb903093a33ef6114099d97e5"
PANDA_COMMIT = "63c4d67e337017f9d8b298c900e9aabdb69296e7"
AXIS_COLORS = {"x": (0.84, 0.19, 0.27, 1), "y": (0.16, 0.63, 0.35, 1), "z": (0.19, 0.39, 0.86, 1)}
# roll is about x, pitch about y, yaw about z: the rotations borrow their axis colours
ROT_COLORS = {"roll": AXIS_COLORS["x"], "pitch": AXIS_COLORS["y"], "yaw": AXIS_COLORS["z"]}
INK = (0.11, 0.13, 0.16, 1)
HIGHLIGHT = (0.88, 0.32, 0.08, 1)


def smooth(t):
    return t * t * (3 - 2 * t)


def main():
    import bpy
    from mathutils import Euler, Matrix, Vector

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="clip names to render (default: all)")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    frames_root = args.frames_dir or (args.demo_dir / "frames")
    trajectories = json.loads((args.demo_dir / "trajectories.json").read_text(encoding="utf-8"))

    # ---- scene -------------------------------------------------------------------------
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_cavity = True
    scene.display.shading.show_shadows = True
    scene.display.shading.shadow_intensity = 0.35
    scene.display.render_aa = "8"
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x, scene.render.resolution_y = WIDTH, HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True

    materials = {}

    def material(name, color):
        if name not in materials:
            m = bpy.data.materials.new(name)
            m.diffuse_color = color
            materials[name] = m
        return materials[name]

    def clear(keep=()):
        for obj in list(bpy.data.objects):
            if obj not in keep:
                bpy.data.objects.remove(obj, do_unlink=True)

    # ---- URDF helpers --------------------------------------------------------------------
    def origin_matrix(origin):
        if origin is None:
            return Matrix.Identity(4)
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
        return Matrix.Translation(Vector(xyz)) @ Euler(rpy, "XYZ").to_matrix().to_4x4()

    def forward(robot, root, angles):
        worlds = {root: Matrix.Identity(4)}
        pending = list(robot.findall("./joint"))
        while pending:
            progress = False
            for joint in pending[:]:
                parent = joint.find("parent").get("link")
                if parent not in worlds:
                    continue
                matrix = origin_matrix(joint.find("origin"))
                kind = joint.get("type")
                axis_el = joint.find("axis")
                axis = Vector([float(v) for v in axis_el.get("xyz").split()]) if axis_el is not None else Vector((0, 0, 1))
                value = angles.get(joint.get("name"), 0.0)
                if kind in ("revolute", "continuous"):
                    matrix = matrix @ Matrix.Rotation(value, 4, axis)
                elif kind == "prismatic":
                    matrix = matrix @ Matrix.Translation(axis * value)
                worlds[joint.find("child").get("link")] = worlds[parent] @ matrix
                pending.remove(joint)
                progress = True
            if not progress:
                raise ValueError("URDF contains an unresolved joint chain")
        return worlds

    def import_mesh(path, mat):
        path = Path(path)
        if path.suffix.lower() == ".stl":
            bpy.ops.wm.stl_import(filepath=str(path), forward_axis="Y", up_axis="Z")
        else:
            bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z")
        objs = [o for o in bpy.context.selected_objects if o.type == "MESH"]
        for o in objs:
            o.data.materials.clear()
            o.data.materials.append(mat)
        return objs

    def load_so101():
        urdf = args.model_dir / SO101_URDF
        robot = ET.parse(urdf).getroot()
        parts = []
        for link in robot.findall("./link"):
            for visual in link.findall("visual"):
                source = visual.find("geometry/mesh").get("filename")
                mat = material("motor", (0.05, 0.06, 0.07, 1)) if "sts3215" in source else material("print", (0.56, 0.62, 0.66, 1))
                for obj in import_mesh(args.model_dir / source, mat):
                    parts.append((obj, link.get("name"), origin_matrix(visual.find("origin"))))
        return robot, "base_link", parts

    def load_panda():
        pdir = args.demo_dir / "panda"
        robot = ET.parse(pdir / "panda.urdf").getroot()
        parts = []
        white = material("panda_white", (0.92, 0.92, 0.9, 1))
        dark = material("panda_dark", (0.2, 0.2, 0.22, 1))
        for link in robot.findall("./link"):
            name = link.get("name")
            visual = link.find("visual")
            if visual is None:
                continue
            source = visual.find("geometry/mesh").get("filename").replace("package://", "")
            path = pdir / source
            if not path.exists() or path.stat().st_size < 100:     # link7's visual is an empty placeholder upstream
                path = pdir / source.replace("visual", "collision")
            mat = dark if ("link0" in source or "finger" in source) else white
            for obj in import_mesh(path, mat):
                parts.append((obj, name, origin_matrix(visual.find("origin"))))
        return robot, "panda_link0", parts

    def pose(robot, root, parts, angles):
        worlds = forward(robot, root, angles)
        for obj, link, origin in parts:
            obj.matrix_world = worlds[link] @ origin
        bpy.context.view_layer.update()
        return worlds

    # ---- drawing helpers ----------------------------------------------------------------
    def tube(points, radius, mat, name="tube"):
        curve = bpy.data.curves.new(name, "CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = radius
        curve.bevel_resolution = 6
        curve.fill_mode = "FULL"
        spline = curve.splines.new("POLY")
        spline.points.add(len(points) - 1)
        for p, v in zip(spline.points, points):
            p.co = (v.x, v.y, v.z, 1.0)
        obj = bpy.data.objects.new(name, curve)
        obj.data.materials.append(mat)
        bpy.context.collection.objects.link(obj)
        return obj

    def cone(tip, direction, length, radius, mat):
        direction = direction.normalized()
        bpy.ops.mesh.primitive_cone_add(radius1=radius, radius2=0.0, depth=length,
                                        location=tip - direction * (length / 2))
        obj = bpy.context.object
        obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
        obj.data.materials.append(mat)
        return obj

    def straight_arrow(a, b, mat, radius=0.004, head=0.02, both=True):
        d = (b - a).normalized()
        objs = [tube([a + d * head if both else a, b - d * head], radius, mat), cone(b, d, head, radius * 3, mat)]
        if both:
            objs.append(cone(a, -d, head, radius * 3, mat))
        return objs

    def arc_arrow(center, axis, start_dir, radius, angle, mat, tube_radius=0.004, head=0.018, both=True, steps=32):
        """Arc about `axis` through `center`, from -angle/2 to +angle/2 around start_dir, arrowheads at both ends."""
        axis = axis.normalized()
        u = (start_dir - axis * start_dir.dot(axis)).normalized()
        pts = []
        for i in range(steps + 1):
            a = -angle / 2 + angle * i / steps
            pts.append(center + (Matrix.Rotation(a, 3, axis) @ u) * radius)
        objs = [tube(pts, tube_radius, mat)]
        tangent_end = (pts[-1] - pts[-2]).normalized()
        objs.append(cone(pts[-1] + tangent_end * head * 0.6, tangent_end, head, tube_radius * 3, mat))
        if both:
            tangent_start = (pts[0] - pts[1]).normalized()
            objs.append(cone(pts[0] + tangent_start * head * 0.6, tangent_start, head, tube_radius * 3, mat))
        return objs

    def axes_triad(origin, length, cam_rot, letters=("x", "y", "z"), radius=0.003):
        objs = []
        for name, direction in zip(("x", "y", "z"), (Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1)))):
            mat = material(f"axis_{name}", AXIS_COLORS[name])
            objs += straight_arrow(origin, origin + direction * length, mat, radius=radius, head=length * 0.18, both=False)
            objs.append(text(letters["xyz".index(name)], origin + direction * (length * 1.12), cam_rot, mat, size=length * 0.28))
        return objs

    def text(body, location, cam_rot, mat, size=0.03):
        bpy.ops.object.text_add()
        obj = bpy.context.object
        obj.data.body = body
        obj.data.align_x = "CENTER"
        obj.data.size = size
        obj.rotation_euler = cam_rot
        obj.location = location
        obj.data.materials.append(mat)
        return obj

    def fit_camera(points, direction=Vector((0.45, -0.85, 0.5)), margin=1.15, aspect=WIDTH / HEIGHT):
        low = Vector(tuple(min(p[i] for p in points) for i in range(3)))
        high = Vector(tuple(max(p[i] for p in points) for i in range(3)))
        target = (low + high) / 2
        bpy.ops.object.camera_add(location=target + direction)
        cam = bpy.context.object
        cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam.data.type = "ORTHO"
        cam.data.clip_start = 0.001
        scene.camera = cam
        local = [cam.rotation_euler.to_matrix().transposed() @ (p - target) for p in points]
        cam.data.ortho_scale = max(max(p.x for p in local) - min(p.x for p in local),
                                   (max(p.y for p in local) - min(p.y for p in local)) * aspect) * margin
        return cam, target

    def cam_vectors(cam):
        R = cam.rotation_euler.to_matrix()
        return R @ Vector((1, 0, 0)), R @ Vector((0, 1, 0)), R @ Vector((0, 0, 1))

    def title(body, cam, target, size=None):
        """A caption line inside the top edge of the camera frame, floated just in front of the camera."""
        right, up, toward = cam_vectors(cam)
        half_height = cam.data.ortho_scale / (WIDTH / HEIGHT) / 2
        size = size or half_height * 0.11
        return text(body, target + up * (half_height - size * 1.5) + toward * ((cam.location - target).length * 0.9),
                    cam.rotation_euler, material("ink", INK), size=size)

    def render_clip(name, n_frames, draw_frame):
        frames_dir = frames_root / name
        frames_dir.mkdir(parents=True, exist_ok=True)
        for old in frames_dir.glob("*.png"):
            old.unlink()
        indices = [0, n_frames // 4, n_frames // 2] if args.preview else range(n_frames)
        for out_index, i in enumerate(indices):
            draw_frame(i)
            scene.render.filepath = str(frames_dir / f"{out_index:04d}.png")
            bpy.ops.render.render(write_still=True)
        webp = args.demo_dir / f"{name}.webp"
        poster = args.demo_dir / f"{name}-poster.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames_dir / "%04d.png"),
                        "-c:v", "libwebp_anim", "-lossless", "0", "-q:v", "62", "-loop", "0", "-pix_fmt", "yuva420p", str(webp)], check=True)
        # shutil, not cp: there is no cp executable on Windows.
        shutil.copyfile(frames_dir / "0000.png", poster)
        print(f"{name}: {len(list(indices))} frames -> {webp.stat().st_size // 1024} KB", flush=True)
        return [webp, poster]

    wanted = set(args.only) if args.only else None
    outputs = []

    # ---- SO-101 clips -------------------------------------------------------------------
    so_names = {"x": "dof-x", "y": "dof-y", "z": "dof-z", "pitch": "dof-pitch", "roll": "dof-roll"}
    labels = {"x": "x: forward / back", "y": "y: left / right", "z": "z: up / down",
              "pitch": "pitch: nose down / up", "roll": "roll: about the approach axis"}
    for key, clip_name in so_names.items():
        if wanted and clip_name not in wanted:
            continue
        clear()
        robot, root, parts = load_so101()
        clip = trajectories["clips"][key]
        frames = clip["frames"]
        joints = clip["joints"]
        # camera over every pose of the clip
        points = []
        for row in frames[::6]:
            pose(robot, root, parts, dict(zip(joints, row)))
            points.extend(obj.matrix_world @ Vector(p) for obj, *_ in parts for p in obj.bound_box)
        cam, target = fit_camera(points, margin=1.22)
        right, up, toward = cam_vectors(cam)
        # base axes, fixed in the world
        axes_triad(Vector((0, 0, 0)) + toward * 0.3, 0.09, cam.rotation_euler)
        # the motion arrow, fixed in the world around the tool's rest position
        p0 = Vector(clip["tool_position"][0])
        R0 = Matrix([[r[0], r[1], r[2]] for r in clip["tool_rotation"][0]])
        if key in ("x", "y", "z"):
            d = {"x": Vector((1, 0, 0)), "y": Vector((0, 1, 0)), "z": Vector((0, 0, 1))}[key]
            straight_arrow(p0 - d * 0.075, p0 + d * 0.075, material(f"axis_{key}", AXIS_COLORS[key]), radius=0.005, head=0.025)
        elif key == "pitch":
            arc_arrow(p0, R0 @ Vector((0, 1, 0)), R0 @ Vector((0, 0, 1)), 0.07, math.radians(80), material("rot_pitch", ROT_COLORS["pitch"]), tube_radius=0.005, head=0.022)
        else:
            approach = R0 @ Vector((0, 0, 1))
            arc_arrow(p0 + approach * 0.035, approach, R0 @ Vector((1, 0, 0)), 0.045, math.radians(280), material("rot_roll", ROT_COLORS["roll"]), tube_radius=0.005, head=0.022)
        title(labels[key], cam, target)

        def draw(i, frames=frames, joints=joints, robot=robot, root=root, parts=parts):
            pose(robot, root, parts, dict(zip(joints, frames[i])))
        outputs += render_clip(clip_name, len(frames), draw)

    # ---- Panda yaw ----------------------------------------------------------------------
    if "yaw" in trajectories["clips"] and (not wanted or "dof-yaw-panda" in wanted):
        clear()
        robot, root, parts = load_panda()
        clip = trajectories["clips"]["yaw"]
        frames, joints = clip["frames"], clip["joints"]
        points = []
        for row in frames[::6]:
            pose(robot, root, parts, {**dict(zip(joints, row)), "panda_finger_joint1": 0.02, "panda_finger_joint2": 0.02})
            points.extend(obj.matrix_world @ Vector(p) for obj, *_ in parts for p in obj.bound_box)
        cam, target = fit_camera(points, direction=Vector((0.9, -1.1, 0.7)), margin=1.08)
        right, up, toward = cam_vectors(cam)
        axes_triad(Vector((0, 0, 0)) + toward * 0.3, 0.16, cam.rotation_euler, radius=0.005)
        p0 = Vector(clip["tool_position"][0])
        arc_arrow(p0 + Vector((0, 0, 0.12)), Vector((0, 0, 1)), Vector((1, 0, 0)), 0.18, math.radians(110), material("rot_yaw", ROT_COLORS["yaw"]), tube_radius=0.008, head=0.035)
        title("yaw: Franka Panda, 7 joints. The hand turns, its position stays", cam, target)

        def draw(i, frames=frames, joints=joints, robot=robot, root=root, parts=parts):
            pose(robot, root, parts, {**dict(zip(joints, frames[i])), "panda_finger_joint1": 0.02, "panda_finger_joint2": 0.02})
        outputs += render_clip("dof-yaw-panda", len(frames), draw)

    # ---- jet ----------------------------------------------------------------------------
    def build_jet():
        """A schematic aircraft, nose along +x, wings along y, fin up z. Returns the parent empty."""
        bpy.ops.object.empty_add()
        root = bpy.context.object
        body = material("jet_body", (0.62, 0.66, 0.72, 1))
        dark = material("jet_dark", (0.25, 0.27, 0.32, 1))
        bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=0.7, location=(0, 0, 0), rotation=(0, math.pi / 2, 0))
        fus = bpy.context.object; fus.data.materials.append(body)
        bpy.ops.mesh.primitive_cone_add(radius1=0.05, radius2=0.0, depth=0.22, location=(0.46, 0, 0), rotation=(0, math.pi / 2, 0))
        nose = bpy.context.object; nose.data.materials.append(dark)
        bpy.ops.mesh.primitive_cone_add(vertices=3, radius1=0.42, radius2=0.42, depth=0.015, location=(-0.08, 0, -0.02))
        wing = bpy.context.object; wing.scale = (0.55, 1.0, 1.0); wing.rotation_euler = (0, 0, math.pi); wing.data.materials.append(body)
        bpy.ops.mesh.primitive_cube_add(size=1, location=(-0.3, 0, 0.11))
        fin = bpy.context.object; fin.scale = (0.14, 0.012, 0.16); fin.data.materials.append(dark)
        bpy.ops.mesh.primitive_cube_add(size=1, location=(-0.32, 0, 0.0))
        stab = bpy.context.object; stab.scale = (0.1, 0.36, 0.012); stab.data.materials.append(body)
        for o in (fus, nose, wing, fin, stab):
            o.parent = root
        return root

    jet_specs = {"jet-roll": ("roll", Vector((1, 0, 0)), math.radians(35), "roll: about the nose-tail axis (x)"),
                 "jet-pitch": ("pitch", Vector((0, 1, 0)), math.radians(25), "pitch: nose up / down (y)"),
                 "jet-yaw": ("yaw", Vector((0, 0, 1)), math.radians(30), "yaw: nose left / right (z)")}
    n_jet = round(FPS * trajectories["seconds"])
    for clip_name, (rot, axis, amp, label) in jet_specs.items():
        if wanted and clip_name not in wanted:
            continue
        clear()
        jet = build_jet()
        bpy.context.view_layer.update()
        pts = [Vector((sx, sy, sz)) for sx in (-0.65, 0.65) for sy in (-0.5, 0.5) for sz in (-0.35, 0.35)]
        cam, target = fit_camera(pts, direction=Vector((0.9, -1.0, 0.65)), margin=1.1)
        right, up, toward = cam_vectors(cam)
        axes_triad(Vector((0, 0, 0)), 0.42, cam.rotation_euler, radius=0.006)
        mat = material(f"rot_{rot}", ROT_COLORS[rot])
        centre = {"roll": Vector((-0.55, 0, 0)), "pitch": Vector((0, 0.55, 0)), "yaw": Vector((0, 0, 0.42))}[rot]
        start = {"roll": Vector((0, 0, 1)), "pitch": Vector((1, 0, 0)), "yaw": Vector((1, 0, 0))}[rot]
        arc_arrow(centre, axis, start, 0.22, math.radians(150), mat, tube_radius=0.012, head=0.06)
        title(label, cam, target)

        def draw(i, jet=jet, axis=axis, amp=amp):
            w = math.sin(2 * math.pi * i / n_jet)
            jet.rotation_euler = Matrix.Rotation(amp * w, 3, axis).to_euler()
            bpy.context.view_layer.update()
        outputs += render_clip(clip_name, n_jet, draw)

    # ---- gimbal lock ----------------------------------------------------------------------
    if not wanted or "gimbal-lock" in wanted:
        clear()
        R_out, R_mid, R_in = 0.56, 0.42, 0.29
        # A gimbal: each ring pivots on the one outside it about a diameter of that ring.
        # outer ring: plane xz, axis z (yaw, blue). middle ring: plane xy, pivots about x (green).
        # inner ring: plane yz, pivots about y (roll, red). Turning the middle ring 90 deg about x
        # carries the inner pivot (y) onto the outer axis (z): that is gimbal lock.
        bpy.ops.mesh.primitive_torus_add(major_radius=R_out, minor_radius=0.022, rotation=(math.pi / 2, 0, 0))
        outer = bpy.context.object; outer.data.materials.append(material("rot_yaw", ROT_COLORS["yaw"]))
        bpy.ops.object.empty_add(); mid_pivot = bpy.context.object      # at the origin, unrotated: its local axes are the world's
        bpy.ops.mesh.primitive_torus_add(major_radius=R_mid, minor_radius=0.022, rotation=(0, 0, 0))
        middle = bpy.context.object; middle.data.materials.append(material("rot_pitch", ROT_COLORS["pitch"])); middle.parent = mid_pivot
        bpy.ops.object.empty_add(); in_pivot = bpy.context.object; in_pivot.parent = mid_pivot
        bpy.ops.mesh.primitive_torus_add(major_radius=R_in, minor_radius=0.022, rotation=(0, math.pi / 2, 0))
        inner = bpy.context.object; inner.data.materials.append(material("rot_roll", ROT_COLORS["roll"])); inner.parent = in_pivot
        # the payload: a small jet inside the inner ring, nose along the inner pivot
        jet = build_jet(); jet.scale = (0.3, 0.3, 0.3); jet.rotation_euler = (0, 0, math.pi / 2); jet.parent = in_pivot
        # axis rods: the outer ring's axis (z, fixed), the middle pivot (x, on the outer ring), the inner pivot (y, turns with the middle ring)
        ink = material("ink", INK)
        outer_axis = tube([Vector((0, 0, -0.78)), Vector((0, 0, 0.78))], 0.012, material("rot_yaw", ROT_COLORS["yaw"]))
        middle_axis = tube([Vector((-0.66, 0, 0)), Vector((0.66, 0, 0))], 0.012, material("rot_pitch", ROT_COLORS["pitch"]))
        inner_axis = tube([Vector((0, -0.66, 0)), Vector((0, 0.66, 0))], 0.012, material("rot_roll", ROT_COLORS["roll"]))
        inner_axis.parent = mid_pivot
        pts = [Vector((sx, sy, sz)) for sx in (-0.75, 0.75) for sy in (-0.75, 0.75) for sz in (-0.75, 0.75)]
        cam, target = fit_camera(pts, direction=Vector((1.2, -1.0, 0.6)), margin=0.82)
        right, up, toward = cam_vectors(cam)
        caption = title("", cam, target)
        n = round(FPS * 4.0)          # 0 -> 90 degrees, hold, back

        def draw(i, n=n):
            t = i / n
            if t < 0.35:
                pitch = math.radians(90) * smooth(t / 0.35)
            elif t < 0.65:
                pitch = math.radians(90)
            else:
                pitch = math.radians(90) * smooth(1 - (t - 0.65) / 0.35)
            mid_pivot.rotation_euler = (pitch, 0, 0)          # the middle ring turns about its pivot on the outer ring (x)
            aligned = abs(pitch - math.radians(90)) < math.radians(3)
            caption.data.body = ("pitch = 90 deg: the roll axis lies on the yaw axis. One freedom lost"
                                 if aligned else f"pitch {math.degrees(pitch):.0f} deg: three separate axes")
            caption.data.materials[0] = material("hl", HIGHLIGHT) if aligned else ink
            bpy.context.view_layer.update()
        outputs += render_clip("gimbal-lock", n, draw)

    # ---- manifest -----------------------------------------------------------------------
    if not wanted and not args.preview:
        manifest = {
            "generated": __import__("datetime").date.today().isoformat(),
            "kind": "3d-teaching-animation",
            "renderer": "Blender Workbench CPU",
            "hardwareEvidence": False,
            "so101Commit": SO101_COMMIT,
            "pandaCommit": PANDA_COMMIT,
            "fps": FPS,
            "clips": {
                "dof-x": "SO-101, tool along base x", "dof-y": "SO-101, tool along base y", "dof-z": "SO-101, tool along base z",
                "dof-pitch": "SO-101, tool pitch about its y axis", "dof-roll": "SO-101, tool roll about its approach axis",
                "dof-yaw-panda": "Franka Panda, hand yaw about the vertical axis with fixed position",
                "jet-roll": "schematic aircraft, roll", "jet-pitch": "schematic aircraft, pitch", "jet-yaw": "schematic aircraft, yaw",
                "gimbal-lock": "three nested rings, middle ring to 90 degrees",
            },
            "assets": [{"name": p.stem, "file": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in sorted(outputs)],
        }
        (args.demo_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("wrote manifest.json", flush=True)


if __name__ == "__main__":
    main()
