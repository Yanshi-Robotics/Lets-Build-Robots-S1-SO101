#!/usr/bin/env python3
"""Identify the two SO-101 course cameras, measure what they actually deliver, and show them.

Nothing here touches a motor. The arm can stay unpowered for every mode, and none of the
three writes to the robot; the only hardware this program opens is a camera.

  list     every V4L2 node on this computer, grouped by the physical camera behind it,
           with the stable path to use for each. A UVC camera registers more than one
           /dev/videoN and only one of them delivers frames, so the numbers alone never
           identify a camera.
  check    open the two cameras named in cameras.json exactly the way LeRobot opens them
           during recording, measure the delivered frame rate, and refuse anything that
           is frozen or that turns out to be one camera opened twice. Ends in one of
           three verdicts, then shows both views unless --no-preview is given.
  preview  the viewer on its own, for when the verdict already passed and the question
           is only which way round a camera is mounted.

The preview exists because a camera that opens is not yet a camera that is aimed and the
right way up. Rotation is part of the model's input: a policy trained on a 180-degree
image fails on the same camera mounted the other way. Press `r` in the viewer to try the
four rotations LeRobot supports and `s` to print the cameras.json line for what you chose.
"""
from __future__ import annotations

import argparse
import array
import fcntl
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

LEROBOT_VERSION = "0.6.1"
COURSE_CAMERA_NAMES = ("top", "wrist")
# The four rotations LeRobot's Cv2Rotation accepts. Anything else cannot be expressed in
# cameras.json, so the viewer refuses to offer it.
ROTATIONS = (0, 90, 180, -90)

# --- V4L2 device query -------------------------------------------------------------
# VIDIOC_QUERYCAP is the only ioctl used here. It answers the two questions the numbers
# cannot: whether a node can capture at all, and which physical USB port it sits on.
# struct v4l2_capability: driver[16] card[32] bus_info[32] version, capabilities,
# device_caps, reserved[3] -> 104 bytes. _IOR('V', 0, that struct).
VIDIOC_QUERYCAP = 0x80685600
V4L2_CAP_VIDEO_CAPTURE = 0x00000001


def query_capability(node):
    """driver, card, bus_info and whether this node can capture video; None if it cannot be asked.

    A node that exists but refuses the ioctl is reported as unknown rather than skipped:
    silently dropping a device would hide exactly the case an operator is looking for.
    """
    buffer = array.array("B", bytes(104))
    try:
        file = os.open(str(node), os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        fcntl.ioctl(file, VIDIOC_QUERYCAP, buffer, True)
    except OSError:
        return None
    finally:
        os.close(file)
    raw = bytes(buffer)
    text = lambda start, length: raw[start:start + length].split(b"\0")[0].decode("utf-8", "replace")
    capabilities = int.from_bytes(raw[84:88], sys.byteorder)
    device_caps = int.from_bytes(raw[88:92], sys.byteorder)
    # device_caps describes this node; capabilities describes the whole device. Prefer the
    # per-node answer, which is what distinguishes a capture node from its metadata sibling.
    effective = device_caps or capabilities
    return {
        "driver": text(0, 16),
        "card": text(16, 32),
        "bus_info": text(48, 32),
        "captures": bool(effective & V4L2_CAP_VIDEO_CAPTURE),
    }


def stable_links():
    """Every /dev/v4l/by-id and by-path link, grouped by the node it points at.

    by-id is preferred when it is unique, but two cameras of the same model with no
    serial number produce the same by-id name and udev keeps only one link. That case is
    common with the small UVC modules this course uses, so by-path is reported too: it
    names the USB socket rather than the device, which is stable as long as the camera
    stays in the same port.
    """
    links = {}
    for directory, kind in ((Path("/dev/v4l/by-id"), "by-id"), (Path("/dev/v4l/by-path"), "by-path")):
        if not directory.is_dir():
            continue
        for link in sorted(directory.iterdir()):
            try:
                target = link.resolve()
            except OSError:
                continue
            links.setdefault(str(target), []).append((kind, link))
    return links


def video_nodes():
    """Every /dev/videoN with what QUERYCAP says about it, in numeric order."""
    directory = Path("/dev")
    nodes = sorted((path for path in directory.glob("video*") if path.name[5:].isdigit()),
                   key=lambda path: int(path.name[5:]))
    return [(node, query_capability(node)) for node in nodes]


def preferred_path(node, links, *, model_is_duplicated):
    """The stable path to write into cameras.json for this node, and why it was chosen.

    by-id is built from the model name and the serial number the camera reports. The small
    UVC modules this course uses report no serial, so two of the same model produce the same
    by-id name and udev keeps a single link pointing at whichever enumerated last. The other
    camera then has no by-id link at all, and the one link that does exist silently moves
    between the two across reboots. `model_is_duplicated` is therefore the deciding fact,
    not whether this particular node happens to have a link today.
    """
    entries = links.get(str(node), [])
    by_id = [link for kind, link in entries if kind == "by-id"]
    by_path = [link for kind, link in entries if kind == "by-path"]
    if by_id and not model_is_duplicated:
        return by_id[0], "by-id carries this camera's own name and no other camera answers to it"
    if by_path:
        return by_path[0], ("two cameras of this model are connected and report no serial number, so by-id "
                            "cannot tell them apart; by-path names the USB socket instead, which holds as "
                            "long as this camera stays in that socket"
                            if model_is_duplicated else "this camera has no by-id link")
    return node, "no stable link exists; this number changes when the camera is replugged"


def run_list(args):
    links = stable_links()
    nodes = video_nodes()
    if not nodes:
        print("No /dev/video* node exists. Plug the cameras in and run this again.")
        return 1
    groups = {}
    for node, capability in nodes:
        key = capability["bus_info"] if capability else f"unknown:{node}"
        groups.setdefault(key, []).append((node, capability))
    # How many separate cameras report each model name. Two of the same model is what breaks
    # by-id, so it is counted once here rather than guessed at per node.
    buses_per_card = {}
    for bus, members in groups.items():
        card = next((capability["card"] for _, capability in members if capability), None)
        if card:
            buses_per_card.setdefault(card, set()).add(bus)
    print(f"{len(nodes)} node(s) on {len(groups)} physical camera(s).")
    for bus, members in groups.items():
        card = next((capability["card"] for _, capability in members if capability), "unknown device")
        print("")
        print(f"=== {card}  on {bus} ===")
        for node, capability in members:
            if capability is None:
                print(f"  {node}  cannot be queried; another program may hold it open")
                continue
            if not capability["captures"]:
                print(f"  {node}  no capture capability (metadata node); never put this in cameras.json")
                continue
            duplicated = len(buses_per_card.get(capability["card"], set())) > 1
            path, reason = preferred_path(node, links, model_is_duplicated=duplicated)
            print(f"  {node}  captures video")
            print(f"      use: {path}")
            print(f"      because {reason}.")
    print("")
    print("Which of these is `top` and which is `wrist` cannot be read from any of the text above.")
    print("Run `preview` on a path and look at the picture; that is the only thing that settles it.")
    return 0


# --- opening cameras the way LeRobot does -------------------------------------------

def open_camera(path, *, width, height, fps, fourcc, rotation):
    """A connected LeRobot OpenCVCamera, the same class `lerobot-record` uses.

    Measuring through LeRobot rather than through cv2 directly is the point: the number
    that matters is what the recording pipeline gets, including its colour conversion and
    its rotation, not what the driver could deliver to a different reader.
    """
    from lerobot.cameras.configs import Cv2Rotation
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    config = OpenCVCameraConfig(
        index_or_path=int(path) if str(path).isdigit() else Path(path),
        width=width, height=height, fps=fps, fourcc=fourcc, rotation=Cv2Rotation(rotation),
    )
    camera = OpenCVCamera(config)
    camera.connect()
    return camera


def measure(camera, *, frames, name):
    """Delivered frame rate over `frames` reads, plus how many arrived byte-identical.

    Two identical frames in a row are normal on a still workbench, so the count is
    reported rather than judged. Every frame identical is different: a real sensor
    carries noise, so an unchanging buffer means the stream is frozen.
    """
    import numpy

    previous, identical = None, 0
    start = time.perf_counter()
    for _ in range(frames):
        frame = camera.read()
        if previous is not None and numpy.array_equal(frame, previous):
            identical += 1
        previous = frame
    elapsed = time.perf_counter() - start
    rate = frames / elapsed if elapsed > 0 else 0.0
    print(f"  {name}: {frames} frames in {elapsed:.1f}s -> {rate:.1f} FPS delivered, "
          f"{frame.shape[1]}x{frame.shape[0]}, {identical}/{frames - 1} repeated")
    return {"fps": rate, "identical": identical, "frames": frames,
            "width": frame.shape[1], "height": frame.shape[0]}


def load_cameras_file(path):
    """The two course cameras from cameras.json, with the fields this program needs.

    Anything beyond `top` and `wrist` is left alone: the file is LeRobot's, and this
    program only checks the two views the course records.
    """
    file = Path(path)
    if not file.is_file():
        raise RuntimeError(f"No camera configuration at {file}. Run `list` first and write cameras.json.")
    entries = json.loads(file.read_text(encoding="utf-8"))
    missing = [name for name in COURSE_CAMERA_NAMES if name not in entries]
    if missing:
        raise RuntimeError(f"cameras.json has no entry named {' or '.join(missing)}. "
                           "The names are feature names in the dataset and cannot be renamed.")
    cameras = {}
    for name in COURSE_CAMERA_NAMES:
        entry = entries[name]
        if entry.get("type") != "opencv":
            raise RuntimeError(f"{name} is type {entry.get('type')!r}; this course records with type opencv.")
        rotation = entry.get("rotation", 0)
        if rotation not in ROTATIONS:
            raise RuntimeError(f"{name} has rotation {rotation!r}; LeRobot accepts {list(ROTATIONS)}.")
        cameras[name] = {
            "path": str(entry["index_or_path"]),
            "width": entry.get("width"), "height": entry.get("height"),
            "fps": entry.get("fps"), "fourcc": entry.get("fourcc"),
            "rotation": rotation,
        }
    return cameras


RESULT_RULE = "-" * 78


def verdict(cameras, results, buses):
    """One of READY, SAME CAMERA, NOT READY or CANNOT TELL, with the reason.

    Conservative on purpose: a measurement that is merely below the requested rate is
    reported as NOT READY with the number, never rounded up into a pass.
    """
    # Asked first, because it is answered without opening anything, and because one camera
    # named twice also fails to open the second time: that failure would otherwise be
    # reported as "could not be opened" and send the operator looking at the wrong thing.
    top_bus, wrist_bus = (buses.get(name) for name in COURSE_CAMERA_NAMES)
    if top_bus and wrist_bus and top_bus == wrist_bus:
        return "SAME CAMERA", f"both entries reach the same physical camera on {top_bus}"
    unread = [name for name in COURSE_CAMERA_NAMES if name not in results]
    if unread:
        return "CANNOT TELL", f"the {' and the '.join(unread)} camera could not be opened"
    frozen = [name for name, result in results.items() if result["identical"] == result["frames"] - 1]
    if frozen:
        return "NOT READY", (f"every frame from the {' and '.join(frozen)} camera was identical, "
                             "so that stream is not updating")
    slow = [f"{name} delivered {results[name]['fps']:.1f} of the {cameras[name]['fps']} FPS it was asked for"
            for name in COURSE_CAMERA_NAMES
            if cameras[name]["fps"] and results[name]["fps"] < cameras[name]["fps"] * 0.9]
    if slow:
        return "NOT READY", "; ".join(slow)
    return "READY", "both cameras are separate devices and both delivered the rate they were configured for"


def say_verdict(name, detail, cameras):
    print("")
    print(RESULT_RULE)
    print(f"RESULT: {name}")
    for camera in COURSE_CAMERA_NAMES:
        print(f"    {camera:<6} {cameras[camera]['path']}")
    print(f"    Because {detail}.")
    if name == "READY":
        print("    These two streams can be recorded. Which view is which, and which way up each")
        print("    one is, is still a question about the picture; the preview below answers it.")
    elif name == "SAME CAMERA":
        print("    One camera cannot supply two views. Run `list`, take the path of the other")
        print("    camera, and correct cameras.json.")
    elif name == "NOT READY":
        print("    Fix the reported problem before recording. A dataset recorded through a frozen")
        print("    or slow stream cannot be repaired afterwards.")
    else:
        print("    Read the lines above before recording anything.")
    print(RESULT_RULE)


# --- the viewer ---------------------------------------------------------------------

def label(frame, lines, cv2):
    """Burn the caption into a copy of the frame, so what is shown carries its own identity."""
    import numpy

    canvas = numpy.ascontiguousarray(frame)
    for row, text in enumerate(lines):
        origin = (12, 30 + row * 30)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    # An arrow at the top edge: the only reliable way to say "this side is up in the data"
    # is to point at it in the picture the model will receive.
    top = (canvas.shape[1] // 2, 18)
    cv2.arrowedLine(canvas, (top[0], top[1] + 46), top, (0, 0, 0), 8, tipLength=0.4)
    cv2.arrowedLine(canvas, (top[0], top[1] + 46), top, (255, 255, 255), 3, tipLength=0.4)
    return canvas


def compose(opened, cameras, rotations, chosen, *, height, cv2):
    """One image holding both views, captioned with the identity each one carries into the data."""
    import numpy

    panels = []
    for name in opened:
        frame = opened[name].read()
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        scale = height / image.shape[0]
        image = cv2.resize(image, (max(1, int(image.shape[1] * scale)), height))
        marker = ">" if name == chosen else " "
        panels.append(label(image, [f"{marker}{name}", f"rotation {rotations[name]}",
                                    f"{frame.shape[1]}x{frame.shape[0]}"], cv2))
    return numpy.hstack(panels)


def run_preview(cameras, *, height=480, snapshot=None):
    """Both views side by side, with `r` to try the rotations and `s` to print the result.

    The viewer is the acceptance step the numbers cannot cover. Wave a hand in front of one
    camera: the view that moves is that camera, and the direction it moves in tells you
    whether the image is mirrored end for end.

    With `snapshot` it writes one captioned still instead of opening a window. That still is
    what the lesson's camera record asks to keep: a picture of what each view actually saw,
    dated, next to the mounting photograph.
    """
    import cv2

    if snapshot is None and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("No display is available, so the preview is skipped. Run with --no-preview to silence this.")
        return 0

    opened = {}
    try:
        for name, camera in cameras.items():
            opened[name] = open_camera(camera["path"], width=camera["width"], height=camera["height"],
                                       fps=camera["fps"], fourcc=camera["fourcc"], rotation=camera["rotation"])
        rotations = {name: camera["rotation"] for name, camera in cameras.items()}
        chosen = list(cameras)[0]
        if snapshot is not None:
            image = compose(opened, cameras, rotations, chosen, height=height, cv2=cv2)
            target = Path(snapshot)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), image):
                raise RuntimeError(f"Could not write {target}")
            print(f"Wrote {target} ({image.shape[1]}x{image.shape[0]}), both views as configured.")
            return 0
        window = "SO-101 cameras: r rotate selected, tab switch, s print, q quit"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print("")
        print("Viewer: `r` rotates the selected view, `tab` switches which view is selected,")
        print("        `s` prints the cameras.json lines for what is on screen, `q` quits.")
        print("Hold a hand in front of one camera and check that the view that moves is the one you expect.")
        while True:
            cv2.imshow(window, compose(opened, cameras, rotations, chosen, height=height, cv2=cv2))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("\t"):
                names = list(opened)
                chosen = names[(names.index(chosen) + 1) % len(names)]
            if key == ord("r"):
                current = rotations[chosen]
                new = ROTATIONS[(ROTATIONS.index(current) + 1) % len(ROTATIONS)]
                # The camera is reopened rather than rotated in place: width and height swap
                # at 90 and 270, and LeRobot validates them at connect time.
                camera = cameras[chosen]
                width, height_config = camera["width"], camera["height"]
                if (current in (90, -90)) != (new in (90, -90)):
                    width, height_config = height_config, width
                opened[chosen].disconnect()
                cameras[chosen] = {**camera, "width": width, "height": height_config, "rotation": new}
                opened[chosen] = open_camera(camera["path"], width=width, height=height_config,
                                             fps=camera["fps"], fourcc=camera["fourcc"], rotation=new)
                rotations[chosen] = new
                print(f"{chosen}: rotation {new}, now {width}x{height_config}")
            if key == ord("s"):
                print("")
                print("cameras.json for what is on screen:")
                print(json.dumps({name: {"type": "opencv", "index_or_path": camera["path"],
                                         "width": camera["width"], "height": camera["height"],
                                         "fps": camera["fps"], "fourcc": camera["fourcc"],
                                         "rotation": camera["rotation"]}
                                  for name, camera in cameras.items()}, indent=2))
    finally:
        for camera in opened.values():
            if camera.is_connected:
                camera.disconnect()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    return 0


def bus_of(path):
    """The USB socket behind a camera path, used to catch one camera entered twice."""
    node = Path(path)
    if not node.exists():
        return None
    capability = query_capability(node.resolve())
    return capability["bus_info"] if capability else None


def run_check(args):
    cameras = load_cameras_file(args.cameras)
    results, buses = {}, {}
    print(f"Opening the two cameras named in {args.cameras} the way lerobot-record opens them.")
    for name, camera in cameras.items():
        buses[name] = bus_of(camera["path"])
        opened = None
        try:
            opened = open_camera(camera["path"], width=camera["width"], height=camera["height"],
                                 fps=camera["fps"], fourcc=camera["fourcc"], rotation=camera["rotation"])
            results[name] = measure(opened, frames=args.frames, name=name)
        except Exception as exc:
            print(f"  {name}: could not be opened: {exc}")
        finally:
            if opened is not None and opened.is_connected:
                opened.disconnect()
    # Both cameras are measured one at a time above and then together below: a pair that
    # each work alone can still exceed what one USB controller delivers.
    if len(results) == len(COURSE_CAMERA_NAMES):
        print("Both at once, which is how they are recorded:")
        both = {}
        try:
            for name, camera in cameras.items():
                both[name] = open_camera(camera["path"], width=camera["width"], height=camera["height"],
                                         fps=camera["fps"], fourcc=camera["fourcc"], rotation=camera["rotation"])
            for name, camera in both.items():
                results[name] = measure(camera, frames=args.frames, name=name)
        except Exception as exc:
            print(f"  the pair could not be opened together: {exc}")
            results = {name: result for name, result in results.items() if name in both}
        finally:
            for camera in both.values():
                if camera.is_connected:
                    camera.disconnect()
    name, detail = verdict(cameras, results, buses)
    say_verdict(name, detail, cameras)
    if name != "READY":
        return 1
    if args.no_preview:
        return 0
    return run_preview(cameras)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("list", help="every video node, grouped by physical camera")
    check = commands.add_parser("check", help="verify cameras.json, then show both views")
    check.add_argument("--cameras", default="cameras.json", help="the LeRobot camera configuration to verify")
    check.add_argument("--frames", type=int, default=90, help="frames per measurement (90 is about 3s at 30 FPS)")
    check.add_argument("--no-preview", action="store_true", help="stop after the verdict")
    preview = commands.add_parser("preview", help="show both views without measuring")
    preview.add_argument("--cameras", default="cameras.json")
    preview.add_argument("--snapshot", help="write one captioned still to this path instead of opening a window")
    args = parser.parse_args(argv)
    try:
        if args.mode != "list" and version("lerobot") != LEROBOT_VERSION:
            raise RuntimeError(f"Use lerobot=={LEROBOT_VERSION} in the course environment.")
        if args.mode == "list":
            return run_list(args)
        if args.mode == "check":
            if args.frames < 2:
                parser.error("at least two frames are needed to tell a moving stream from a frozen one")
            return run_check(args)
        return run_preview(load_cameras_file(args.cameras), snapshot=args.snapshot)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
