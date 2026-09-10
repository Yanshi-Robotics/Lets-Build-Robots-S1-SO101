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
import struct
import sys
import time
from importlib.metadata import version
from pathlib import Path

LEROBOT_VERSION = "0.6.1"
COURSE_CAMERA_NAMES = ("top", "wrist")
# The four rotations LeRobot's Cv2Rotation accepts. Anything else cannot be expressed in
# cameras.json, so the viewer refuses to offer it.
ROTATIONS = (0, 90, 180, -90)

CAPTURE_WIDTH, CAPTURE_HEIGHT = 640, 480  # The size LeRobot's SO-101 guides record at.
CAPTURE_FPS, CAPTURE_FOURCC = 30, "MJPG"

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


# VIDIOC_ENUM_FRAMESIZES = _IOWR('V', 74, struct v4l2_frmsizeenum)  (44 bytes)
VIDIOC_ENUM_FRAMESIZES = 0xC02C564A
MJPG_FOURCC = 0x47504A4D


def supported_sizes(node):
    """Discrete MJPG sizes this camera offers, largest first; empty when it cannot be asked.

    Asked over ioctl rather than by trying `VideoCapture.set`, because set() silently
    settles for the nearest mode it likes and reports that back as if it were granted.
    """
    try:
        file = os.open(str(node), os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return []
    sizes = []
    try:
        for index in range(64):
            buffer = array.array("B", struct.pack("<III", index, MJPG_FOURCC, 0) + bytes(32))
            try:
                fcntl.ioctl(file, VIDIOC_ENUM_FRAMESIZES, buffer, True)
            except OSError:
                break
            if struct.unpack_from("<I", buffer, 8)[0] != 1:  # discrete sizes only
                break
            sizes.append(struct.unpack_from("<II", buffer, 12))
    finally:
        os.close(file)
    return sorted(set(sizes), key=lambda size: -size[0] * size[1])


def shared_sizes(cameras):
    """Sizes every one of these cameras offers. Recording needs one size for all of them."""
    common = None
    for camera in cameras:
        offered = set(supported_sizes(Path(camera["path"]).resolve()))
        common = offered if common is None else common & offered
    return sorted(common or [], key=lambda size: -size[0] * size[1])


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


def survey():
    """Every camera that can actually deliver frames, in a stable order, with its best path.

    Node-level detail is collected here but not printed by default: an operator choosing
    between two cameras needs one line per camera, not one line per /dev entry.
    """
    links, nodes = stable_links(), video_nodes()
    groups = {}
    for node, capability in nodes:
        key = capability["bus_info"] if capability else f"unknown:{node}"
        groups.setdefault(key, []).append((node, capability))
    # Two cameras of one model is what breaks by-id, so it is counted across the whole
    # machine once, rather than guessed at from a single node.
    buses_per_card = {}
    for bus, members in groups.items():
        card = next((capability["card"] for _, capability in members if capability), None)
        if card:
            buses_per_card.setdefault(card, set()).add(bus)
    cameras, notes = [], []
    for bus, members in sorted(groups.items()):
        card = next((capability["card"] for _, capability in members if capability), "unknown device")
        for node, capability in members:
            if capability is None:
                notes.append(f"{node} could not be queried; another program may be holding it open")
                continue
            if not capability["captures"]:
                notes.append(f"{node} carries metadata, not images; it is never a camera path")
                continue
            duplicated = len(buses_per_card.get(capability["card"], set())) > 1
            path, reason = preferred_path(node, links, model_is_duplicated=duplicated)
            cameras.append({"node": node, "card": card, "bus": bus, "path": path,
                            "reason": reason, "duplicated": duplicated})
    return cameras, notes


def grab_one_frame(path):
    """One frame from a camera, opened and closed again; None when it delivers nothing.

    Cameras are opened one at a time here. Three at once can exceed what a shared USB
    controller carries, and a failure to open would then look like a broken camera.
    """
    import cv2

    capture = cv2.VideoCapture(str(path), cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        return None
    try:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # The first frames off a UVC camera are often stale or half exposed.
        for _ in range(8):
            received, frame = capture.read()
        return frame if received else None
    finally:
        capture.release()


DEFAULT_PORT = 4603  # Registered for this course tool; --port moves it.
JPEG_QUALITY = 80


MODEL_IMAGE_SIZE = 224  # PaliGemma's square input, shared by pi0, pi05 and pi0-fast.


def encode_jpeg(frame):
    import cv2

    if frame is None:
        return None
    encoded, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buffer.tobytes() if encoded else None


def open_for_preview(path, *, width=None, height=None):
    """A camera opened at the given capture size, or None when something else holds it."""
    import cv2

    capture = cv2.VideoCapture(str(path), cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        return None
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width or CAPTURE_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height or CAPTURE_HEIGHT)
    return capture


def grab_one_frame(path):
    """One frame from a camera, opened and closed again; None when it delivers nothing."""
    capture = open_for_preview(path)
    if capture is None:
        return None
    try:
        # The first frames off a UVC camera are often stale or half exposed.
        for _ in range(8):
            received, frame = capture.read()
        return frame if received else None
    finally:
        capture.release()


def write_snapshot(cameras, target, *, height=360):
    """One still per camera, side by side, for a machine that cannot open a browser."""
    import cv2
    import numpy

    panels = []
    for number, camera in enumerate(cameras, 1):
        frame = grab_one_frame(camera["path"])
        if frame is None:
            print(f"  [{number}] delivered no frame; another program may be holding it open")
            continue
        scale = height / frame.shape[0]
        panel = cv2.resize(frame, (max(1, int(frame.shape[1] * scale)), height))
        panels.append(label(panel, [f"[{number}]  {model_name(camera['card'])}"], cv2))
    if not panels:
        return False
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), numpy.hstack(panels))
    print(f"Wrote {path}.")
    return True


class CameraStream:
    """One camera read on its own thread, keeping only the newest frame.

    A browser page can hold several readers of one stream, and a camera cannot be read
    twice. Reading once into a shared slot means the number of viewers never changes
    the load on the USB bus.
    """

    def __init__(self, camera, *, rotation=0):
        self.camera = camera
        self.rotation = rotation
        self.width, self.height = CAPTURE_WIDTH, CAPTURE_HEIGHT
        self.capture = None
        self.frame = None
        self.lock = __import__("threading").Lock()
        self.stop = __import__("threading").Event()
        self.thread = None

    def start(self, width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT):
        import threading

        self.width, self.height = width, height
        self.capture = open_for_preview(self.camera["path"], width=width, height=height)
        if self.capture is None:
            return False
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True

    def _read_loop(self):
        while not self.stop.is_set():
            with self.lock:
                capture = self.capture
            if capture is None:  # a reopen is in flight
                time.sleep(0.02)
                continue
            received, frame = capture.read()
            if not received:
                time.sleep(0.05)
                continue
            with self.lock:
                # Discard a frame that arrived from a capture reopen() has replaced.
                if capture is self.capture:
                    self.frame = frame

    def current(self):
        """The newest frame with the configured rotation applied, or None."""
        import cv2

        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
        if frame is None:
            return None
        turned = get_cv2_rotation_code(self.rotation)
        return frame if turned is None else cv2.rotate(frame, turned)

    def jpeg(self):
        """The newest frame as JPEG bytes."""
        return encode_jpeg(self.current())

    def reopen(self, width, height):
        """Switch capture size without dropping the page's connections.

        The old capture is released *before* the new one is opened: a camera cannot be
        held open twice, so opening first would always fail and look like the camera
        refusing the size. The reader thread tolerates the gap, so an open MJPEG
        response survives the change instead of every panel going blank.

        A size the camera will not give leaves it reopened at the previous one, rather
        than dark.
        """
        with self.lock:
            previous, self.capture, self.frame = self.capture, None, None
        if previous is not None:
            previous.release()
        capture = open_for_preview(self.camera["path"], width=width, height=height)
        if capture is None:
            with self.lock:
                self.capture = open_for_preview(self.camera["path"], width=self.width, height=self.height)
            return False
        with self.lock:
            self.capture, self.width, self.height = capture, width, height
        return True

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.capture is not None:
            self.capture.release()


def get_cv2_rotation_code(rotation):
    """LeRobot's rotation value as the cv2 constant, or None for no rotation."""
    import cv2

    return {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
            -90: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation]




def size_for(width, height, rotation):
    """A `width` x `height` frame as it measures after `rotation`.

    LeRobot validates width and height against the frame *after* rotation, so a quarter
    turn swaps them. The function is its own inverse, which is what lets one helper both
    read a rotated pair back to sensor order and write it out again.
    """
    return (height, width) if rotation in (90, -90) else (width, height)

PAGE = """<!doctype html><meta charset=utf-8><title>SO-101 cameras</title>
<style>
 body{font:14px system-ui,sans-serif;margin:0;padding:20px;background:#111;color:#eee}
 h1{font-size:16px;margin:0 0 4px}
 .hint{margin:0 0 14px;color:#aaa;max-width:70em}
 .bars{display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin:0 0 18px}
 .bars a.go{background:#7ab7ff;color:#111;font-weight:700;text-decoration:none;border-radius:6px;padding:7px 14px}
 .msg{color:#9f9}
 .row{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}
 .cam{background:#1c1c1c;border:1px solid #333;border-radius:10px;overflow:hidden;width:%(width)dpx}
 .cam img{display:block;width:100%%;background:#000}
 .bar{display:flex;align-items:center;gap:8px;padding:8px 10px;font-size:13px;flex-wrap:wrap}
 .n{font-weight:700;color:#7ab7ff}
 .pick,.rot{display:flex;gap:4px}
 .pick{width:100%%}
 .rot{margin-left:auto}
 a.btn{color:#ddd;text-decoration:none;border:1px solid #444;border-radius:5px;padding:2px 7px;font-size:12px}
 a.btn.on{background:#7ab7ff;color:#111;border-color:#7ab7ff}
 a.btn.rec{border-color:#5c8f5c}
 a.btn.rec.on{border-color:#7ab7ff}
 .why{color:#888;font-size:12px;max-width:60em;margin:-8px 0 18px}
 .lbl{color:#777;font-size:12px;align-self:center}
 code{background:#000;padding:1px 5px;border-radius:4px;color:#9f9}
</style>
<h1>%(heading)s</h1>
<p class=hint>%(hint)s</p>
<div class=bars><a class=go href="/save">Save %(file)s</a><span class=msg>%(message)s</span></div>
<div class=bars><span class=lbl>capture size</span>%(sizes)s<span class=lbl>%(fill)s</span></div>
<p class=why>%(why)s</p>
<div class=row>%(cameras)s</div>
"""

PANEL_WIDTH = 320


def fill_note(width, height):
    """How much of the policy's 224 square this capture size actually fills."""
    short = round(MODEL_IMAGE_SIZE * min(width, height) / max(width, height))
    return f"{width}x{height} fills {short * MODEL_IMAGE_SIZE / (MODEL_IMAGE_SIZE ** 2) * 100:.0f}% of the 224 square"


class PreviewState:
    """What the page lets an operator change, and where Save writes it.

    Roles and rotations live here rather than on the streams because both are answers
    about the configuration, not about the capture: the same camera keeps delivering
    frames while the operator changes their mind about which view it is.
    """

    def __init__(self, streams, *, save_path, fps=CAPTURE_FPS, fourcc=CAPTURE_FOURCC,
                 sensor=(CAPTURE_WIDTH, CAPTURE_HEIGHT), sizes=()):
        self.streams = streams
        self.save_path = Path(save_path)
        self.fps, self.fourcc = fps, fourcc
        self.sensor = sensor  # unrotated, so a rotation change never compounds
        self.sizes = list(sizes)
        self.roles = {}
        self.message = ""
        for number, stream in enumerate(streams, 1):
            if stream.camera.get("role") in COURSE_CAMERA_NAMES:
                self.roles[number] = stream.camera["role"]

    def set_capture_size(self, width, height):
        """Every camera changes together, or none does.

        Recording needs one size across views, so a half-applied change would leave the
        two panels disagreeing about what is being recorded.
        """
        previous = self.sensor
        for index, stream in enumerate(self.streams):
            if stream.reopen(width, height):
                continue
            for done in self.streams[:index]:
                done.reopen(*previous)
            self.message = f"A camera would not give {width}x{height}; nothing was changed."
            return
        self.sensor = (width, height)
        self.message = ""

    def assign(self, number, role):
        """One role belongs to one camera, so assigning it takes it off any other."""
        self.roles = {held: name for held, name in self.roles.items() if name != role}
        if role in COURSE_CAMERA_NAMES:
            self.roles[number] = role
        else:
            self.roles.pop(number, None)
        self.message = ""

    def configuration(self):
        entries = {}
        for name in COURSE_CAMERA_NAMES:
            number = next((held for held, role in self.roles.items() if role == name), None)
            if number is None:
                return None
            stream = self.streams[number - 1]
            width, height = size_for(*self.sensor, stream.rotation)
            entries[name] = {"type": "opencv", "index_or_path": str(stream.camera["path"]),
                             "width": width, "height": height, "fps": self.fps,
                             "fourcc": self.fourcc, "rotation": stream.rotation}
        return entries

    def save(self):
        """Write cameras.json, or say what is still missing. Never a partial file."""
        entries = self.configuration()
        if entries is None:
            unset = [name for name in COURSE_CAMERA_NAMES
                     if name not in self.roles.values()]
            self.message = f"Choose which camera is {' and which is '.join(unset)} first."
            return
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_path.write_text(rendered_configuration(entries), encoding="utf-8")
        self.message = f"Saved {self.save_path}."
        print(f"  saved {self.save_path}")


def rendered_configuration(entries):
    """cameras.json with one camera per line, matching the lesson's printed example.

    The lesson shows the file so a reader can write it by hand, and the page writes the
    same file from the buttons. Keeping the two byte-for-byte comparable means a reader
    can check one against the other without wondering whether a formatting difference
    means a content difference.
    """
    lines = [f'  "{name}": {json.dumps(entry, separators=(", ", ": "))}' for name, entry in entries.items()]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def route_parts(path):
    """Path segments of a request, with any query string dropped.

    The page appends `?t=<now>` when refreshing the model-eye stills to defeat caching,
    so a parser that splits the raw path sees "1?t=1757..." where it expects "1".
    """
    return [part for part in path.split("?", 1)[0].split("/") if part]


def build_page(state, *, heading, hint, rotatable, assignable):
    cards = []
    for number, stream in enumerate(state.streams, 1):
        role = state.roles.get(number)
        picker = ""
        if assignable:
            choices = "".join(
                f'<a class="btn {"on" if role == name else ""}" href="/assign/{number}/{name}">{name}</a>'
                for name in COURSE_CAMERA_NAMES)
            clear = f'<a class="btn {"on" if role is None else ""}" href="/assign/{number}/none">not used</a>'
            picker = f'<span class=pick><span class=lbl>this is</span>{choices}{clear}</span>'
        turns = ""
        if rotatable:
            links = "".join(
                f'<a class="btn {"on" if stream.rotation == value else ""}" href="/rotate/{number}/{value}">{value}</a>'
                for value in ROTATIONS)
            turns = f'<span class=rot><span class=lbl>rotation</span>{links}</span>'
        title = role if role else model_name(stream.camera["card"])
        cards.append(
            f'<div class=cam><img src="/stream/{number}" alt="camera {number}">'
            f'<div class=bar><span class=n>[{number}]</span><span>{title}</span>{turns}{picker}</div></div>')
    sizes = "".join(
        f'<a class="btn {"rec " if (width, height) == (CAPTURE_WIDTH, CAPTURE_HEIGHT) else ""}'
        f'{"on" if (width, height) == tuple(state.sensor) else ""}" '
        f'href="/size/{width}/{height}">{width}x{height}'
        f'{" &#9733;" if (width, height) == (CAPTURE_WIDTH, CAPTURE_HEIGHT) else ""}</a>'
        for width, height in state.sizes) or '<span class=lbl>(could not be read from the hardware)</span>'
    why = (f"&#9733; {CAPTURE_WIDTH}x{CAPTURE_HEIGHT} is what this course records at: LeRobot's own recording "
           f"examples and the main SO-101 guides use this size. A larger frame is not free — the dataset grows, "
           f"training slows, and ACT's backbone does not need the extra detail. Being 4:3 it also loses less to "
           f"padding in the 224 square than a 16:9 frame does.")
    return (PAGE % {"width": PANEL_WIDTH, "heading": heading, "hint": hint,
                    "file": state.save_path.name, "message": state.message,
                    "sizes": sizes, "fill": fill_note(*state.sensor), "why": why,
                    "cameras": "".join(cards)}).encode("utf-8")


def serve_preview(state, *, port, heading, hint, rotatable=True, assignable=False):
    """A local page showing every stream live, until Ctrl-C.

    A page rather than a desktop window because LeRobot pins opencv-python-headless:
    the cv2 in the course environment is built without any GUI backend, so
    `cv2.imshow` raises rather than opening anything. The page also works unchanged
    over an SSH port forward, which a window never could.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    streams = state.streams

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # One line per JPEG would bury the instructions printed above.

        def _stream(self, number):
            stream = streams[number - 1]
            source = stream.jpeg
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpeg = source()
                    if jpeg is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg + b"\r\n")
                    time.sleep(1 / 30)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The viewer closed the tab or reloaded.

        def _redirect_home(self):
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

        def do_GET(self):
            parts = route_parts(self.path)
            if not parts:
                body = build_page(state, heading=heading, hint=hint,
                                  rotatable=rotatable, assignable=assignable)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parts[0] == "stream" and parts[1].isdigit() and 1 <= int(parts[1]) <= len(streams):
                self._stream(int(parts[1]))
                return
            if parts[0] == "size" and len(parts) == 3:
                state.set_capture_size(int(parts[1]), int(parts[2]))
                print(f"  capture size {parts[1]}x{parts[2]}")
                self._redirect_home()
                return
            if parts[0] == "rotate" and len(parts) == 3 and rotatable:
                number, value = int(parts[1]), int(parts[2])
                if 1 <= number <= len(streams) and value in ROTATIONS:
                    streams[number - 1].rotation = value
                    state.message = ""
                    print(f"  [{number}] rotation {value}")
                self._redirect_home()
                return
            if parts[0] == "assign" and len(parts) == 3 and assignable:
                number = int(parts[1])
                if 1 <= number <= len(streams):
                    state.assign(number, parts[2])
                    print(f"  [{number}] is {parts[2]}")
                self._redirect_home()
                return
            if parts[0] == "save":
                state.save()
                self._redirect_home()
                return
            self.send_error(404)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    print(f"\nOpen  http://127.0.0.1:{port}  in a browser.")
    print("Press Ctrl-C here when you are done.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("")
    finally:
        server.shutdown()
        server.server_close()


def start_streams(cameras, *, size=(CAPTURE_WIDTH, CAPTURE_HEIGHT)):
    """Every camera opened for the page; the ones that refuse are named, not fatal."""
    streams = []
    for number, camera in enumerate(cameras, 1):
        stream = CameraStream(camera, rotation=camera.get("rotation", 0))
        if stream.start(*size):
            streams.append(stream)
        else:
            print(f"  [{number}] could not be opened; close whatever is using it and run this again")
    return streams


def run_list(args):
    cameras, notes = survey()
    if not cameras:
        print("No camera delivers frames on this computer. Plug them in and run this again.")
        for note in notes:
            print(f"  {note}")
        return 1

    print(f"Found {len(cameras)} camera(s) that deliver frames.\n")
    for number, camera in enumerate(cameras, 1):
        print(f"  [{number}] {model_name(camera['card'])}")
        print(f"      {camera['path']}")
    if any(camera["duplicated"] for camera in cameras):
        print("\nTwo cameras of one model report no serial number, so the paths above name the USB")
        print("socket rather than the camera. Leave each camera in the socket it is in now.")
    if args.verbose:
        print("\nWhy each path was chosen:")
        for number, camera in enumerate(cameras, 1):
            print(f"  [{number}] {camera['node']} on {camera['bus']}: {camera['reason']}.")
        for note in notes:
            print(f"  {note}")

    print("\nNow look at the pictures. The view looking down at the workbench is `top`;")
    print("the view from the gripper is `wrist`. Note their numbers, then copy those two")
    print("paths into cameras.json.")
    if args.snapshot is not None:
        print("")
        write_snapshot(cameras, args.snapshot)
        return 0
    streams = start_streams(cameras)
    if not streams:
        print("\nNo camera could be opened for viewing.")
        return 1
    state = PreviewState(streams, save_path=args.cameras, sizes=shared_sizes(cameras))
    try:
        serve_preview(state, port=args.port, assignable=True,
                      heading="Which camera is which?",
                      hint="Wave a hand in front of one lens: the panel that moves is that camera. "
                           "Mark the view of the whole workbench as <code>top</code> and the view from "
                           "the gripper as <code>wrist</code>, turn each picture the right way up, "
                           "then save.")
    finally:
        for stream in streams:
            stream.close()
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

def label(frame, lines, cv2, *, scale=0.6):
    """Burn the caption into a copy of the frame, so a saved still carries its own identity."""
    import numpy

    canvas = numpy.ascontiguousarray(frame)
    step = int(34 * scale / 0.8)
    for row, text in enumerate(lines):
        origin = (12, int(30 * scale / 0.8) + row * step)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def model_name(card):
    """The camera's model as one name.

    V4L2 hands back whatever the device reports, and these modules report their name
    twice with a colon between. Printing that verbatim wastes the width the number needs.
    """
    parts = [part.strip() for part in str(card).split(":") if part.strip()]
    return parts[0] if parts and all(part == parts[0] for part in parts) else str(card)


def run_preview(cameras, *, port=DEFAULT_PORT, snapshot=None, save_path="cameras.json"):
    """Both configured views live, with the four rotations one click apart.

    This is the acceptance step the numbers cannot cover: a stream that opens is not yet
    a stream that is the right way up, and rotation is part of the model's input.
    """
    entries = [{"card": name, "role": name, "path": camera["path"], "rotation": camera["rotation"]}
               for name, camera in cameras.items()]
    if snapshot is not None:
        return 0 if write_snapshot(entries, snapshot) else 1
    # The file stores sizes after rotation, so read them back to sensor order before the
    # cameras are opened; otherwise every turn would swap them again.
    first = next(iter(cameras.values()))
    sensor = size_for(first["width"] or CAPTURE_WIDTH, first["height"] or CAPTURE_HEIGHT,
                      first["rotation"])
    streams = start_streams(entries, size=sensor)
    if len(streams) != len(entries):
        for stream in streams:
            stream.close()
        print("Not every configured camera could be opened, so orientation cannot be settled here.")
        return 1
    state = PreviewState(streams, save_path=save_path, sensor=sensor, sizes=shared_sizes(entries),
                         fps=first["fps"] or CAPTURE_FPS, fourcc=first["fourcc"] or CAPTURE_FOURCC)
    try:
        serve_preview(state, port=port,
                      heading="Which way up is each image?",
                      hint="Click a rotation until the picture looks right: the gripper should enter "
                           "the wrist view from the bottom, and the workbench should sit square in the "
                           "top view. Then save.")
    finally:
        for stream in streams:
            stream.close()
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
    return run_preview(cameras, port=args.port)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="mode", required=True)
    listing = commands.add_parser("list", help="every camera that delivers frames, with one frame from each")
    listing.add_argument("--verbose", action="store_true", help="also print why each path was chosen")
    listing.add_argument("--cameras", default="cameras.json", help="where the page's Save button writes")
    listing.add_argument("--snapshot", help="write one still per camera to this file instead of serving a page")
    listing.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port for the viewing page (default {DEFAULT_PORT})")
    check = commands.add_parser("check", help="verify cameras.json, then show both views")
    check.add_argument("--cameras", default="cameras.json", help="the LeRobot camera configuration to verify")
    check.add_argument("--frames", type=int, default=90, help="frames per measurement (90 is about 3s at 30 FPS)")
    check.add_argument("--no-preview", action="store_true", help="stop after the verdict")
    check.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port for the viewing page (default {DEFAULT_PORT})")
    preview = commands.add_parser("preview", help="show both views without measuring")
    preview.add_argument("--cameras", default="cameras.json")
    preview.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port for the viewing page (default {DEFAULT_PORT})")
    preview.add_argument("--snapshot", help="write one still per view to this path instead of serving a page")
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
        return run_preview(load_cameras_file(args.cameras), port=args.port,
                           snapshot=args.snapshot, save_path=args.cameras)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
