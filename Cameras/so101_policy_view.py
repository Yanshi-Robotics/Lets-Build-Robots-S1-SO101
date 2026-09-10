#!/usr/bin/env python3
"""Show what each policy actually receives from the two cameras in cameras.json.

`cameras.json` says what the cameras deliver. It does not say what a policy sees:
every policy resizes on the way in, and they do not agree on how. ACT does not resize
at all. The pi0 family pads the frame into a 224 square with the image centred.
SmolVLA pads into 512 with the bars on the left and top, so the image sits bottom
right. The differences are easy to read about and easy to forget; this program puts
them side by side, live, while the cameras can still be moved.

Run it after `so101_camera_check.py check` has settled the configuration. It reads
cameras.json, opens those two cameras and touches nothing else: no motor register is
written and no action is sent, so the arm can stay unpowered throughout.

    python Cameras/so101_policy_view.py
    python Cameras/so101_policy_view.py --cameras cameras.json --port 4603

Every transform below is a reproduction of the code in LeRobot 0.6.1, not an
approximation: the same ratio, the same integer truncation, the same padding sides.
A policy whose resizing this program cannot reproduce exactly is left out rather than
drawn wrongly; the page names those at the bottom.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_camera_check as check  # noqa: E402  (same folder, as the lesson downloads it)

REFRESH_MS = 900  # These panels show framing, not motion; a slow refresh keeps requests cheap.

# How each policy reshapes an incoming frame, read out of LeRobot 0.6.1.
#   centred  -> policies/common/vla_utils.py resize_with_pad_torch (openpi convention)
#   top-left -> policies/common/vla_utils.py resize_with_pad       (smolvla / xvla)
POLICIES = (
    {
        "id": "act",
        "name": "ACT",
        "target": None,
        "pad": None,
        "note": "No resizing at all. A ResNet-18 backbone takes the frame exactly as recorded, "
                "so every captured pixel reaches the network — and every extra pixel costs "
                "memory and training time. All cameras must share one shape.",
    },
    {
        "id": "pi0",
        "name": "π₀",
        "target": (224, 224),
        "pad": "centred",
        "note": "224x224, aspect ratio preserved, black bars centred. That is the pretrained "
                "resolution of the PaliGemma vision encoder it is built on. Square is enforced "
                "in code; the size itself is a config value.",
    },
    {
        "id": "pi05",
        "name": "π₀.₅",
        "target": (224, 224),
        "pad": "centred",
        "note": "Identical image handling to π₀, and so is π₀-FAST. What differs between "
                "them is the action expert and the training recipe, not what the camera feeds them.",
    },
    {
        "id": "smolvla",
        "name": "SmolVLA",
        "target": (512, 512),
        "pad": "top-left",
        "note": "512x512, aspect ratio preserved, but the bars go on the LEFT and TOP: the image "
                "sits in the bottom-right corner instead of the middle. Same frame, different "
                "place in the square.",
    },
)

# Reproduced only where the resizing is in LeRobot itself. GR00T's config carries a
# 256x256 image_size, but the resizing is done by its backbone's own image processor,
# so drawing it here would be a guess.
NOT_DRAWN = "GR00T N1.5 targets 256x256, but its backbone's image processor does the resizing, so it is not drawn here."


def resized(frame, target, pad):
    """`frame` as a policy will receive it, reproducing LeRobot's own arithmetic.

    Both helpers scale by the larger of the two ratios, truncate to int, and pad what is
    left. They differ only in which sides the padding goes on, which is exactly the thing
    a still picture makes obvious and a paragraph does not.
    """
    import cv2
    import numpy

    if frame is None:
        return None
    if target is None:
        return frame
    height, width = target
    current_height, current_width = frame.shape[:2]
    ratio = max(current_width / width, current_height / height)
    new_height, new_width = int(current_height / ratio), int(current_width / ratio)
    scaled = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = numpy.zeros((height, width, frame.shape[2]), dtype=frame.dtype)
    if pad == "top-left":
        top, left = max(0, height - new_height), max(0, width - new_width)
    else:
        top = (height - new_height) // 2
        left = (width - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = scaled
    return canvas


PAGE = """<!doctype html><meta charset=utf-8><title>What each policy sees</title>
<style>
 body{font:14px system-ui,sans-serif;margin:0;padding:20px;background:#111;color:#eee}
 h1{font-size:17px;margin:0 0 6px}
 .hint{margin:0 0 18px;color:#aaa;max-width:78em;line-height:1.6}
 table{border-collapse:collapse;width:100%%;max-width:1200px}
 th,td{border:1px solid #333;padding:10px;vertical-align:top;text-align:left}
 th{background:#1c1c1c;font-size:13px;color:#bbb;font-weight:600}
 td.name{font-weight:700;color:#7ab7ff;white-space:nowrap;font-size:15px}
 td.shot{width:%(cell)dpx;text-align:center}
 td.shot img{max-width:%(shot)dpx;max-height:%(shot)dpx;background:#000;border:1px solid #333}
 td.shot .size{display:block;margin-top:6px;color:#777;font-size:12px}
 td.note{color:#aaa;font-size:13px;line-height:1.6}
 .foot{margin-top:16px;color:#777;font-size:12px;max-width:78em;line-height:1.6}
 code{background:#000;padding:1px 5px;border-radius:4px;color:#9f9}
</style>
<h1>What each policy actually receives</h1>
<p class=hint>The two cameras are delivering %(capture)s. Each row shows the same live frame
after that policy's own preprocessing. Black area is padding — pixels the policy never sees.
Nothing here changes your configuration; it reads <code>%(file)s</code> and shows the result.</p>
<table>
<tr><th>Policy</th><th>top</th><th>wrist</th><th>What its input is</th></tr>
%(rows)s
</table>
<p class=foot>%(foot)s</p>
<script>
 function refresh() {
   document.querySelectorAll('img.shot').forEach(function (img) {
     img.src = img.dataset.src + '?t=' + Date.now();
   });
 }
 refresh();
 setInterval(refresh, %(refresh)d);
</script>
"""


def build_page(capture, save_path, shot):
    rows = []
    for policy in POLICIES:
        cells = []
        for name in check.COURSE_CAMERA_NAMES:
            target = policy["target"]
            label = f"{target[1]}x{target[0]}" if target else f"{capture} (unchanged)"
            cells.append(f'<td class=shot><img class=shot data-src="/frame/{policy["id"]}/{name}" '
                         f'alt="{policy["name"]} {name}"><span class=size>{label}</span></td>')
        rows.append(f'<tr><td class=name>{policy["name"]}</td>{"".join(cells)}'
                    f'<td class=note>{policy["note"]}</td></tr>')
    return (PAGE % {"cell": shot + 30, "shot": shot, "capture": capture, "file": save_path,
                    "rows": "".join(rows), "foot": NOT_DRAWN, "refresh": REFRESH_MS}).encode("utf-8")


def serve(streams, cameras, *, port, shot):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    by_name = dict(zip(check.COURSE_CAMERA_NAMES, streams))
    by_id = {policy["id"]: policy for policy in POLICIES}
    first = next(iter(cameras.values()))
    capture = f'{first["width"]}x{first["height"]}'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            parts = check.route_parts(self.path)
            if not parts:
                body = build_page(capture, "cameras.json", shot)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parts[0] == "frame" and len(parts) == 3 and parts[1] in by_id and parts[2] in by_name:
                policy = by_id[parts[1]]
                jpeg = check.encode_jpeg(
                    resized(by_name[parts[2]].current(), policy["target"], policy["pad"]))
                if jpeg is None:
                    self.send_error(503)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", default="cameras.json", help="the configuration to read")
    parser.add_argument("--port", type=int, default=check.DEFAULT_PORT, help="port for the page")
    parser.add_argument("--shot", type=int, default=224, help="how large to draw each panel, in pixels")
    args = parser.parse_args(argv)
    try:
        cameras = check.load_cameras_file(args.cameras)
        entries = [{"card": name, "role": name, "path": camera["path"], "rotation": camera["rotation"]}
                   for name, camera in cameras.items()]
        first = next(iter(cameras.values()))
        size = check.size_for(first["width"], first["height"], first["rotation"])
        streams = check.start_streams(entries, size=size)
        if len(streams) != len(entries):
            for stream in streams:
                stream.close()
            print("STOP: not every configured camera could be opened.", file=sys.stderr)
            return 1
        try:
            serve(streams, cameras, port=args.port, shot=args.shot)
        finally:
            for stream in streams:
                stream.close()
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
