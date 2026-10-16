"""Independent cloud deadline. Python 3.8+ standard library; no API credential."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    shutdown = Path("/usr/bin/shutdown")
    if not os.access(shutdown, os.X_OK):
        raise RuntimeError("AutoDL shutdown executable unavailable")
    args.receipt.write_text(json.dumps({"pid": os.getpid(), "deadline": args.deadline,
                                        "ready": True, "checked_at": time.time()}), encoding="utf-8")
    time.sleep(max(0, args.deadline - time.time()))
    subprocess.run([str(shutdown)], check=True)


if __name__ == "__main__":
    main()
