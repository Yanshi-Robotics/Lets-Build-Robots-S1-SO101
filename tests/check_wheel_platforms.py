#!/usr/bin/env python3
"""Check that the packages behind Lesson 8 publish wheels where the course says they do.

    python tests/check_wheel_platforms.py

Lesson 8 tells the reader it cannot be installed on Windows. That is a claim about what is
on PyPI today, not about this repository, so it belongs in a check rather than in prose
alone: if placo starts publishing a Windows wheel, this fails and the lesson gets rewritten
instead of quietly staying wrong.

Nothing here touches hardware or installs anything; it reads the PyPI JSON API.
"""
import json
import sys
import urllib.error
import urllib.request

sys.dont_write_bytecode = True

# Each package maps to the releases to check. `None` means "whatever is current".
#
# placo gets two entries on purpose. 0.9.15 is what a reader actually installs, because
# lerobot 0.6.1 pins `placo>=0.9.6,<0.9.16` through its placo-dep extra. The current release
# is checked as well, since the course says no version has a Windows wheel and a new release
# is where one would first appear.
PACKAGES = {
    "placo": ["0.9.15", None],
    "cmeel-urdfdom": ["4.0.1"],
    "cmeel-tinyxml2": ["10.0.0"],
    "cmeel-boost": [None],
    "cmeel-assimp": [None],
    "cmeel-octomap": [None],
}
# What Lesson 8, and footnote 7 of Lesson 1, promise: a macOS wheel for Intel and one for
# Apple Silicon. The check looks at the architecture, not the minimum macOS version in the
# tag — a package raising its floor from macosx_10_9 to macosx_10_13 is routine and does not
# change what the course tells the reader.
EXPECTED_MACOS_ARCHITECTURES = {"x86_64": "Intel", "arm64": "Apple Silicon"}


def wheel_platforms(name, version):
    """The platform tags of every wheel published for one release."""
    url = f"https://pypi.org/pypi/{name}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.load(response)
    release = version or data["info"]["version"]
    files = data["releases"].get(release)
    if not files:
        raise SystemExit(f"{name}: no release {release} on PyPI")
    tags = set()
    for entry in files:
        if not entry["filename"].endswith(".whl"):
            continue
        # A wheel filename ends with python-abi-platform.whl; the platform can hold dots,
        # so take everything after the ABI tag rather than splitting on every dash.
        tags.add(entry["filename"][: -len(".whl")].rsplit("-", 1)[-1])
    return release, tags


def main():
    problems = []
    checks = [(name, pin) for name, pins in sorted(PACKAGES.items()) for pin in pins]
    for name, pin in checks:
        try:
            release, tags = wheel_platforms(name, pin)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SystemExit(f"{name}: could not reach PyPI ({exc})")
        windows = sorted(tag for tag in tags if "win" in tag)
        macos = sorted(tag for tag in tags if tag.startswith("macosx"))
        print(f"{name} {release}: {len(tags)} wheel platform(s)")
        if windows:
            problems.append(
                f"{name} {release} now publishes a Windows wheel ({', '.join(windows)}). "
                "Lesson 8 says it cannot be installed on Windows; rewrite that lesson, the "
                "Lesson 1 footnotes, the PLATFORM_COPY boundaries and the README."
            )
        # Intel and Apple Silicon are promised separately, so both are checked.
        for architecture, plain_name in EXPECTED_MACOS_ARCHITECTURES.items():
            if not any(tag.endswith(architecture) for tag in macos):
                problems.append(
                    f"{name} {release} publishes no {plain_name} macOS wheel "
                    f"(has {', '.join(macos) or 'none'}). Lesson 8 and footnote 7 of Lesson 1 "
                    "say macOS works on both Intel and Apple Silicon."
                )
    if problems:
        print("\nThe course and PyPI disagree:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"\n{len(checks)} releases checked. Every one matches what Lesson 8 claims: macOS on both architectures, no Windows wheel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
