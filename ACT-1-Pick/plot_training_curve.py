"""Plot recorded ACT metrics from a CSV or a LeRobot console log; never runs a model."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re
import webbrowser

METRICS = ("loss", "l1_loss", "kld_loss")
LABELS = ("Total loss", "Action L1", "KL (unweighted)")
DEFAULT_GAP_STEPS = 10000  # Separate widely spaced fragments instead of drawing across missing logs.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_TOKEN = re.compile(r"(?:^|\s)step:(\d+(?:\.\d+)?[KMB]?)\s")
SCALES = {"K": 1000, "M": 1000000, "B": 1000000000}


def read_metrics(path: Path) -> tuple[list[dict[str, float]], bool]:
    """CSV steps are exact; console step labels such as 1K are explicitly approximate."""
    rows: list[dict[str, float]] = []
    approximate = False
    with path.open('rb') as stream:
        prefix = stream.read(3)
    # Windows PowerShell Tee-Object can write UTF-16; modern logs and the example use UTF-8.
    encoding = 'utf-16' if prefix[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig'
    if path.suffix.lower() == ".csv":
        with path.open(encoding=encoding, newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"step", *METRICS}.issubset(reader.fieldnames or []):
                raise ValueError("CSV requires step, loss, l1_loss and kld_loss columns")
            for record in reader:
                rows.append({key: float(record[key]) for key in ("step", *METRICS)})
    else:
        with path.open(encoding=encoding, errors="replace") as stream:
            for raw in stream:
                # Progress bars use carriage returns; log messages can share a line with them.
                for line in ANSI_ESCAPE.sub("", raw).split("\r"):
                    step = STEP_TOKEN.search(line)
                    if not step:
                        continue
                    values = {key: re.search(r"(?:^|\s)" + key + r":([^\s]+)", line) for key in METRICS}
                    if not all(values.values()):
                        continue
                    token = step[1]
                    scale = SCALES.get(token[-1], 1)
                    approximate |= scale != 1
                    number = token[:-1] if scale != 1 else token
                    rows.append({"step": float(number) * scale, **{key: float(match[1]) for key, match in values.items()}})
    if not rows:
        raise ValueError("No ACT loss records found; checkpoints do not contain historical loss")
    if any(not all(math.isfinite(value) for value in row.values()) or row["step"] < 0 for row in rows):
        raise ValueError("Nonfinite metrics or negative steps found; inspect the source log")
    return rows, approximate


def metric_sections(rows: list[dict[str, float]], gap_steps: int) -> list[list[dict[str, float]]]:
    sections: list[list[dict[str, float]]] = []
    for row in rows:
        if not sections or row["step"] < sections[-1][-1]["step"] or row["step"] - sections[-1][-1]["step"] > gap_steps:
            sections.append([])
        sections[-1].append(row)
    return sections


def plot_metrics(rows: list[dict[str, float]], approximate: bool, output: Path, gap_steps: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sections = metric_sections(rows, gap_steps)
    columns = len(sections)
    fig, axes = plt.subplots(len(METRICS), columns, figsize=(7 * columns, 9), squeeze=False)
    for column, section in enumerate(sections):
        for row_index, (metric, label) in enumerate(zip(METRICS, LABELS)):
            axis = axes[row_index, column]
            axis.plot([row["step"] for row in section], [row[metric] for row in section], marker="o", markersize=4, label=label)
            axis.set_title(f"{label} | recorded steps {section[0]['step']:g} - {section[-1]['step']:g}")
            axis.set_xlabel("Training step (rounded console labels)" if approximate else "Training step")
            axis.set_ylabel("Training interval mean")
            axis.grid(alpha=0.2)
    fig.suptitle(f"ACT training metrics: {len(rows)} recorded points")
    note = "Raw recorded points; large gaps and resumed sections are separated.\nTraining metrics, not validation loss or robot success. No missing values reconstructed.\nValues retain input precision; console-derived zeros may be rounded."
    if approximate:
        note += " Console step labels are approximate; use checkpoint JSON for exact completion."
    fig.text(0.5, 0.02, note, ha="center", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV metrics or a captured LeRobot console log")
    parser.add_argument("--output", required=True, type=Path, help="Derived PNG file (replaced when regenerated)")
    parser.add_argument("--gap-steps", type=int, default=DEFAULT_GAP_STEPS, help="Split fragments beyond this step gap")
    parser.add_argument("--open", action="store_true", help="Open the resulting PNG in the default browser")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png" or args.gap_steps <= 0:
        parser.error("--output must end in .png and --gap-steps must be positive")
    try:
        rows, approximate = read_metrics(args.input)
        plot_metrics(rows, approximate, args.output, args.gap_steps)
    except ImportError as error:
        parser.exit(1, f"{error}\nInstall plotting dependencies with: python -m pip install 'matplotlib>=3.10.3,<4'\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Recorded points: {len(rows)}; step range: {rows[0]['step']:g} - {rows[-1]['step']:g}")
    if approximate:
        print("Console step labels are rounded; the plot does not recover exact step indices.")
    print(f"Curve: {args.output.resolve()}")
    if args.open and not webbrowser.open(args.output.resolve().as_uri()):
        print("Browser opening was unavailable; open the PNG file manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
