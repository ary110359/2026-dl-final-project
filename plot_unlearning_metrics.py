import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt


def _parse_class_list(value):
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _load_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _infer_forget_sequence(rows):
    sequence = []
    for row in rows:
        class_id = int(row["current_class"])
        if class_id not in sequence:
            sequence.append(class_id)
    return sequence


def _load_cifar100_names(data_root):
    meta_path = Path(data_root) / "cifar-100-python" / "meta"
    if not meta_path.exists():
        return {}

    import pickle

    with meta_path.open("rb") as f:
        meta = pickle.load(f, encoding="latin1")
    return {idx: name for idx, name in enumerate(meta.get("fine_label_names", []))}


def _class_accuracy_key(class_id):
    return f"class_{class_id}_accuracy"


def _class_values(rows, class_id, csv_path):
    key = _class_accuracy_key(class_id)
    if key not in rows[0]:
        raise KeyError(f"{key} is not present in {csv_path}")
    return [float(row[key]) for row in rows]


def _save_figure(output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_overlay_trajectories(
    csv_path,
    output_path,
    classes=None,
    data_root="./data",
    threshold=10.0,
    title=None,
):
    rows = _load_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    steps = [int(row["step"]) for row in rows]
    class_ids = classes if classes else _infer_forget_sequence(rows)
    class_names = _load_cifar100_names(data_root)

    plt.figure(figsize=(14, 7.6))
    colors = plt.get_cmap("tab20").colors
    for idx, class_id in enumerate(class_ids):
        values = _class_values(rows, class_id, csv_path)
        label = class_names.get(class_id, f"class {class_id}")
        plt.plot(
            steps,
            values,
            color=colors[idx % len(colors)],
            linewidth=1.9,
            alpha=0.88,
            label=f"{label} ({class_id})",
        )

    plt.axhline(
        threshold,
        color="#555555",
        linestyle="--",
        linewidth=1.3,
        label=f"{threshold:g}% threshold",
    )
    plt.xlabel("Unlearning step", fontsize=12)
    plt.ylabel("Class accuracy (%)", fontsize=12)
    plt.title(title or "Per-class Accuracy During Sequential Unlearning", fontsize=14)
    plt.xticks(steps)
    plt.ylim(-5, 105)
    plt.grid(alpha=0.28)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=5,
        fontsize=8.5,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.8,
    )
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    _save_figure(output_path)


def plot_class_trajectories(
    csv_path,
    output_path,
    classes=None,
    data_root="./data",
    threshold=10.0,
    title=None,
):
    plot_overlay_trajectories(
        csv_path=csv_path,
        output_path=output_path,
        classes=classes,
        data_root=data_root,
        threshold=threshold,
        title=title,
    )


def main():
    parser = argparse.ArgumentParser(description="Plot sequential unlearning metrics from CSV.")
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to sequential_metrics.csv produced by main_forget.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to <csv_dir>/class_trajectories.png.",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class ids to plot. Defaults to classes in forget_sequence.",
    )
    parser.add_argument(
        "--data",
        default="./data",
        help="Dataset root used to read CIFAR-100 class names.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Horizontal forgetting threshold line.",
    )
    parser.add_argument("--title", default=None, help="Optional plot title.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output = args.output or str(csv_path.parent / "class_trajectories.png")
    classes = _parse_class_list(args.classes) if args.classes else None

    plot_class_trajectories(
        csv_path=csv_path,
        output_path=output,
        classes=classes,
        data_root=args.data,
        threshold=args.threshold,
        title=args.title,
    )


if __name__ == "__main__":
    main()
