import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected {path} to contain a JSON list, got {type(data).__name__}")

    return data


def write_json(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def find_val_files(dataset_root: Path, val_filename: str) -> List[Path]:
    val_files = sorted(dataset_root.glob(f"*/{val_filename}"))

    if not val_files:
        raise FileNotFoundError(
            f"No '{val_filename}' files found in immediate subfolders of {dataset_root}"
        )

    return val_files


def split_val_file(
    val_file: Path,
    dataset_root: Path,
    ref_fraction: float,
    rng: random.Random,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = read_json(val_file)

    if not rows:
        raise ValueError(f"Validation file is empty: {val_file}")

    indices = list(range(len(rows)))
    rng.shuffle(indices)

    n_ref = max(1, round(len(rows) * ref_fraction))
    n_ref = min(n_ref, len(rows))

    ref_indices = set(indices[:n_ref])
    task_id = val_file.parent.name

    reference_rows = []
    val_dev_rows = []

    for i, row in enumerate(rows):
        row = dict(row)

        row.setdefault("task_id", task_id)
        row.setdefault("task_name", task_id)
        row.setdefault("dataset_root", str(dataset_root))
        row.setdefault("source_file", str(val_file.relative_to(dataset_root)))

        if i in ref_indices:
            row["split"] = "reference"
            reference_rows.append(row)
        else:
            row["split"] = "val_dev"
            val_dev_rows.append(row)

    return val_dev_rows, reference_rows


def build_reference_set(
    dataset_root: Path,
    val_filename: str,
    val_dev_filename: str,
    reference_filename: str,
    ref_fraction: float,
    seed: int,
    overwrite: bool,
) -> None:
    if not 0.0 < ref_fraction < 1.0:
        raise ValueError("ref_fraction must be between 0 and 1.")

    rng = random.Random(seed)
    val_files = find_val_files(dataset_root, val_filename)

    reference_rows = []

    print(f"Dataset root: {dataset_root}")
    print(f"Validation filename: {val_filename}")
    print(f"Reference fraction: {ref_fraction}")
    print(f"Seed: {seed}")
    print()

    for val_file in val_files:
        val_dev_path = val_file.parent / val_dev_filename

        if val_dev_path.exists() and not overwrite:
            raise FileExistsError(
                f"{val_dev_path} already exists. Use --overwrite to replace it."
            )

        val_dev_rows, ref_rows = split_val_file(
            val_file=val_file,
            dataset_root=dataset_root,
            ref_fraction=ref_fraction,
            rng=rng,
        )

        write_json(val_dev_rows, val_dev_path)
        reference_rows.extend(ref_rows)

        print(
            f"{val_file.parent.name}: "
            f"original_val={len(val_dev_rows) + len(ref_rows)}, "
            f"val_dev={len(val_dev_rows)}, "
            f"reference={len(ref_rows)}"
        )

    rng.shuffle(reference_rows)

    reference_path = dataset_root / reference_filename

    if reference_path.exists() and not overwrite:
        raise FileExistsError(
            f"{reference_path} already exists. Use --overwrite to replace it."
        )

    write_json(reference_rows, reference_path)

    print()
    print(f"Saved combined reference set: {reference_path}")
    print(f"Total reference examples: {len(reference_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split task validation JSON files into val_dev files and a combined reference set."
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root folder containing one subfolder per task.",
    )

    parser.add_argument(
        "--val-filename",
        type=str,
        default="eval.json",
        help="Name of the validation JSON file inside each task folder.",
    )

    parser.add_argument(
        "--val-dev-filename",
        type=str,
        default="eval_dev.json",
        help="Name of the new validation/dev JSON file written inside each task folder.",
    )

    parser.add_argument(
        "--reference-filename",
        type=str,
        default="reference.json",
        help="Name of the combined reference JSON file written at the dataset root.",
    )

    parser.add_argument(
        "--ref-fraction",
        type=float,
        default=0.2,
        help="Fraction of each validation file to place in the reference set.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=33,
        help="Random seed for deterministic splitting.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing val_dev/reference files.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    build_reference_set(
        dataset_root=Path(args.dataset_root),
        val_filename=args.val_filename,
        val_dev_filename=args.val_dev_filename,
        reference_filename=args.reference_filename,
        ref_fraction=args.ref_fraction,
        seed=args.seed,
        overwrite=args.overwrite,
    )