import argparse
from pathlib import Path

import pandas as pd


def format_mean_std(mean, std, decimals=3):
    if pd.isna(std):
        std = 0.0
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def load_score_files(run_dir: Path) -> pd.DataFrame:
    score_files = sorted(run_dir.glob("scores_seed_*.csv"))

    if not score_files:
        raise FileNotFoundError(f"No scores_seed_*.csv files found in {run_dir}")

    dfs = []
    for path in score_files:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        dfs.append(df)

    scores = pd.concat(dfs, ignore_index=True)

    required_cols = {
        "seed",
        "step",
        "checkpoint_task",
        "eval_task",
        "eval_task_name",
        "accuracy",
        "correct",
        "total",
    }

    missing = required_cols - set(scores.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    scores["seed"] = scores["seed"].astype(int)
    scores["step"] = scores["step"].astype(int)
    scores["accuracy"] = scores["accuracy"].astype(float)

    # Avoid duplicated rows if a file was appended multiple times.
    scores = scores.drop_duplicates(
        subset=["seed", "step", "checkpoint_task", "eval_task"],
        keep="last",
    )

    return scores


def build_checkpoint_label(row):
    if row["step"] == 0:
        return "Base"

    return f"After {row['checkpoint_task']}"


def combine_scores(run_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = load_score_files(run_dir)
    scores.to_csv(output_dir / "scores_all.csv", index=False)

    scores["checkpoint"] = scores.apply(build_checkpoint_label, axis=1)

    # Per-seed wide matrix.
    by_seed = scores.pivot_table(
        index=["seed", "step", "checkpoint", "checkpoint_task"],
        columns="eval_task_name",
        values="accuracy",
        aggfunc="last",
    ).reset_index()

    by_seed.to_csv(output_dir / "accuracy_matrix_by_seed.csv", index=False)

    # Mean/std long summary.
    summary_long = scores.groupby(
        ["step", "checkpoint", "checkpoint_task", "eval_task", "eval_task_name"],
        as_index=False,
    ).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        n_seeds=("accuracy", "count"),
    )

    summary_long.to_csv(output_dir / "accuracy_matrix_long_summary.csv", index=False)

    # Mean-only matrix.
    mean_matrix = summary_long.pivot_table(
        index=["step", "checkpoint", "checkpoint_task"],
        columns="eval_task_name",
        values="accuracy_mean",
        aggfunc="last",
    ).reset_index()

    mean_matrix.to_csv(output_dir / "accuracy_matrix_mean.csv", index=False)

    # Mean ± std matrix.
    summary_long["mean_std"] = summary_long.apply(
        lambda row: format_mean_std(row["accuracy_mean"], row["accuracy_std"]),
        axis=1,
    )

    mean_std_matrix = summary_long.pivot_table(
        index=["step", "checkpoint", "checkpoint_task"],
        columns="eval_task_name",
        values="mean_std",
        aggfunc="last",
    ).reset_index()

    mean_std_matrix.to_csv(output_dir / "accuracy_matrix_mean_std.csv", index=False)

    print(f"Loaded {len(scores)} score rows from {run_dir}")
    print(f"Seeds: {sorted(scores['seed'].unique())}")
    print(f"Saved outputs to {output_dir}")
    print()
    print("Mean ± std accuracy matrix:")
    print(mean_std_matrix.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Combine scores_seed_*.csv files into accuracy matrices."
    )

    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Directory containing scores_seed_*.csv files.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <run-dir>/analysis.",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis"

    combine_scores(run_dir=run_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()
