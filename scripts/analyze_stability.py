import argparse
from pathlib import Path
from typing import List

import pandas as pd

STABILITY_METRICS = [
    "entropy_mean",
    "entropy_p95",
    "entropy_p10",
    "margin_mean",
    "margin_p95",
    "margin_p10",
    "kl_to_base_mean",
    "kl_to_base_p95",
    "kl_to_base_p10",
]

DELTA_METRICS = [
    "entropy_mean",
    "entropy_p95",
    "margin_mean",
    "margin_p95",
    "kl_to_base_mean",
    "kl_to_base_p95",
]


def format_mean_std(mean_value, std_value, decimals: int = 3) -> str:
    if pd.isna(mean_value):
        return "—"
    if pd.isna(std_value):
        std_value = 0.0
    return f"{mean_value:.{decimals}f} ± {std_value:.{decimals}f}"


def make_checkpoint_label(step: int, checkpoint_task: str) -> str:
    return "Base" if step == 0 else f"After {checkpoint_task}"


def load_stability_scores(input_path: Path) -> pd.DataFrame:
    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted(input_path.glob("stability_seed_*/stability_scores.csv"))

    if not paths:
        raise FileNotFoundError(f"No stability scores found at {input_path}")

    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    required = {
        "seed",
        "step",
        "checkpoint_task",
        "num_reference_examples",
        *STABILITY_METRICS,
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["seed"] = df["seed"].astype(int)
    df["step"] = df["step"].astype(int)

    for metric in STABILITY_METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    base_mask = df["step"] == 0
    for col in ["kl_to_base_mean", "kl_to_base_p95", "kl_to_base_p10"]:
        df.loc[base_mask, col] = df.loc[base_mask, col].fillna(0.0)

    df = df.drop_duplicates(
        subset=["seed", "step", "checkpoint_task"],
        keep="last",
    )

    df["checkpoint"] = df.apply(
        lambda row: make_checkpoint_label(row["step"], row["checkpoint_task"]),
        axis=1,
    )

    return df


def add_delta_vs_base(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for seed in sorted(df["seed"].unique()):
        seed_df = df[df["seed"] == seed].copy()
        base = seed_df[seed_df["step"] == 0]

        if base.empty:
            raise ValueError(f"Seed {seed} has no base row.")

        base_row = base.iloc[0]

        for metric in DELTA_METRICS:
            seed_df[f"{metric}_delta_vs_base"] = seed_df[metric] - base_row[metric]

        rows.append(seed_df)

    return pd.concat(rows, ignore_index=True)


def summarize_by_step(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []

    for (step, checkpoint_task, checkpoint), group in df.groupby(
        ["step", "checkpoint_task", "checkpoint"],
        dropna=False,
    ):
        row = {
            "step": int(step),
            "checkpoint_task": checkpoint_task,
            "checkpoint": checkpoint,
            "n_seeds": group["seed"].nunique(),
            "num_reference_examples": int(group["num_reference_examples"].iloc[0]),
        }

        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
            row[f"{metric}_count"] = values.count()

        rows.append(row)

    return pd.DataFrame(rows).sort_values("step")


def make_stability_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for row in summary.itertuples(index=False):
        rows.append(
            {
                "step": row.step,
                "checkpoint": row.checkpoint,
                "entropy_mean": format_mean_std(
                    row.entropy_mean_mean, row.entropy_mean_std
                ),
                "entropy_p95": format_mean_std(
                    row.entropy_p95_mean, row.entropy_p95_std
                ),
                "margin_mean": format_mean_std(
                    row.margin_mean_mean, row.margin_mean_std
                ),
                "margin_p95": format_mean_std(row.margin_p95_mean, row.margin_p95_std),
                "kl_to_base_mean": format_mean_std(
                    row.kl_to_base_mean_mean, row.kl_to_base_mean_std
                ),
                "kl_to_base_p95": format_mean_std(
                    row.kl_to_base_p95_mean, row.kl_to_base_p95_std
                ),
            }
        )

    return pd.DataFrame(rows)


def make_delta_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for row in summary.itertuples(index=False):
        rows.append(
            {
                "step": row.step,
                "checkpoint": row.checkpoint,
                "delta_entropy_mean": format_mean_std(
                    row.entropy_mean_delta_vs_base_mean,
                    row.entropy_mean_delta_vs_base_std,
                ),
                "delta_entropy_p95": format_mean_std(
                    row.entropy_p95_delta_vs_base_mean,
                    row.entropy_p95_delta_vs_base_std,
                ),
                "delta_margin_mean": format_mean_std(
                    row.margin_mean_delta_vs_base_mean,
                    row.margin_mean_delta_vs_base_std,
                ),
                "delta_margin_p95": format_mean_std(
                    row.margin_p95_delta_vs_base_mean,
                    row.margin_p95_delta_vs_base_std,
                ),
                "kl_to_base_mean": format_mean_std(
                    row.kl_to_base_mean_mean,
                    row.kl_to_base_mean_std,
                ),
            }
        )

    return pd.DataFrame(rows)


def analyze_stability(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_stability_scores(input_path)
    df = add_delta_vs_base(df)

    all_metrics = STABILITY_METRICS + [
        f"{metric}_delta_vs_base" for metric in DELTA_METRICS
    ]

    summary = summarize_by_step(df, all_metrics)

    stability_table = make_stability_table(summary)
    delta_table = make_delta_table(summary)

    df.to_csv(output_dir / "stability_all.csv", index=False)
    summary.to_csv(output_dir / "stability_summary_numeric.csv", index=False)
    stability_table.to_csv(output_dir / "table_stability.csv", index=False)
    delta_table.to_csv(output_dir / "table_stability_delta_vs_base.csv", index=False)

    print(f"Saved: {output_dir / 'stability_all.csv'}")
    print(f"Saved: {output_dir / 'stability_summary_numeric.csv'}")
    print(f"Saved: {output_dir / 'table_stability.csv'}")
    print(f"Saved: {output_dir / 'table_stability_delta_vs_base.csv'}")

    print("\nStability table:")
    print(stability_table.to_string(index=False))

    print("\nDelta vs base table:")
    print(delta_table.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif input_path.is_dir():
        output_dir = input_path / "analysis"
    else:
        output_dir = input_path.parent

    analyze_stability(input_path, output_dir)


if __name__ == "__main__":
    main()
