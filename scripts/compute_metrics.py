#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def make_checkpoint_label(step: int, checkpoint_task: str) -> str:
    if step == 0:
        return "Base"
    return f"After {checkpoint_task}"


def format_mean_std(mean_value, std_value, decimals: int = 3) -> str:
    if pd.isna(mean_value):
        return "—"

    if pd.isna(std_value):
        std_value = 0.0

    return f"{mean_value:.{decimals}f} ± {std_value:.{decimals}f}"


def first_non_null(values):
    for value in values:
        if pd.notna(value):
            return value
    return None


def mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def load_scores(scores_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(scores_csv)

    required_columns = {
        "seed",
        "step",
        "checkpoint_task",
        "eval_task",
        "eval_task_name",
        "accuracy",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["seed"] = df["seed"].astype(int)
    df["step"] = df["step"].astype(int)
    df["accuracy"] = df["accuracy"].astype(float)

    return df


def infer_task_order(df: pd.DataFrame) -> List[str]:
    """
    Infer task order from the step=0 rows.

    Assumes eval_task names are ordered as task_1, task_2, task_3, ...
    """
    base_df = df[df["step"] == 0]

    if base_df.empty:
        raise ValueError("Cannot infer task order because no step=0 rows were found.")

    return (
        base_df[["eval_task"]]
        .drop_duplicates()
        .sort_values("eval_task")["eval_task"]
        .tolist()
    )


def make_score_matrix(seed_df: pd.DataFrame) -> Dict[tuple[int, str], float]:
    """
    Build R[(step, task)] = accuracy.
    """
    return {
        (int(row.step), str(row.eval_task)): float(row.accuracy)
        for row in seed_df.itertuples()
    }


def compute_op(
    R: Dict[tuple[int, str], float],
    step: int,
    task_order: List[str],
) -> Optional[float]:
    """
    Overall Performance:
    OP_t = average accuracy over tasks seen so far.
    """
    seen_tasks = task_order[:step]
    values = [R[(step, task)] for task in seen_tasks if (step, task) in R]
    return mean(values)


def compute_bwt(
    R: Dict[tuple[int, str], float],
    step: int,
    task_order: List[str],
) -> tuple[Optional[float], Dict[str, float]]:
    """
    Backward Transfer:
    BWT_t = average over previous tasks of:
        current accuracy on task i - accuracy right after task i was learned

    Negative BWT means forgetting.
    """
    if step <= 1:
        return None, {}

    previous_tasks = task_order[: step - 1]
    bwt_by_task = {}

    for task_index, task in enumerate(previous_tasks):
        learned_step = task_index + 1

        if (step, task) in R and (learned_step, task) in R:
            bwt_by_task[task] = R[(step, task)] - R[(learned_step, task)]

    return mean(list(bwt_by_task.values())), bwt_by_task


def compute_forgetting(
    bwt_by_task: Dict[str, float],
) -> tuple[Optional[float], Dict[str, float]]:
    """
    Forgetting is reported as the positive version of negative BWT.

    If BWT for a task is -0.10, forgetting is +0.10.
    """
    if not bwt_by_task:
        return None, {}

    forgetting_by_task = {task: -bwt_value for task, bwt_value in bwt_by_task.items()}

    return mean(list(forgetting_by_task.values())), forgetting_by_task


def compute_fwt(
    R: Dict[tuple[int, str], float],
    step: int,
    task_order: List[str],
) -> tuple[Optional[float], Dict[str, float]]:
    """
    Forward Transfer:
    FWT_t = average over future tasks of:
        current accuracy on task i - base accuracy on task i

    At step 0, FWT is defined as 0.0.
    """
    if step == 0:
        return 0.0, {task: 0.0 for task in task_order}

    future_tasks = task_order[step:]
    fwt_by_task = {}

    for task in future_tasks:
        if (step, task) in R and (0, task) in R:
            fwt_by_task[task] = R[(step, task)] - R[(0, task)]

    return mean(list(fwt_by_task.values())), fwt_by_task


def compute_learning_accuracy(
    R: Dict[tuple[int, str], float],
    task_order: List[str],
) -> tuple[Optional[float], Dict[str, float]]:
    """
    Learning Accuracy:
        LA = mean_k a_{k,k}

    where a_{k,k} is performance on task k immediately after it is learned.
    """
    diagonal = {}

    for task_index, task in enumerate(task_order, start=1):
        if (task_index, task) in R:
            diagonal[task] = R[(task_index, task)]

    return mean(list(diagonal.values())), diagonal


def compute_adaptation_gain(
    R: Dict[tuple[int, str], float],
    step: int,
    task_order: List[str],
) -> Optional[float]:
    """
    Immediate adaptation gain:
        Δ_adapt(k) = a_{k,k} - a_{k-1,k}

    This measures whether training on task k improved performance
    relative to the checkpoint immediately before task k was learned.
    """
    if step == 0:
        return None

    current_task = task_order[step - 1]

    before = R.get((step - 1, current_task))
    after = R.get((step, current_task))

    if before is None or after is None:
        return None

    return after - before


def compute_current_task_metrics(
    R: Dict[tuple[int, str], float],
    step: int,
    task_order: List[str],
) -> dict:
    """
    Current-task diagnostics:
      - a_{k,k}: performance immediately after task k is learned
      - Δ vs base: a_{k,k} - a_{0,k}
      - Δ_adapt: a_{k,k} - a_{k-1,k}
    """
    if step == 0:
        return {
            "current_task": None,
            "current_acc": None,
            "current_base_acc": None,
            "current_prev_acc": None,
            "current_delta_vs_base": None,
            "adaptation_gain": None,
        }

    current_task = task_order[step - 1]
    current_acc = R.get((step, current_task))
    current_base_acc = R.get((0, current_task))
    current_prev_acc = R.get((step - 1, current_task))

    if current_acc is None or current_base_acc is None:
        current_delta_vs_base = None
    else:
        current_delta_vs_base = current_acc - current_base_acc

    if current_acc is None or current_prev_acc is None:
        adaptation_gain = None
    else:
        adaptation_gain = current_acc - current_prev_acc

    return {
        "current_task": current_task,
        "current_acc": current_acc,
        "current_base_acc": current_base_acc,
        "current_prev_acc": current_prev_acc,
        "current_delta_vs_base": current_delta_vs_base,
        "adaptation_gain": adaptation_gain,
    }


def compute_avg_all_tasks(
    R: Dict[tuple[int, str], float],
    step: int,
    task_order: List[str],
) -> Optional[float]:
    """
    Average accuracy over all tasks, including seen and unseen.
    """
    values = [R[(step, task)] for task in task_order if (step, task) in R]
    return mean(values)


def compute_metrics_for_seed(
    seed_df: pd.DataFrame,
    task_order: List[str],
) -> pd.DataFrame:
    seed = int(seed_df["seed"].iloc[0])
    R = make_score_matrix(seed_df)

    learning_accuracy, learning_accuracy_by_task = compute_learning_accuracy(
        R,
        task_order,
    )

    rows = []

    for step in range(len(task_order) + 1):
        checkpoint_task = "base" if step == 0 else task_order[step - 1]

        op = compute_op(R, step, task_order)

        bwt, bwt_by_task = compute_bwt(R, step, task_order)
        forgetting, forgetting_by_task = compute_forgetting(bwt_by_task)

        fwt, fwt_by_task = compute_fwt(R, step, task_order)

        current_metrics = compute_current_task_metrics(R, step, task_order)

        next_task = task_order[step] if step < len(task_order) else None
        next_fwt = fwt_by_task.get(next_task) if next_task else None

        row = {
            "seed": seed,
            "step": step,
            "checkpoint": make_checkpoint_label(step, checkpoint_task),
            "checkpoint_task": checkpoint_task,
            "op": op,
            "bwt": bwt,
            "forgetting": forgetting,
            "fwt": fwt,
            "next_task": next_task,
            "next_fwt": next_fwt,
            "avg_all_tasks": compute_avg_all_tasks(R, step, task_order),
            "bwt_by_task": json.dumps(bwt_by_task),
            "forgetting_by_task": json.dumps(forgetting_by_task),
            "fwt_by_task": json.dumps(fwt_by_task),
            "learning_accuracy": learning_accuracy,
            "learning_accuracy_by_task": json.dumps(learning_accuracy_by_task),
        }

        row.update(current_metrics)
        rows.append(row)

    return pd.DataFrame(rows)


def compute_metrics(
    scores_df: pd.DataFrame,
    task_order: List[str],
) -> pd.DataFrame:
    all_metrics = []

    for seed in sorted(scores_df["seed"].unique()):
        seed_df = scores_df[scores_df["seed"] == seed]
        seed_metrics = compute_metrics_for_seed(seed_df, task_order)
        all_metrics.append(seed_metrics)

    return pd.concat(all_metrics, ignore_index=True)


METRIC_COLUMNS = [
    "op",
    "bwt",
    "forgetting",
    "fwt",
    "next_fwt",
    "avg_all_tasks",
    "current_acc",
    "current_base_acc",
    "current_prev_acc",
    "current_delta_vs_base",
    "adaptation_gain",
    "learning_accuracy",
]


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create numeric mean/std/count summary per checkpoint.
    """
    rows = []

    for (step, checkpoint_task), group in metrics_df.groupby(
        ["step", "checkpoint_task"],
        dropna=False,
    ):
        row = {
            "step": int(step),
            "checkpoint_task": checkpoint_task,
            "checkpoint": make_checkpoint_label(int(step), checkpoint_task),
            "current_task": first_non_null(group["current_task"]),
            "next_task": first_non_null(group["next_task"]),
            "n_seeds": group["seed"].nunique(),
        }

        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
            row[f"{metric}_count"] = values.count()

        rows.append(row)

    return pd.DataFrame(rows).sort_values("step")


def make_acquisition_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Current-task acquisition and overall average performance.
    """
    rows = []

    for row in summary_df.itertuples(index=False):
        rows.append(
            {
                "Step": row.step,
                "Checkpoint": row.checkpoint,
                "Current task": row.current_task if pd.notna(row.current_task) else "—",
                "Current accuracy": format_mean_std(
                    row.current_acc_mean,
                    row.current_acc_std,
                ),
                "Δ current vs base": format_mean_std(
                    row.current_delta_vs_base_mean,
                    row.current_delta_vs_base_std,
                ),
                "Average over all tasks": format_mean_std(
                    row.avg_all_tasks_mean,
                    row.avg_all_tasks_std,
                ),
            }
        )

    return pd.DataFrame(rows)


def make_learning_diagnostics_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for row in summary_df.itertuples(index=False):
        if row.step == 0:
            continue

        rows.append(
            {
                "Step": row.step,
                "Checkpoint": row.checkpoint,
                "Current task": row.current_task,
                "a_k,k": format_mean_std(
                    row.current_acc_mean,
                    row.current_acc_std,
                ),
                "a_k-1,k": format_mean_std(
                    row.current_prev_acc_mean,
                    row.current_prev_acc_std,
                ),
                "Δ adapt": format_mean_std(
                    row.adaptation_gain_mean,
                    row.adaptation_gain_std,
                ),
                "Δ vs base": format_mean_std(
                    row.current_delta_vs_base_mean,
                    row.current_delta_vs_base_std,
                ),
            }
        )

    return pd.DataFrame(rows)


def make_transfer_forgetting_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    OP, BWT, Forgetting, FWT, and next-task FWT.
    """
    rows = []

    for row in summary_df.itertuples(index=False):
        rows.append(
            {
                "Step": row.step,
                "Checkpoint": row.checkpoint,
                "OP": format_mean_std(row.op_mean, row.op_std),
                "BWT": format_mean_std(row.bwt_mean, row.bwt_std),
                "Forgetting": format_mean_std(row.forgetting_mean, row.forgetting_std),
                "FWT": format_mean_std(row.fwt_mean, row.fwt_std),
                "Next task": row.next_task if pd.notna(row.next_task) else "—",
                "Next-task FWT": format_mean_std(row.next_fwt_mean, row.next_fwt_std),
            }
        )

    return pd.DataFrame(rows)


def make_final_checkpoint_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compact final checkpoint summary.
    """
    final_step = summary_df["step"].max()
    final_df = summary_df[summary_df["step"] == final_step]

    rows = []

    for row in final_df.itertuples(index=False):
        rows.append(
            {
                "Final checkpoint": row.checkpoint,
                "OP": format_mean_std(row.op_mean, row.op_std),
                "BWT": format_mean_std(row.bwt_mean, row.bwt_std),
                "Forgetting": format_mean_std(row.forgetting_mean, row.forgetting_std),
                "Average over all tasks": format_mean_std(
                    row.avg_all_tasks_mean,
                    row.avg_all_tasks_std,
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute CL metrics from scores_all.csv."
    )

    parser.add_argument(
        "--scores-csv",
        type=str,
        required=True,
        help="Path to analysis/scores_all.csv created by combine_scores.py.",
    )

    parser.add_argument(
        "--task-order",
        type=str,
        default=None,
        help="Comma-separated task order, e.g. task_1,task_2,task_3. "
        "If omitted, inferred from step=0 rows.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save metrics. Defaults to the directory of scores-csv.",
    )

    args = parser.parse_args()

    scores_csv = Path(args.scores_csv)
    output_dir = Path(args.output_dir) if args.output_dir else scores_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_df = load_scores(scores_csv)

    if args.task_order:
        task_order = [x.strip() for x in args.task_order.split(",")]
    else:
        task_order = infer_task_order(scores_df)

    print(f"Task order: {task_order}")

    metrics_df = compute_metrics(scores_df, task_order)
    summary_df = summarize_metrics(metrics_df)

    acquisition_table = make_acquisition_table(summary_df)
    transfer_table = make_transfer_forgetting_table(summary_df)
    learning_table = make_learning_diagnostics_table(summary_df)
    final_table = make_final_checkpoint_table(summary_df)

    metrics_path = output_dir / "cl_metrics_by_seed.csv"
    summary_path = output_dir / "cl_metrics_summary_numeric.csv"

    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    acquisition_table.to_csv(output_dir / "table_acquisition.csv", index=False)
    transfer_table.to_csv(output_dir / "table_transfer_forgetting.csv", index=False)
    learning_table.to_csv(output_dir / "table_learning_diagnostics.csv", index=False)
    final_table.to_csv(output_dir / "table_final_checkpoint.csv", index=False)

    print(f"Saved: {metrics_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {output_dir / 'table_acquisition.csv'}")
    print(f"Saved: {output_dir / 'table_transfer_forgetting.csv'}")
    print(f"Saved: {output_dir / 'table_learning_diagnostics.csv'}")
    print(f"Saved: {output_dir / 'table_final_checkpoint.csv'}")

    print("\nAcquisition table:")
    print(acquisition_table.to_string(index=False))

    print("\nTransfer / forgetting table:")
    print(transfer_table.to_string(index=False))

    print("\nLearning diagnostics table:")
    print(learning_table.to_string(index=False))

    print("\nFinal checkpoint table:")
    print(final_table.to_string(index=False))

    print("\nLearning diagnostics table:")
    print(learning_table.to_string(index=False))


if __name__ == "__main__":
    main()
