import argparse
import csv
import os
import re
import shutil
from dataclasses import dataclass


CKPT_RE = re.compile(r"^ckpt_iter_(\d{4})\.pt$")
SAMPLES_RE = re.compile(r"^samples_iter_(\d{4})\.npz$")
METRIC_FILES = ("metrics.csv", "metrics_steps.csv", "metrics_gates.csv")


@dataclass
class CsvTrimStats:
    kept_rows: int = 0
    dropped_rows: int = 0


def _move_if_iter_gt(filename: str, target_iter: int, pattern: re.Pattern[str], src_dir: str, dst_dir: str) -> bool:
    match = pattern.match(filename)
    if not match:
        return False
    iteration = int(match.group(1))
    if iteration <= target_iter:
        return False
    src = os.path.join(src_dir, filename)
    dst = os.path.join(dst_dir, filename)
    shutil.move(src, dst)
    return True


def _trim_metrics_csv(path: str, target_iter: int) -> CsvTrimStats:
    stats = CsvTrimStats()
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return stats

    header = rows[0]
    kept = [header]
    for row in rows[1:]:
        if not row:
            continue
        try:
            row_iter = int(row[0])
        except (ValueError, IndexError):
            stats.dropped_rows += 1
            continue
        if row_iter <= target_iter:
            kept.append(row)
            stats.kept_rows += 1
        else:
            stats.dropped_rows += 1

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(kept)
    return stats


def rewind_checkpoint_dir(checkpoint_dir: str, target_iter: int) -> None:
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")

    target_ckpt = os.path.join(checkpoint_dir, f"ckpt_iter_{target_iter:04d}.pt")
    if not os.path.exists(target_ckpt):
        raise FileNotFoundError(f"target checkpoint not found: {target_ckpt}")

    archive_dir = os.path.join(checkpoint_dir, f"rewind_backup_to_{target_iter:04d}")
    os.makedirs(archive_dir, exist_ok=True)

    moved_ckpts = 0
    moved_samples = 0
    for name in os.listdir(checkpoint_dir):
        if _move_if_iter_gt(name, target_iter, CKPT_RE, checkpoint_dir, archive_dir):
            moved_ckpts += 1
            continue
        if _move_if_iter_gt(name, target_iter, SAMPLES_RE, checkpoint_dir, archive_dir):
            moved_samples += 1

    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(latest_path):
        shutil.copy2(latest_path, os.path.join(archive_dir, "latest.pt.before_rewind"))
    shutil.copy2(target_ckpt, latest_path)

    csv_stats: dict[str, CsvTrimStats] = {}
    for metrics_name in METRIC_FILES:
        metrics_path = os.path.join(checkpoint_dir, metrics_name)
        if not os.path.exists(metrics_path):
            continue
        backup_path = os.path.join(archive_dir, f"{metrics_name}.before_rewind")
        shutil.copy2(metrics_path, backup_path)
        csv_stats[metrics_name] = _trim_metrics_csv(metrics_path, target_iter)

    print(f"rewind complete: checkpoint_dir={checkpoint_dir}")
    print(f"target_iter={target_iter}")
    print(f"archive_dir={archive_dir}")
    print(f"moved_checkpoints={moved_ckpts}")
    print(f"moved_sample_shards={moved_samples}")
    print(f"latest_pt_now_points_to=ckpt_iter_{target_iter:04d}.pt")
    for metrics_name in METRIC_FILES:
        stats = csv_stats.get(metrics_name)
        if stats is None:
            print(f"{metrics_name}: missing (skipped)")
        else:
            print(
                f"{metrics_name}: kept_rows={stats.kept_rows}, "
                f"dropped_rows={stats.dropped_rows}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewind training artifacts to a target iteration.")
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory to rewind.")
    parser.add_argument("--to-iter", type=int, required=True, help="Target iteration to keep.")
    args = parser.parse_args()

    rewind_checkpoint_dir(args.checkpoint_dir, args.to_iter)


if __name__ == "__main__":
    main()
