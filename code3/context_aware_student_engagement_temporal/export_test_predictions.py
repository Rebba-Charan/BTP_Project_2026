"""Export per-video baseline and temporal predictions for the held-out test set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from evaluate_comparison import (
    build_model,
    build_test_loader,
    load_class_names,
    make_config,
    predict,
    select_device,
    set_seed,
)
from ViFiCLIP.utils.logger import create_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint",
        default="pretrained_action_recog_models/ckpt_epoch_30.pth",
    )
    parser.add_argument(
        "--temporal-checkpoint",
        default="outputs/temporal_training/best_temporal.pth",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/test_prediction_csv",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def write_requested_csv(rows: list[dict], dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["folder_name", "predicted", "actual"],
        )
        writer.writeheader()
        for row, video_info in zip(rows, dataset.video_infos):
            relative_path = Path(video_info["filename"])
            writer.writerow(
                {
                    "folder_name": relative_path.parent.name,
                    "predicted": row["predicted_label"],
                    "actual": row["true_label"],
                }
            )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    config, _ = make_config(args)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(str(output_dir), 0, "export_test_predictions")
    set_seed(config.SEED)
    device = select_device(args.device)
    names = load_class_names(config.DATA.LABEL_LIST)
    dataset, loader = build_test_loader(config, root)

    baseline = build_model(
        config, names, args.base_checkpoint, None, device, logger
    )
    baseline_rows = predict(
        baseline, loader, dataset, names, device, "baseline_mean_pooling"
    )
    write_requested_csv(
        baseline_rows,
        dataset,
        output_dir / "baseline_mean_pooling_test.csv",
    )
    del baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()

    temporal = build_model(
        config, names, args.base_checkpoint, args.temporal_checkpoint,
        device, logger
    )
    temporal_rows = predict(
        temporal, loader, dataset, names, device, "temporal_attention"
    )
    write_requested_csv(
        temporal_rows,
        dataset,
        output_dir / "temporal_attention_test.csv",
    )
    logger.info(
        "Wrote %d rows per experiment to %s",
        len(baseline_rows),
        output_dir,
    )


if __name__ == "__main__":
    main()
