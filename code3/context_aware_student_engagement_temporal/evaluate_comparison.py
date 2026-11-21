"""Compare the base and LoRA models on the complete held-out test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from ViFiCLIP.datasets.build import VideoDataset, mmcv_collate
from ViFiCLIP.trainers.vificlip import returnCLIP
from ViFiCLIP.utils.config import get_config
from ViFiCLIP.utils.logger import create_logger
from finetune import load_class_names, load_model_checkpoint, select_device, set_seed
from lora import apply_lora, load_lora_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint",
        default="pretrained_action_recog_models/ckpt_epoch_30.pth",
    )
    parser.add_argument("--adapter", default="outputs/best_lora.pth")
    parser.add_argument("--output-dir", default="outputs/full_comparison")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def make_config(args: argparse.Namespace):
    class ConfigArgs:
        config = "pretrained_action_recog_models/16_32_vifi_clip_16_shot_cvpr_n_actions.yaml"
        opts = []
        batch_size = None
        pretrained = None
        resume = None
        accumulation_steps = None
        output = "outputs"
        only_test = False
        local_rank = 0

    config = get_config(ConfigArgs)
    config.defrost()
    root = Path(__file__).resolve().parent
    config.DATA.ROOT = str(root)
    config.DATA.LABEL_LIST = str(root / config.DATA.LABEL_LIST)
    config.OUTPUT = str(root / config.OUTPUT)
    config.freeze()
    return config, root


def build_test_loader(config, root: Path):
    pipeline = [
        {"type": "DecordInit"},
        {"type": "UniformSampleFrames", "clip_len": config.DATA.NUM_FRAMES,
         "num_clips": 1, "test_mode": True},
        {"type": "DecordDecode"},
        {"type": "Resize", "scale": (-1, 256)},
        {"type": "CenterCrop", "crop_size": config.DATA.INPUT_SIZE},
        {"type": "Normalize", "mean": [123.675, 116.28, 103.53],
         "std": [58.395, 57.12, 57.375], "to_bgr": False},
        {"type": "FormatShape", "input_format": "NCHW"},
        {"type": "Collect", "keys": ["imgs", "label"], "meta_keys": []},
        {"type": "ToTensor", "keys": ["imgs", "label"]},
    ]
    dataset = VideoDataset(
        ann_file=str(root / "annotations/test.txt"),
        data_prefix=str(root),
        labels_file=str(root / "action_labels.csv"),
        pipeline=pipeline,
        test_mode=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        collate_fn=lambda batch: mmcv_collate(batch, samples_per_gpu=1),
    )
    return dataset, loader


def build_model(config, names, base_checkpoint, adapter, device, logger):
    model = returnCLIP(config, logger, names)
    load_model_checkpoint(model, base_checkpoint, logger)
    adapter_mode = "mean_pooling"
    if adapter:
        adapter_data = torch.load(adapter, map_location="cpu", weights_only=False)
        adapter_mode = (
            "combined_lora_temporal"
            if adapter_data.get("lora")
            else "temporal_attention"
        )
        if adapter_data.get("temporal"):
            result = model.temporal_pool.load_state_dict(
                adapter_data["temporal"], strict=True)
            if result.missing_keys or result.unexpected_keys:
                raise ValueError(f"Temporal adapter mismatch: {result}")
        if adapter_data.get("lora"):
            apply_lora(model)
            load_lora_state_dict(model, adapter_data["lora"])
    else:
        model.use_temporal_attention = False
    model.to(device)
    model.eval()
    return model, adapter_mode


def predict(model, loader, dataset, names, device, model_name):
    rows = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            images = batch["imgs"].to(device, non_blocking=True)
            true_id = int(batch["label"].view(-1)[0].item())
            logits = model(images)[0]
            probabilities = torch.softmax(logits, dim=0)
            predicted_id = int(logits.argmax().item())
            rows.append({
                "filename": Path(dataset.video_infos[index]["filename"]).name,
                "true_label": names[true_id],
                "predicted_label": names[predicted_id],
                "true_id": true_id,
                "predicted_id": predicted_id,
                "confidence": f"{probabilities[predicted_id].item():.6f}",
                "model": model_name,
            })
    return rows


def write_metrics(rows, names, output_dir, model_name):
    labels = sorted({int(row["true_id"]) for row in rows})
    y_true = [int(row["true_id"]) for row in rows]
    y_pred = [int(row["predicted_id"]) for row in rows]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = []
    for index, label_id in enumerate(labels):
        per_class.append({
            "label_id": label_id,
            "label": names[label_id],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        })
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    summary = {
        "model": model_name,
        "videos": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "labels_evaluated": [names[label_id] for label_id in labels],
    }
    with (output_dir / f"{model_name}_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / f"{model_name}_per_class.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=per_class[0].keys())
        writer.writeheader()
        writer.writerows(per_class)
    with (output_dir / f"{model_name}_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true\\pred"] + [names[label_id] for label_id in labels])
        writer.writerows([[names[label_id], *row] for label_id, row in zip(labels, matrix)])
    (output_dir / f"{model_name}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    config, root = make_config(args)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(str(output_dir), 0, "evaluate_comparison")
    set_seed(config.SEED)
    device = select_device(args.device)
    names = load_class_names(config.DATA.LABEL_LIST)
    dataset, loader = build_test_loader(config, root)
    base, base_mode = build_model(
        config, names, args.base_checkpoint, None, device, logger
    )
    base_rows = predict(base, loader, dataset, names, device, base_mode)
    base_summary = write_metrics(base_rows, names, output_dir, base_mode)
    del base
    adapted, adapted_mode = build_model(
        config, names, args.base_checkpoint, args.adapter, device, logger
    )
    adapted_rows = predict(adapted, loader, dataset, names, device, adapted_mode)
    adapted_summary = write_metrics(
        adapted_rows, names, output_dir, adapted_mode
    )
    comparison = {
        "base": base_summary,
        adapted_mode: adapted_summary,
        "accuracy_change": adapted_summary["accuracy"] - base_summary["accuracy"],
        "macro_f1_change": adapted_summary["macro_f1"] - base_summary["macro_f1"],
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Comparison: %s", json.dumps(comparison))


if __name__ == "__main__":
    main()
