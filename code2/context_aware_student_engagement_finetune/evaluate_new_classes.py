"""Evaluate only Sleeping and Talking videos and write predictions to CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ViFiCLIP.datasets.build import VideoDataset, mmcv_collate
from ViFiCLIP.trainers.vificlip import returnCLIP
from ViFiCLIP.utils.config import get_config
from ViFiCLIP.utils.logger import create_logger
from finetune import (
    load_class_names,
    load_model_checkpoint,
    select_device,
    set_seed,
)
from lora import apply_lora, load_lora_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="pretrained_action_recog_models/"
        "16_32_vifi_clip_16_shot_cvpr_n_actions.yaml",
    )
    parser.add_argument(
        "--base-checkpoint",
        default="pretrained_action_recog_models/ckpt_epoch_30.pth",
    )
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", default="outputs/new_classes_predictions.csv")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def make_config(args: argparse.Namespace):
    class ConfigArgs:
        config = args.config
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
    project_root = Path(__file__).resolve().parent
    config.DATA.ROOT = str(project_root)
    config.DATA.LABEL_LIST = str(project_root / config.DATA.LABEL_LIST)
    config.OUTPUT = str(project_root / config.OUTPUT)
    config.freeze()
    return config, project_root


def build_test_loader(config, project_root: Path):
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
        ann_file=str(project_root / "annotations/test.txt"),
        data_prefix=str(project_root),
        labels_file=str(project_root / "action_labels.csv"),
        pipeline=pipeline,
        test_mode=True,
    )
    return dataset, DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        collate_fn=lambda batch: mmcv_collate(batch, samples_per_gpu=1),
    )


def main() -> None:
    args = parse_args()
    config, project_root = make_config(args)
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = create_logger(str(output_path.parent), 0, "evaluate_new_classes")
    set_seed(config.SEED)
    device = select_device(args.device)

    class_names = load_class_names(config.DATA.LABEL_LIST)
    if len(class_names) != config.DATA.NUM_CLASSES:
        raise ValueError("The label file and NUM_CLASSES setting do not match")
    class_ids = {"Sleeping": 13, "Talking": 14}
    dataset, loader = build_test_loader(config, project_root)
    model = returnCLIP(config, logger, class_names)
    load_model_checkpoint(model, args.base_checkpoint, logger)
    adapter = torch.load(args.adapter, map_location="cpu", weights_only=False)
    if "lora" not in adapter:
        raise ValueError("The evaluation checkpoint does not contain LoRA adapters")
    apply_lora(model)
    load_lora_state_dict(model, adapter["lora"])
    model.to(device)
    model.eval()

    rows = []
    correct = 0
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            images = batch["imgs"].to(device, non_blocking=True)
            true_id = int(batch["label"].view(-1)[0].item())
            if true_id not in class_ids.values():
                continue
            logits = model(images)[0]
            binary_logits = logits[[class_ids["Sleeping"], class_ids["Talking"]]]
            probabilities = torch.softmax(binary_logits, dim=0)
            binary_index = int(probabilities.argmax().item())
            predicted_id = (13, 14)[binary_index]
            is_correct = predicted_id == true_id
            correct += int(is_correct)
            rows.append({
                "filename": Path(dataset.video_infos[index]["filename"]).name,
                "true_label": class_names[true_id],
                "predicted_label": class_names[predicted_id],
                "confidence": f"{probabilities[binary_index].item():.6f}",
                "correct": str(is_correct).lower(),
            })

    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Evaluated %d videos; binary accuracy=%.4f; CSV=%s",
                len(rows), correct / len(rows), output_path)


if __name__ == "__main__":
    main()
