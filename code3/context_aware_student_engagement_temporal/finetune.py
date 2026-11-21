"""Single-process ViFiCLIP fine-tuning and evaluation entry point."""

from __future__ import annotations

import argparse
import csv
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ViFiCLIP.datasets.build import build_dataloader
from ViFiCLIP.trainers.vificlip import returnCLIP
from ViFiCLIP.utils.config import get_config
from ViFiCLIP.utils.logger import create_logger
from ViFiCLIP.utils.optimizer import build_optimizer, build_scheduler
from lora import apply_lora, lora_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="pretrained_action_recog_models/"
        "16_32_vifi_clip_16_shot_cvpr_n_actions.yaml",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", choices=("temporal", "lora"), default="temporal")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def make_config(args: argparse.Namespace):
    class ConfigArgs:
        config = args.config
        opts = args.opts
        batch_size = None
        pretrained = None
        resume = args.resume
        accumulation_steps = None
        output = args.output
        only_test = False
        local_rank = 0

    config = get_config(ConfigArgs)
    config.defrost()
    project_root = Path(__file__).resolve().parent
    config.DATA.ROOT = str(project_root)
    config.DATA.TRAIN_FILE = str(project_root / config.DATA.TRAIN_FILE)
    config.DATA.VAL_FILE = str(project_root / config.DATA.VAL_FILE)
    config.DATA.LABEL_LIST = str(project_root / config.DATA.LABEL_LIST)
    config.OUTPUT = str(project_root / config.OUTPUT)
    if args.epochs is not None:
        config.TRAIN.EPOCHS = args.epochs
    config.freeze()
    return config


def load_class_names(path: str) -> list[str]:
    with open(path, newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        names = [row["name"] for row in rows]
    if not names:
        raise ValueError(f"No class names found in {path}")
    return names


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["imgs"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).long().view(-1)
            logits = model(images)
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
    if total == 0:
        raise RuntimeError("Validation loader returned no samples")
    return total_loss / total, correct / total


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str, logger) -> None:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint has no model state dictionary: {path}")
    cleaned = {}
    for name, value in state_dict.items():
        if name.startswith("module."):
            name = name[7:]
        if name in {
            "prompt_learner.token_prefix",
            "prompt_learner.token_suffix",
            "prompt_learner.complete_text_embeddings",
        }:
            continue
        cleaned[name] = value
    result = model.load_state_dict(cleaned, strict=False)
    logger.info(
        "Loaded checkpoint %s (missing=%d unexpected=%d)",
        path,
        len(result.missing_keys),
        len(result.unexpected_keys),
    )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    config = make_config(args)
    output_dir = Path(config.OUTPUT)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(str(output_dir), 0, "finetune")

    set_seed(config.SEED)
    device = select_device(args.device)
    logger.info("Using device: %s", device)

    init_file = tempfile.NamedTemporaryFile(prefix="vificlip-dist-", delete=False)
    init_file.close()
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file.name}",
        rank=0,
        world_size=1,
    )
    try:
        class_names = load_class_names(config.DATA.LABEL_LIST)
        if len(class_names) != config.DATA.NUM_CLASSES:
            raise ValueError(
                f"NUM_CLASSES={config.DATA.NUM_CLASSES} but "
                f"{len(class_names)} labels were found"
            )
        train_data, val_data, train_loader, val_loader = build_dataloader(
            logger, config
        )
        logger.info("Loaded %d training and %d validation videos",
                    len(train_data), len(val_data))
        model = returnCLIP(config, logger, class_names)
        if args.resume:
            load_model_checkpoint(model, args.resume, logger)
        if args.mode == "lora":
            wrapped = apply_lora(model, args.lora_rank, args.lora_alpha)
            for parameter in model.temporal_pool.parameters():
                parameter.requires_grad_(True)
            logger.info(
                "Enabled LoRA adapters in %d visual layers and temporal pooling",
                wrapped,
            )
        else:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for parameter in model.temporal_pool.parameters():
                parameter.requires_grad_(True)
            logger.info("Training temporal pooling parameters only")
        model.to(device)

        if args.dry_run:
            batch = next(iter(train_loader))
            images, labels = batch["imgs"], batch["label"]
            images = images.to(device)
            labels = labels.to(device).long().view(-1)
            with torch.no_grad():
                logits = model(images)
            logger.info("Dry run passed: images=%s logits=%s labels=%s",
                        tuple(images.shape), tuple(logits.shape), tuple(labels.shape))
            return

        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer, max(1, len(train_loader)))
        lora_b_parameters = [
            parameter for name, parameter in model.named_parameters()
            if ".lora_b" in name and parameter.requires_grad
        ]
        if args.mode == "lora" and not lora_b_parameters:
            raise RuntimeError("No trainable LoRA B parameters were found")
        optimizer.zero_grad(set_to_none=True)
        checked_lora_update = args.mode != "lora"
        best_accuracy = 0.0
        for epoch in range(config.TRAIN.EPOCHS):
            model.train()
            train_loader.sampler.set_epoch(epoch)
            running_loss = 0.0
            for step, batch in enumerate(train_loader):
                images = batch["imgs"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True).long().view(-1)
                loss = F.cross_entropy(model(images), labels)
                loss.backward()
                should_update = (
                    (step + 1) % config.TRAIN.ACCUMULATION_STEPS == 0
                    or step + 1 == len(train_loader)
                )
                if should_update:
                    gradient_norms = [
                        parameter.grad.detach().norm()
                        for parameter in lora_b_parameters
                        if parameter.grad is not None
                    ]
                    effective_lr = max(
                        group["lr"] for group in optimizer.param_groups
                    )
                    if not checked_lora_update and effective_lr > 0 and (
                        not gradient_norms
                        or not any(norm.item() > 0 for norm in gradient_norms)
                    ):
                        raise RuntimeError(
                            "LoRA B parameters received no gradient on the first "
                            "optimizer step"
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if not checked_lora_update and effective_lr > 0:
                        if not any(
                            torch.count_nonzero(parameter.detach()).item()
                            for parameter in lora_b_parameters
                        ):
                            raise RuntimeError(
                                "LoRA B parameters remained zero after the first "
                                "optimizer step"
                            )
                        logger.info(
                            "First LoRA update: B gradient norm=%.6g, "
                            "B weight norm=%.6g",
                            torch.stack(gradient_norms).norm().item(),
                            torch.stack(
                                [parameter.detach().norm()
                                 for parameter in lora_b_parameters]
                            ).norm().item(),
                        )
                        checked_lora_update = True
                scheduler.step_update(epoch * len(train_loader) + step)
                running_loss += loss.item()
            val_loss, val_accuracy = evaluate(model, val_loader, device)
            logger.info(
                "epoch=%d train_loss=%.4f val_loss=%.4f val_accuracy=%.4f",
                epoch + 1,
                running_loss / max(1, len(train_loader)),
                val_loss,
                val_accuracy,
            )
            if val_accuracy >= best_accuracy:
                best_accuracy = val_accuracy
                torch.save(
                    {
                        "temporal": {
                            name: parameter.detach().cpu()
                            for name, parameter in model.temporal_pool.state_dict().items()
                        },
                        "lora": lora_state_dict(model) if args.mode == "lora" else {},
                        "epoch": epoch,
                        "mode": args.mode,
                     "val_accuracy": val_accuracy},
                    output_dir / (
                        "best_temporal.pth"
                        if args.mode == "temporal"
                        else "best_combined.pth"
                    ),
                )
    finally:
        dist.destroy_process_group()
        try:
            os.unlink(init_file.name)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
