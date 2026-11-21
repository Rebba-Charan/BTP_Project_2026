"""Classify every video in an action-folder dataset with ViFiCLIP."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd


VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


def video_files(dataset_root: Path) -> list[Path]:
    return sorted(
        path for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def read_video(path: Path) -> tuple[list, float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    try:
        for _ in range(frame_count):
            success, frame = capture.read()
            if not success:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise OSError(f"Video contains no readable frames: {path}")
    return frames, fps, frame_count


def classify_dataset(dataset_root: Path, output_dir: Path) -> None:
    from actionClassification import ActionClassifier

    labels_path = Path("action_labels.csv")
    labels = pd.read_csv(labels_path, usecols=["id", "name"])
    action_names = labels["name"].tolist()
    classifier = ActionClassifier(action_names)
    files = video_files(dataset_root)
    if not files:
        raise FileNotFoundError(f"No supported videos found under {dataset_root}")

    fields = [
        "filename",
        "source_folder",
        "num_frames",
        "fps",
        "predicted_label",
        "status",
        "error",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    handles = {}
    try:
        for index, path in enumerate(files, start=1):
            folder = path.parent.name
            if folder not in writers:
                folder_output = output_dir / f"{folder}_predictions.csv"
                handle = folder_output.open(
                    "w", newline="", encoding="utf-8"
                )
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                handles[folder] = handle
                writers[folder] = writer
            row = {
                "filename": path.name,
                "source_folder": path.parent.name,
                "num_frames": "",
                "fps": "",
                "predicted_label": "",
                "status": "error",
                "error": "",
            }
            try:
                frames, fps, frame_count = read_video(path)
                sampled_indices = classifier.uniformSampling(len(frames))
                sampled_frames = [frames[i] for i in sampled_indices]
                row.update({
                    "num_frames": frame_count,
                    "fps": round(fps, 3) if fps > 0 else "",
                    "predicted_label": classifier.classify(sampled_frames),
                    "status": "ok",
                })
                print(
                    f"[{index}/{len(files)}] {path.name}: "
                    f"{row['predicted_label']}"
                )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{index}/{len(files)}] {path}: {row['error']}")
            writers[folder].writerow(row)
            handles[folder].flush()
    finally:
        for handle in handles.values():
            handle.close()
    print(f"Saved {len(files)} results in {len(writers)} folder CSV files to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--output-dir", default="batch_predictions", type=Path,
        help="Directory receiving one CSV per source folder."
    )
    args = parser.parse_args()
    if not args.dataset.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {args.dataset}")
    classify_dataset(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
