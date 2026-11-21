"""Classify classroom videos into overall engagement labels per two-minute interval."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import pandas as pd

from actionClassification import ActionClassifier
from detectionTracking import Tracking
from engagementClassification import EngagementClassifier
from engageConfig import labels_file
import utils


ACTION_WINDOW_SECONDS = 3
ENGAGEMENT_WINDOW_SECONDS = 120


def parse_engagement_response(response: object) -> tuple[str, str]:
    """Extract the category and reasoning from an LLM response."""
    if isinstance(response, dict):
        return str(response.get("Category", "unknown")), str(response.get("reasoning", ""))
    text = str(response or "").strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return str(parsed.get("Category", "unknown")), str(parsed.get("reasoning", ""))
    except (SyntaxError, ValueError):
        pass
    category_match = re.search(r"""['"]Category['"]\s*:\s*['"]([^'"]+)""", text, re.IGNORECASE)
    category = category_match.group(1) if category_match else text or "unknown"
    return category, text


def majority_action_summary(
    action_windows: list[tuple[float, float, str, int]],
    interval_start: float,
    interval_end: float,
    bin_seconds: int = 10,
) -> str:
    """Summarize tracked-student actions using one majority action per time bin."""
    summaries: list[str] = []
    current = interval_start
    while current < interval_end:
        bin_end = min(current + bin_seconds, interval_end)
        actions = [
            action
            for start, end, action, _track_id in action_windows
            if start < bin_end and end > current
        ]
        if actions:
            action = Counter(actions).most_common(1)[0][0]
            start_text = f"{int(current // 60):02d}:{int(current % 60):02d}"
            end_text = f"{int(bin_end // 60):02d}:{int(bin_end % 60):02d}"
            summaries.append(f"{action}({start_text}-{end_text})")
        current = bin_end
    return "; ".join(summaries)


def classify_video(
    video_path: Path,
    action_classifier: ActionClassifier,
    engagement_classifier: EngagementClassifier,
) -> list[dict[str, object]]:
    """Run tracking and classification for one video."""
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = video.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        video.release()
        raise RuntimeError(f"Could not read FPS from video: {video_path}")

    action_window_frames = max(1, round(fps * ACTION_WINDOW_SECONDS))
    engagement_window_frames = max(1, round(fps * ENGAGEMENT_WINDOW_SECONDS))
    tracking = Tracking(objectDetector="fastercnn", tracker="ByteTrack")
    track_boxes: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    frames: list[object] = []
    action_windows: list[tuple[float, float, str, int]] = []
    results: list[dict[str, object]] = []
    frame_number = 0
    interval_start_frame = 0

    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            frame_number += 1
            frames.append(frame)
            detections = tracking.detect(frame)
            for track_id, box, _state in tracking.track(detections, frame):
                track_boxes[track_id].append(box)

            if frame_number % action_window_frames == 0:
                window_start = frame_number - action_window_frames
                window_start_seconds = window_start / fps
                window_end_seconds = frame_number / fps
                for track_id, boxes in list(track_boxes.items()):
                    if len(boxes) < max(2, action_window_frames // 4):
                        continue
                    min_x, min_y, max_x, max_y = utils.getEnclosingBBox(boxes)
                    if max_x - min_x < 100 or max_y - min_y < 100:
                        continue
                    min_x, min_y, max_x, max_y = utils.checkBoundary(
                        (min_x, min_y, max_x, max_y), frame.shape[:-1]
                    )
                    cropped_frames = [
                        source[min_y:max_y, min_x:max_x] for source in frames
                    ]
                    if not cropped_frames or any(
                        crop.size == 0 for crop in cropped_frames
                    ):
                        continue
                    sampled = action_classifier.uniformSampling(len(cropped_frames))
                    action = action_classifier.classify(
                        [cropped_frames[index] for index in sampled]
                    )
                    action_windows.append(
                        (window_start_seconds, window_end_seconds, action, track_id)
                    )

            if frame_number % engagement_window_frames == 0:
                results.append(
                    classify_interval(
                        video_path,
                        interval_start_frame / fps,
                        frame_number / fps,
                        action_windows,
                        engagement_classifier,
                    )
                )
                interval_start_frame = frame_number
                action_windows.clear()
                frames.clear()
                track_boxes.clear()

        if frame_number > interval_start_frame:
            results.append(
                classify_interval(
                    video_path,
                    interval_start_frame / fps,
                    frame_number / fps,
                    action_windows,
                    engagement_classifier,
                )
            )
    finally:
        video.release()
    return results


def classify_interval(
    video_path: Path,
    start_seconds: float,
    end_seconds: float,
    action_windows: list[tuple[float, float, str, int]],
    engagement_classifier: EngagementClassifier,
) -> dict[str, object]:
    summary = majority_action_summary(
        action_windows, start_seconds, end_seconds
    )
    if summary:
        response = engagement_classifier.annotate(summary)
        category, reasoning = parse_engagement_response(response)
    else:
        category, reasoning = "unknown", "No usable tracked-student action windows."
    return {
        "video": video_path.name,
        "start_time": round(start_seconds, 3),
        "end_time": round(end_seconds, 3),
        "predicted_label": category.lower(),
        "reasoning": reasoning,
        "students_detected": len({track_id for *_, track_id in action_windows}),
        "action_summary": summary,
    }


def write_results(results: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video",
        "start_time",
        "end_time",
        "predicted_label",
        "reasoning",
        "students_detected",
        "action_summary",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def output_stem(video_path: Path) -> str:
    """Create a portable output filename from a source video name."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", video_path.stem).strip("._")
    return stem[:160] or "video"


def write_failures(
    failures: list[dict[str, str]], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["video", "error"])
        writer.writeheader()
        writer.writerows(failures)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/overall_2min_predictions.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/overall_2min_by_video"),
        help="Directory for one CSV per source video and failures.csv.",
    )
    parser.add_argument("--llm", choices=("ollama", "llama", "gpt"), default="ollama")
    args = parser.parse_args()

    if not args.video_dir.is_dir():
        raise NotADirectoryError(args.video_dir)
    video_paths = sorted(args.video_dir.glob("*.mp4"))
    if not video_paths:
        raise FileNotFoundError(f"No MP4 files found in {args.video_dir}")

    classes = pd.read_csv(labels_file, usecols=["id", "name"])
    action_names = [name for _id, name in classes.values.tolist()]
    action_classifier = ActionClassifier(action_names)
    engagement_classifier = EngagementClassifier(
        action_names, usedLLM=args.llm, contextAware=False
    )

    all_results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for video_path in video_paths:
        print(f"Processing {video_path}")
        try:
            video_results = classify_video(
                video_path, action_classifier, engagement_classifier
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"Skipping {video_path}: {error}")
            failures.append({"video": video_path.name, "error": error})
            continue
        all_results.extend(video_results)
        write_results(
            video_results,
            args.output_dir / f"{output_stem(video_path)}_2min_predictions.csv",
        )
    write_results(all_results, args.output)
    write_failures(failures, args.output_dir / "failures.csv")
    print(f"Saved {len(all_results)} interval predictions to {args.output}")
    print(json.dumps(all_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
