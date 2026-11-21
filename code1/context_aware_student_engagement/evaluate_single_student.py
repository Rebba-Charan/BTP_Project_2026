"""Evaluate single-student action predictions against manually annotated segments."""

import argparse
import csv
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(
            row for row in stream if not row.lstrip().startswith("#")
        ))


def interval_key(row):
    return (
        round(float(row["start_time"]), 3),
        round(float(row["end_time"]), 3),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="predictions_single_student.csv")
    parser.add_argument("--annotations", default="annotations_single_student.csv")
    parser.add_argument("--output", default="accuracy_matrix_single_student.csv")
    args = parser.parse_args()

    predictions = {
        interval_key(row): row["predicted_label"]
        for row in read_csv(Path(args.predictions))
    }
    annotations = read_csv(Path(args.annotations))
    y_true = []
    y_pred = []
    rows = []
    for annotation in annotations:
        key = interval_key(annotation)
        if key not in predictions:
            raise ValueError(
                f"No prediction matches interval "
                f"{annotation['start_time']}-{annotation['end_time']}."
            )
        true_label = annotation["true_label"].strip()
        predicted_label = predictions[key]
        y_true.append(true_label)
        y_pred.append(predicted_label)
        rows.append({
            "start_time": annotation["start_time"],
            "end_time": annotation["end_time"],
            "true_label": true_label,
            "predicted_label": predicted_label,
            "correct": str(true_label == predicted_label),
        })

    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    with Path(args.output).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["confusion_matrix"] + labels)
        for label, values in zip(labels, matrix):
            writer.writerow([label] + values.tolist())
        writer.writerow([])
        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy", accuracy_score(y_true, y_pred)])
        report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        for label in labels + ["macro avg", "weighted avg"]:
            writer.writerow([f"{label}_precision", report[label]["precision"]])
            writer.writerow([f"{label}_recall", report[label]["recall"]])
            writer.writerow([f"{label}_f1", report[label]["f1-score"]])
        writer.writerow([])
        writer.writerow(["segment_results"])
        writer.writerow(rows[0].keys())
        writer.writerows(row.values() for row in rows)

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Evaluated {len(rows)} annotated segments")
    print(f"Saved accuracy matrix to {args.output}")


if __name__ == "__main__":
    main()
