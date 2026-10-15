import csv
import json
from hashlib import sha256
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from experiment import load_or_create_splits, trace_knn_prediction


def summarize_predictions(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(10))
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    correct = np.diag(matrix)
    wrong_rows = np.flatnonzero(y_true != y_pred)
    assert matrix.sum() == len(y_true), "Confusion matrix must cover every query"
    assert correct.sum() == len(y_true) - len(wrong_rows), "Matrix and prediction errors disagree"
    normalized = np.divide(matrix, support[:, None], out=np.zeros_like(matrix, dtype=float), where=support[:, None] != 0)
    # No examples or no predictions means an undefined ratio, not zero performance.
    recall = np.divide(correct, support, out=np.full(10, np.nan), where=support != 0)
    precision = np.divide(correct, predicted_count, out=np.full(10, np.nan), where=predicted_count != 0)
    return {
        "matrix": matrix, "normalized": normalized, "support": support,
        "predicted_count": predicted_count, "correct": correct,
        "recall": recall, "precision": precision, "wrong_rows": wrong_rows,
    }


def load_selected_experiment(artifacts):
    selection = json.loads((artifacts / "selected-model.json").read_text())
    if selection["representation"] != "raw_pixels" or selection["final_fit_policy"] != "original_training_partition_only":
        raise ValueError("This analysis requires the recorded raw-pixel, training-only procedure")
    if sha256((artifacts / "splits.npz").read_bytes()).hexdigest() != selection["split_file_sha256"]:
        raise ValueError("Saved partition indices differ from the selected experiment")
    digits = load_digits()
    X, y = digits.data, digits.target
    splits, metadata = load_or_create_splits(X, y, artifacts)
    if metadata["dataset_digest"] != selection["dataset_digest"]:
        raise ValueError("The selected model and partitions refer to different data")
    training, validation = splits["train"], splits["validation"]
    X_train, y_train = X[training], y[training]
    X_validation, y_validation = X[validation], y[validation]
    model = KNeighborsClassifier(**selection["model_parameters"])
    model.fit(X_train, y_train)
    predictions = model.predict(X_validation)
    correct = int(np.count_nonzero(predictions == y_validation))
    if correct != selection["validation"]["correct"] or len(validation) != selection["validation"]["total"]:
        raise ValueError("Validation results no longer reproduce the recorded selection")
    return digits, splits, selection, model, predictions


def save_error_analysis(digits, splits, selection, model, predictions, artifacts, partition):
    if partition not in ("validation", "test"):
        raise ValueError("Error analysis requires validation or test predictions")
    training, evaluation = splits["train"], splits[partition]
    X_train, y_train = digits.data[training], digits.target[training]
    X_evaluation, y_evaluation = digits.data[evaluation], digits.target[evaluation]
    summary = summarize_predictions(y_evaluation, predictions)
    correct = int(summary["correct"].sum())

    print(f"Selected k={model.n_neighbors}: {partition} {correct}/{len(evaluation)}, {len(summary['wrong_rows'])} mistakes.")
    print("Rows are true labels; columns are predicted labels.")
    print(summary["matrix"])
    print("\nLabel  True  Predicted  Correct  Missed  Recall   Precision")
    class_rows = []
    for label in range(10):
        recall, precision = summary["recall"][label], summary["precision"][label]
        recall_text = f"{recall:.2%}" if np.isfinite(recall) else "n/a"
        precision_text = f"{precision:.2%}" if np.isfinite(precision) else "n/a"
        row = {
            "label": label, "true_count": int(summary["support"][label]),
            "predicted_count": int(summary["predicted_count"][label]),
            "correct": int(summary["correct"][label]),
            "missed": int(summary["support"][label] - summary["correct"][label]),
            "recall": float(recall) if np.isfinite(recall) else "",
            "precision": float(precision) if np.isfinite(precision) else "",
        }
        class_rows.append(row)
        print(f"{label:>5}  {row['true_count']:>4}  {row['predicted_count']:>9}  {row['correct']:>7}  {row['missed']:>6}  {recall_text:>7}  {precision_text:>9}")
    with (artifacts / f"{partition}-per-class.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=class_rows[0].keys())
        writer.writeheader()
        writer.writerows(class_rows)

    for matrix, title, number_format, suffix in [
        (summary["matrix"], f"{partition.capitalize()} counts", "d", "counts"),
        (summary["normalized"] * 100, f"{partition.capitalize()} row percentages (%)", ".1f", "percent"),
    ]:
        figure, ax = plt.subplots(figsize=(6, 5.2), layout="constrained")
        ConfusionMatrixDisplay(matrix, display_labels=np.arange(10)).plot(
            ax=ax, cmap="Blues", values_format=number_format, colorbar=False
        )
        ax.set_title(f"{title} | selected k={model.n_neighbors}")
        ax.set_xlabel("Predicted digit")
        ax.set_ylabel("True digit")
        figure.savefig(artifacts / f"{partition}-confusion-{suffix}.png", dpi=160)
        plt.close(figure)

    wrong_rows = summary["wrong_rows"].tolist()
    displayed_rows = wrong_rows.copy()
    correct_rows = np.flatnonzero(predictions == y_evaluation)
    if len(correct_rows):
        displayed_rows.append(int(correct_rows[0]))
    cases = []
    for row in displayed_rows:
        case = trace_knn_prediction(model, X_evaluation[row], X_train, y_train, training)
        assert case["prediction"] == predictions[row], "Trace and batch prediction disagree"
        assert all(neighbor["dataset_index"] in training for neighbor in case["neighbors"])
        case.update({
            f"{partition}_row": row, "query_dataset_index": int(evaluation[row]),
            "true_label": int(y_evaluation[row]), "correct": bool(predictions[row] == y_evaluation[row]),
        })
        cases.append(case)
        if not case["correct"]:
            print(f"\nMistake: sample {case['query_dataset_index']}, true {case['true_label']}, predicted {case['prediction']}")
            print(f"Voting neighbors: {case['neighbors']}; votes: {case['votes']}")
    assert sum(not case["correct"] for case in cases) == len(wrong_rows), "Gallery must include every mistake"

    columns = model.n_neighbors + 1
    gallery, axes = plt.subplots(len(cases), columns, figsize=(3 * columns, 2.2 * len(cases)), squeeze=False, layout="constrained")
    gallery.suptitle(f"All {len(wrong_rows)} {partition} mistakes and one correct comparison | k={model.n_neighbors}", fontsize=13)
    for row, case in enumerate(cases):
        status = "Correct comparison" if case["correct"] else "Mistake"
        axes[row, 0].imshow(digits.images[case["query_dataset_index"]], cmap="gray", vmin=0, vmax=16, interpolation="nearest")
        axes[row, 0].set_title(f"{status} | sample {case['query_dataset_index']}\nTrue {case['true_label']} | predicted {case['prediction']}\nVotes: {case['votes']}", fontsize=10)
        for column, neighbor in enumerate(case["neighbors"], start=1):
            axes[row, column].imshow(digits.images[neighbor["dataset_index"]], cmap="gray", vmin=0, vmax=16, interpolation="nearest")
            axes[row, column].set_title(f"Voting training sample {neighbor['dataset_index']}\nLabel {neighbor['label']} | distance {neighbor['distance']:.2f}", fontsize=10)
    for ax in axes.flat:
        ax.set_axis_off()
    gallery.savefig(artifacts / f"{partition}-failure-gallery.png", dpi=160)
    plt.close(gallery)

    report = {
        "partition": partition, "selection": selection, "total": len(evaluation), "correct": correct,
        "label_order": list(range(10)), "confusion_matrix": summary["matrix"].tolist(),
        "row_percentages": (summary["normalized"] * 100).tolist(),
        "mistakes": [case for case in cases if not case["correct"]],
        "correct_comparison": next((case for case in cases if case["correct"]), None),
    }
    (artifacts / f"{partition}-error-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"\nSaved confusion plots, per-class results, all mistake traces, and gallery in: {artifacts}")
    return report


def main():
    artifacts = Path(__file__).resolve().parent / "artifacts"
    digits, splits, selection, model, predictions = load_selected_experiment(artifacts)
    save_error_analysis(digits, splits, selection, model, predictions, artifacts, "validation")
    print("Allocation, selection, and comparison records were not rewritten. Test examples were not evaluated.")


if __name__ == "__main__":
    main()
