import json
import sys
from hashlib import sha256
from pathlib import Path

import matplotlib
import numpy as np
import sklearn

from analyze_errors import load_selected_experiment, save_error_analysis, summarize_predictions
from experiment import SEED, evaluate_majority_baseline, freeze_selection


def load_test_predictions(artifacts, selection, splits):
    protocol_path = artifacts / "final-protocol.json"
    protocol = json.loads(protocol_path.read_text())
    results = json.loads((artifacts / "final-test-results.json").read_text())
    archive_path = artifacts / "test-predictions.npz"
    if protocol["selection"] != selection:
        raise ValueError("Test results belong to a different selected procedure")
    if results["protocol_sha256"] != sha256(protocol_path.read_bytes()).hexdigest():
        raise ValueError("Final protocol differs from the evaluated procedure")
    if results["predictions_sha256"] != sha256(archive_path.read_bytes()).hexdigest():
        raise ValueError("Saved test predictions differ from the evaluated results")
    with np.load(archive_path, allow_pickle=False) as saved:
        if not np.array_equal(saved["test_indices"], splits["test"]):
            raise ValueError("Test predictions refer to different partition indices")
        predictions = saved["knn_predictions"]
        if predictions.shape != splits["test"].shape or not np.isin(predictions, np.arange(10)).all():
            raise ValueError("Invalid saved test prediction array")
    return predictions


def main():
    root = Path(__file__).resolve().parent
    artifacts = root / "artifacts"
    digits, splits, selection, model, _ = load_selected_experiment(artifacts)
    training, test = splits["train"], splits["test"]
    X_train, y_train = digits.data[training], digits.target[training]
    labels, counts = np.unique(y_train, return_counts=True)
    baseline_label = int(labels[counts.argmax()])
    protocol = {
        "selection": selection, "seed": SEED, "training_count": len(training),
        "training_indices_sha256": sha256(training.tobytes()).hexdigest(),
        "test_count": len(test), "baseline": {"strategy": "most_frequent", "label": baseline_label,
                                                "tie_rule": "lowest label"},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "source_sha256": {name: sha256((root / name).read_bytes()).hexdigest()
                          for name in ("experiment.py", "neighbor_math.py", "analyze_errors.py", "final_evaluate.py", "requirements.txt")},
        "test_policy": "final evaluation of frozen procedure; no test-driven tuning or train-plus-validation refit",
    }
    protocol_path = artifacts / "final-protocol.json"
    freeze_selection(protocol_path, protocol)
    print("Final protocol saved/verified BEFORE test prediction. No settings are being compared.")

    X_test, y_test = digits.data[test], digits.target[test]
    predictions = model.predict(X_test)
    majority_label, baseline_correct, baseline_accuracy = evaluate_majority_baseline(X_train, y_train, X_test, y_test)
    assert majority_label == baseline_label
    summary = summarize_predictions(y_test, predictions)
    correct = int(summary["correct"].sum())
    archive = {"test_indices": test, "labels": y_test, "knn_predictions": predictions,
               "baseline_predictions": np.full(len(test), baseline_label, dtype=y_test.dtype)}
    archive_path = artifacts / "test-predictions.npz"
    if archive_path.exists():
        with np.load(archive_path, allow_pickle=False) as saved:
            for key, value in archive.items():
                np.testing.assert_array_equal(saved[key], value, err_msg="Frozen test predictions no longer reproduce")
    else:
        np.savez(archive_path, **archive)
    results = {
        "partition": "test", "protocol_sha256": sha256(protocol_path.read_bytes()).hexdigest(),
        "predictions_sha256": sha256(archive_path.read_bytes()).hexdigest(), "total": len(test),
        "baseline": {"label": baseline_label, "correct": baseline_correct,
                     "errors": len(test) - baseline_correct, "accuracy": baseline_accuracy},
        "knn": {"k": model.n_neighbors, "correct": correct,
                "errors": len(test) - correct, "accuracy": correct / len(test)},
        "label_order": list(range(10)), "class_support": summary["support"].tolist(),
        "confusion_matrix": summary["matrix"].tolist(), "per_class_recall": summary["recall"].tolist(),
    }
    freeze_selection(artifacts / "final-test-results.json", results)
    np.testing.assert_array_equal(load_test_predictions(artifacts, selection, splits), predictions)
    print(f"Baseline (always {baseline_label}): {baseline_correct}/{len(test)} = {baseline_accuracy:.2%}")
    print(f"Selected k={model.n_neighbors}: {correct}/{len(test)} = {correct / len(test):.2%}")
    print("Metrics and exact predictions are recorded. Now inspecting errors; no retuning follows.")
    save_error_analysis(digits, splits, selection, model, predictions, artifacts, "test")


if __name__ == "__main__":
    main()
