import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from final_evaluate import load_test_predictions
from sklearn.datasets import load_digits

from analyze_errors import summarize_predictions


def check_final():
    with TemporaryDirectory() as temporary:
        artifacts = Path(temporary)
        selection = {"model_parameters": {"n_neighbors": 1}}
        protocol_path = artifacts / "final-protocol.json"
        protocol_path.write_text(json.dumps({"selection": selection}))
        archive_path = artifacts / "test-predictions.npz"
        np.savez(archive_path, test_indices=np.array([20, 10]), knn_predictions=np.array([7, 3]))
        results = {"protocol_sha256": sha256(protocol_path.read_bytes()).hexdigest(),
                   "predictions_sha256": sha256(archive_path.read_bytes()).hexdigest()}
        (artifacts / "final-test-results.json").write_text(json.dumps(results))
        splits = {"test": np.array([20, 10])}
        np.testing.assert_array_equal(load_test_predictions(artifacts, selection, splits), [7, 3])
        for other_selection, other_splits in [({"model_parameters": {"n_neighbors": 3}}, splits),
                                               (selection, {"test": np.array([10, 20])})]:
            try:
                load_test_predictions(artifacts, other_selection, other_splits)
            except ValueError:
                pass
            else:
                raise AssertionError("Mismatched test procedure or partition was accepted")
        archive_path.write_bytes(archive_path.read_bytes() + b"changed")
        try:
            load_test_predictions(artifacts, selection, splits)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed prediction archive was accepted")
    print("Final-result checks passed: recorded predictions, procedure identity, partition order, and archive integrity.")

    artifacts = Path(__file__).resolve().parent / "artifacts"
    results = json.loads((artifacts / "final-test-results.json").read_text())
    report = json.loads((artifacts / "test-error-report.json").read_text())
    selection = json.loads((artifacts / "selected-model.json").read_text())
    digits = load_digits()
    np.testing.assert_array_equal(digits.images.reshape(len(digits.target), 64), digits.data)
    with np.load(artifacts / "splits.npz", allow_pickle=False) as saved:
        splits = {name: saved[name] for name in saved.files}
    predictions = load_test_predictions(artifacts, selection, splits)
    with np.load(artifacts / "test-predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["labels"], digits.target[splits["test"]])
        baseline_correct = int(np.count_nonzero(saved["baseline_predictions"] == saved["labels"]))
    assert baseline_correct == results["baseline"]["correct"]
    summary = summarize_predictions(digits.target[splits["test"]], predictions)
    np.testing.assert_array_equal(summary["matrix"], results["confusion_matrix"])
    np.testing.assert_array_equal(summary["support"], results["class_support"])
    np.testing.assert_allclose(summary["recall"], results["per_class_recall"])
    assert int(summary["correct"].sum()) == results["knn"]["correct"] == report["correct"]
    assert len(summary["wrong_rows"]) == results["knn"]["errors"] == len(report["mistakes"])
    assert [case["query_dataset_index"] for case in report["mistakes"]] == splits["test"][summary["wrong_rows"]].tolist()
    for case in report["mistakes"] + [report["correct_comparison"]]:
        assert case["query_dataset_index"] == splits["test"][case["test_row"]]
        assert case["prediction"] == predictions[case["test_row"]]
        for neighbor in case["neighbors"]:
            assert neighbor["dataset_index"] == splits["train"][neighbor["training_row"]]
            assert neighbor["label"] == digits.target[neighbor["dataset_index"]]
    explanation = (artifacts.parent / "report.md").read_text()
    for model in ("baseline", "knn"):
        assert f"{results[model]['correct']}/{results['total']}" in explanation
        assert f"{results[model]['accuracy']:.2%}" in explanation
    print("Saved-result checks passed: image/vector mapping, labels, matrix, recall, all gallery IDs, and report totals.")


if __name__ == "__main__":
    from check_splits import check_splits
    from check_baseline import check_baseline
    from check_knn import check_knn
    from check_errors import check_errors
    from check_learning_curve import check_learning_curve
    from check_robustness import check_robustness
    from check_viewer import check_viewer
    from neighbor_math import demo

    for check in (check_splits, check_baseline, demo, check_knn, check_errors,
                  check_learning_curve, check_robustness, check_viewer):
        check()
    check_final()
