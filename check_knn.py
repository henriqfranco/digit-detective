import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from experiment import compare_knn_settings, freeze_selection, trace_knn_prediction


def check_knn():
    X_train = np.array([[1, 2, 4], [1, 4, 3], [4, 2, 3], [1, 2, 7]], dtype=float)
    y_train = np.array([7, 3, 3, 7])
    original_indices = np.array([100, 200, 300, 400])
    query = np.array([1, 2, 3], dtype=float)
    model = KNeighborsClassifier(n_neighbors=3, metric="euclidean", weights="uniform", algorithm="brute")
    model.fit(X_train, y_train)
    trace = trace_knn_prediction(model, query, X_train, y_train, original_indices)
    assert trace["prediction"] == 3 and trace["votes"] == {3: 2, 7: 1}
    assert [neighbor["training_row"] for neighbor in trace["neighbors"]] == [0, 1, 2]
    assert [neighbor["dataset_index"] for neighbor in trace["neighbors"]] == [100, 200, 300]
    assert [neighbor["label"] for neighbor in trace["neighbors"]] == [7, 3, 3]
    np.testing.assert_allclose([neighbor["distance"] for neighbor in trace["neighbors"]], [1, 2, 3])

    # Explain a tied vote using the same convention as the actual classifier.
    tied_labels = np.array([7, 3, 9, 7])
    model.fit(X_train, tied_labels)
    trace = trace_knn_prediction(model, query, X_train, tied_labels, original_indices)
    assert trace["prediction"] == 3 and trace["votes"] == {3: 1, 7: 1, 9: 1}

    comparison_X = np.arange(9, dtype=float).reshape(9, 1)
    comparison_y = np.array([7, 3, 3, 3, 3, 7, 7, 7, 7])
    for actual_label, expected_correct, expected_k in [(7, [1, 0, 0, 1], 1), (3, [0, 1, 1, 0], 3)]:
        candidates, chosen_k = compare_knn_settings(
            comparison_X, comparison_y, np.array([[0.1]]), np.array([actual_label])
        )
        assert list(candidates) == [1, 3, 5, 9]
        assert [candidate["correct"] for candidate in candidates.values()] == expected_correct
        assert chosen_k == expected_k, "Selection must maximize accuracy, then minimize k"

    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "selection.json"
        selection = {"model_parameters": {"n_neighbors": 1}}
        freeze_selection(path, selection)
        timestamp = path.stat().st_mtime_ns
        freeze_selection(path, selection)
        assert path.stat().st_mtime_ns == timestamp and json.loads(path.read_text()) == selection
        try:
            freeze_selection(path, {"model_parameters": {"n_neighbors": 3}})
        except ValueError:
            pass
        else:
            raise AssertionError("A different selection silently replaced the frozen choice")
        assert json.loads(path.read_text()) == selection, "Rejected change damaged the saved choice"
    print("KNN check passed: original sample mapping, labels, distances, votes, and tied-vote agreement.")
    print("Selection check passed: fixed candidates, best score, smaller-k tie rule, and frozen choice.")


if __name__ == "__main__":
    check_knn()
