import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from experiment import trace_knn_prediction


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
    print("KNN check passed: original sample mapping, labels, distances, votes, and tied-vote agreement.")


if __name__ == "__main__":
    check_knn()
