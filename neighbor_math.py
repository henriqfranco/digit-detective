import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def squared_distance(a, b):
    return np.sum((a - b) ** 2)


def demo():
    query = np.array([1, 2, 3], dtype=float)
    a = np.array([2, 4, 3], dtype=float)
    b = np.array([1, 2, 5], dtype=float)
    assert squared_distance(query, a) == 5
    assert squared_distance(query, b) == 4
    print(f"Query: {query}")
    print(f"Example a: differences {query - a}, squared distance 5, distance {np.sqrt(5):.6f}")
    print(f"Example b: differences {query - b}, squared distance 4, distance 2; b is nearer.")

    # A new tiny training set: each row has three features and one label.
    training = np.array([[1, 2, 4], [1, 4, 3], [4, 2, 3], [1, 2, 7]], dtype=float)
    labels = np.array([7, 3, 3, 7])
    squared = np.array([squared_distance(query, sample) for sample in training])
    assert np.array_equal(squared, [1, 4, 9, 16])
    distances = np.sqrt(squared)
    order = np.argsort(squared)
    assert np.array_equal(order, np.argsort(distances)), "Square root must preserve neighbor order"
    batch = query.reshape(1, 3)

    print("\nTraining example  Label  Squared distance  Distance")
    for index in order:
        print(f"{index:>16}  {labels[index]:>5}  {squared[index]:>16g}  {distances[index]:>8g}")

    for k, expected in [(1, 7), (3, 3)]:
        selected = order[:k]
        classes, votes = np.unique(labels[selected], return_counts=True)
        prediction = int(classes[votes.argmax()])
        assert prediction == expected

        model = KNeighborsClassifier(n_neighbors=k, metric="euclidean", weights="uniform", algorithm="brute")
        model.fit(training, labels)
        library_distances, library_indices = model.kneighbors(batch)
        np.testing.assert_allclose(library_distances[0], distances[selected])
        assert np.array_equal(library_indices[0], selected)
        assert model.predict(batch)[0] == prediction
        print(f"\nk={k}: neighbor labels {labels[selected].tolist()} -> one prediction: {prediction}")
        print(f"Votes: {dict(zip(classes.tolist(), votes.tolist()))}; manual and library results match.")

    # Three different labels can tie despite odd k and distinct distances.
    tied_vote = KNeighborsClassifier(n_neighbors=3, metric="euclidean", weights="uniform", algorithm="brute")
    tied_vote.fit(training[:3], np.array([3, 1, 2]))
    tied_prediction = int(tied_vote.predict(batch)[0])
    assert tied_prediction == 1, "Unexpected uniform-vote tie behavior in the recorded library"
    print(f"\nVote tie: [3, 1, 2], one vote each; library prediction: {tied_prediction}.")

    # A different issue: equally distant candidates compete for the final slot.
    boundary_points = np.array([[1, 0, 0], [-1, 0, 0]], dtype=float)
    boundary_labels = np.array([7, 3])
    origin = np.zeros((1, 3))
    print("\nDistance tie: two candidates at distance 1, but k=1 has only one slot.")
    for ordering in [[0, 1], [1, 0]]:
        model = KNeighborsClassifier(n_neighbors=1, metric="euclidean", weights="uniform", algorithm="brute")
        model.fit(boundary_points[ordering], boundary_labels[ordering])
        print(f"Training label order {boundary_labels[ordering].tolist()} -> prediction {model.predict(origin)[0]}")

    print("\nTiny-vector checks passed. No digit partitions or benchmark results were accessed.")


if __name__ == "__main__":
    demo()
