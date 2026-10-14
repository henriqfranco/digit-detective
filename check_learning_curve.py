import numpy as np

from learning_curve import FRACTIONS, nested_training_subsets


def check_learning_curve():
    labels = np.repeat([2, 5, 8], [7, 8, 9])
    training = np.array([23, 0, 22, 1, 21, 2, 20, 3, 19, 4, 18, 5,
                         17, 6, 16, 7, 15, 8, 14, 9, 13, 10, 12, 11])
    original_training, original_labels = training.copy(), labels.copy()
    subsets = nested_training_subsets(training, labels)
    repeated = nested_training_subsets(training, labels)
    assert list(subsets) == list(FRACTIONS)
    assert [len(pool) for pool in subsets.values()] == [3, 6, 12, 24]
    previous = np.array([], dtype=int)
    for fraction, pool in subsets.items():
        assert np.isin(previous, pool).all(), "Smaller pools must be nested"
        assert len(np.unique(pool)) == len(pool), "No repeated training examples"
        np.testing.assert_array_equal(pool, training[np.isin(training, pool)])
        np.testing.assert_array_equal(pool, repeated[fraction])
        for label in np.unique(labels):
            expected = max(1, round(np.count_nonzero(labels[training] == label) * fraction))
            assert np.count_nonzero(labels[pool] == label) == expected
        previous = pool
    np.testing.assert_array_equal(subsets[1.0], training)
    np.testing.assert_array_equal(training, original_training)
    np.testing.assert_array_equal(labels, original_labels)
    print("Subset checks passed: nesting, class counts, original order, repeatability, and unchanged inputs.")


if __name__ == "__main__":
    check_learning_curve()
