import numpy as np

from experiment import evaluate_majority_baseline


def check_baseline():
    # Training favors 7; validation favors 2. The rule must still predict 7.
    X_train = np.zeros((3, 1))
    y_train = np.array([7, 2, 7])
    X_validation = np.zeros((4, 1))
    y_validation = np.array([2, 2, 2, 7])
    result = evaluate_majority_baseline(X_train, y_train, X_validation, y_validation)
    assert result == (7, 1, 0.25), "The baseline must use training labels, not validation frequencies"
    assert evaluate_majority_baseline(
        X_train, y_train, np.full((4, 1), 16.0), y_validation
    ) == result, "Pixel brightness must not affect this constant baseline"

    # A tie chooses the lowest class label, not its position in the input.
    assert evaluate_majority_baseline(
        X_train[:2], np.array([7, 2]), X_validation, np.array([2, 7, 2, 7])
    ) == (2, 2, 0.5), "Unexpected tie rule or accuracy calculation"
    print("Baseline check passed: training-only choice, ignored pixels, ties, and known accuracies.")


if __name__ == "__main__":
    check_baseline()
