import numpy as np

from analyze_errors import summarize_predictions


def check_errors():
    summary = summarize_predictions(np.array([0, 0, 1, 1, 2]), np.array([0, 1, 1, 2, 2]))
    assert summary["matrix"].sum() == 5 and np.trace(summary["matrix"]) == 3
    assert summary["matrix"][0, 1] == 1 and summary["matrix"][1, 2] == 1
    assert summary["wrong_rows"].tolist() == [1, 3]
    np.testing.assert_allclose(summary["normalized"][:3, :3], [[0.5, 0.5, 0], [0, 0.5, 0.5], [0, 0, 1]])
    np.testing.assert_allclose(summary["recall"][:3], [0.5, 0.5, 1])
    np.testing.assert_allclose(summary["precision"][:3], [1, 0.5, 0.5])
    assert np.isnan(summary["recall"][3]) and np.isnan(summary["precision"][3])

    perfect = summarize_predictions(np.array([0]), np.array([0]))
    assert perfect["wrong_rows"].size == 0 and perfect["recall"][0] == 1
    missed = summarize_predictions(np.array([0, 1]), np.array([0, 0]))
    assert missed["recall"][1] == 0 and np.isnan(missed["precision"][1])
    print("Error-analysis check passed: matrix direction, counts, normalization, precision/recall, and absent classes.")


if __name__ == "__main__":
    check_errors()
