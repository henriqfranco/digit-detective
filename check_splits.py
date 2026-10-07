from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.datasets import load_digits

from experiment import load_or_create_splits


def check_splits():
    digits = load_digits()
    X, y = digits.data, digits.target
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        first, metadata = load_or_create_splits(X, y, directory)
        assert [len(first[name]) for name in ("train", "validation", "test")] == [1077, 360, 360]
        assert np.array_equal(np.sort(np.concatenate(list(first.values()))), np.arange(len(y)))
        for indices in first.values():
            assert np.array_equal(np.unique(y[indices]), np.arange(10))
            expected_counts = np.bincount(y, minlength=10) * len(indices) / len(y)
            assert np.max(np.abs(np.bincount(y[indices], minlength=10) - expected_counts)) <= 2

        paths = [directory / "splits.npz", directory / "split-metadata.json"]
        timestamps = [path.stat().st_mtime_ns for path in paths]
        second, reloaded_metadata = load_or_create_splits(X, y, directory)
        assert metadata == reloaded_metadata
        assert timestamps == [path.stat().st_mtime_ns for path in paths], "Reload rewrote saved files"
        for name in first:
            assert np.array_equal(first[name], second[name]), "Reload changed membership or ordering"

        changed_X = X.copy()
        changed_X[0, 0] += 1
        try:
            load_or_create_splits(changed_X, y, directory)
        except ValueError:
            pass
        else:
            raise AssertionError("Saved indices were accepted for a different dataset")

        broken = {name: indices.copy() for name, indices in first.items()}
        broken["train"][0] = broken["test"][0]
        np.savez(paths[0], **broken)
        try:
            load_or_create_splits(X, y, directory)
        except ValueError:
            pass
        else:
            raise AssertionError("An overlapping split was accepted")

        paths[1].unlink()
        try:
            load_or_create_splits(X, y, directory)
        except ValueError:
            pass
        else:
            raise AssertionError("An incomplete saved split was silently replaced")

    print("Split check passed: coverage, class presence, exact reload, and invalid-save rejection.")


if __name__ == "__main__":
    check_splits()
