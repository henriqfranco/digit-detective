import csv
import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 42
PARTITIONS = ("train", "validation", "test")


def load_or_create_splits(X, y, directory):
    """Preserve one 60/20/20 split of the default load_digits dataset."""
    if X.shape != (1797, 64) or y.shape != (1797,):
        raise ValueError("Expected the default ten-class load_digits dataset")
    directory.mkdir(parents=True, exist_ok=True)
    split_path = directory / "splits.npz"
    metadata_path = directory / "split-metadata.json"
    if split_path.exists() != metadata_path.exists():
        raise ValueError("The saved split is incomplete; inspect both files before proceeding")
    dataset_digest = sha256(X.tobytes() + y.tobytes()).hexdigest()
    existing = split_path.exists()

    if existing:
        metadata = json.loads(metadata_path.read_text())
        if metadata["dataset_digest"] != dataset_digest:
            raise ValueError("The saved split belongs to different data or sample ordering")
        with np.load(split_path, allow_pickle=False) as saved:
            if set(saved.files) != set(PARTITIONS):
                raise ValueError("Expected train, validation, and test index arrays")
            splits = {name: saved[name] for name in PARTITIONS}
    else:
        indices = np.arange(len(y))
        remaining, test = train_test_split(
            indices, test_size=0.20, random_state=SEED, stratify=y
        )
        train, validation = train_test_split(
            remaining, test_size=0.25, random_state=SEED, stratify=y[remaining]
        )
        splits = {"train": train, "validation": validation, "test": test}
        metadata = {
            "dataset": "sklearn.datasets.load_digits (default ten classes)",
            "dataset_shape": list(X.shape),
            "dataset_digest": dataset_digest,
            "seed": SEED,
            "fractions": {"train": 0.60, "validation": 0.20, "test": 0.20},
            "stratified": True,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        }

    for name, size in zip(PARTITIONS, (1077, 360, 360)):
        indices = splits[name]
        if indices.shape != (size,) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"Invalid {name} index array: expected {size} integer indices")
    combined = np.concatenate([splits[name] for name in PARTITIONS])
    if not np.array_equal(np.sort(combined), np.arange(len(y))):
        raise ValueError("Splits must cover every sample exactly once, with no overlap or missing samples")
    for name in PARTITIONS:
        if not np.array_equal(np.unique(y[splits[name]]), np.arange(10)):
            raise ValueError(f"The {name} partition must contain every class")

    if not existing:
        np.savez(split_path, **splits)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return splits, metadata


def evaluate_majority_baseline(X_train, y_train, X_validation, y_validation):
    """Calculate the constant rule manually, then check the library result."""
    labels, counts = np.unique(y_train, return_counts=True)
    # Sorted labels and the first maximum choose the lowest label on a tie.
    majority_label = int(labels[counts.argmax()])
    manual_predictions = np.full(len(X_validation), majority_label, dtype=y_train.dtype)
    correct = int(np.count_nonzero(manual_predictions == y_validation))
    manual_accuracy = correct / len(y_validation)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    library_predictions = baseline.predict(X_validation)
    library_accuracy = accuracy_score(y_validation, library_predictions)
    assert np.array_equal(manual_predictions, library_predictions), "Manual/library predictions disagree"
    assert manual_accuracy == library_accuracy, "Manual/library accuracy disagrees"
    return majority_label, correct, manual_accuracy


def main():
    digits = load_digits()
    X, y = digits.data, digits.target
    artifacts = Path(__file__).resolve().parent / "artifacts"
    existing = (artifacts / "splits.npz").exists()
    splits, metadata = load_or_create_splits(X, y, artifacts)
    print("Loaded the saved split." if existing else "Created and saved the split.")
    print(f"Recorded seed: {metadata['seed']}")

    print("\nPartition   Samples  Share")
    for name in PARTITIONS:
        print(f"{name:<11} {len(splits[name]):>7}  {len(splits[name]) / len(y):.2%}")

    counts = {name: np.bincount(y[splits[name]], minlength=10) for name in PARTITIONS}
    print("\nLabel  Train  Validation  Test")
    for label in range(10):
        print(f"{label:>5}  {counts['train'][label]:>5}  {counts['validation'][label]:>10}  {counts['test'][label]:>4}")

    # Use the same sample indices for features and labels to preserve pairing.
    X_train, y_train = X[splits["train"]], y[splits["train"]]
    X_validation, y_validation = X[splits["validation"]], y[splits["validation"]]
    print(f"\nTraining shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation shapes: X={X_validation.shape}, y={y_validation.shape}")
    print("Test partition reserved: its labels are counted only for allocation checks.")
    print("Split checks passed: integer indices, complete coverage, no overlap, and all classes present.")
    print(f"\nSaved indices: {artifacts / 'splits.npz'}")
    print(f"Saved metadata: {artifacts / 'split-metadata.json'}")

    majority_label, correct, accuracy = evaluate_majority_baseline(
        X_train, y_train, X_validation, y_validation
    )
    tied_labels = np.flatnonzero(counts["train"] == counts["train"].max())
    print("\nMajority-class baseline (training-only choice):")
    print(f"Most frequent training label(s): {tied_labels}; count each: {counts['train'].max()}")
    print(f"Rule: always predict {majority_label}; ties choose the lowest label.")
    print(f"Validation: {correct}/{len(y_validation)} correct, {len(y_validation) - correct} incorrect")
    print(f"Accuracy: {correct} / {len(y_validation)} = {accuracy:.6f} = {accuracy:.2%}")
    print("Manual predictions and accuracy match DummyClassifier and accuracy_score.")

    result = {
        "model": "majority_class",
        "configuration": f"always predict {majority_label}; lowest-label tie rule",
        "correct": correct,
        "total": len(y_validation),
        "accuracy": accuracy,
    }
    result_path = artifacts / "validation-results.csv"
    with result_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)
    print(f"Saved validation results: {result_path}")


if __name__ == "__main__":
    main()
