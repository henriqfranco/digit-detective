import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from analyze_errors import load_selected_experiment
from experiment import SEED


FRACTIONS = (0.10, 0.25, 0.50, 1.0)


def nested_training_subsets(training, labels):
    rng = np.random.default_rng(SEED)
    class_orders = [rng.permutation(training[labels[training] == label])
                    for label in np.unique(labels[training])]
    subsets = {}
    for fraction in FRACTIONS:
        chosen = np.concatenate([pool[:max(1, round(len(pool) * fraction))]
                                 for pool in class_orders])
        # Preserve original order so ordering is not another changed factor.
        subsets[fraction] = training[np.isin(training, chosen)]
    return subsets


def main():
    artifacts = Path(__file__).resolve().parent / "artifacts"
    digits, splits, selection, full_model, full_predictions = load_selected_experiment(artifacts)
    training, validation = splits["train"], splits["validation"]
    X, y = digits.data, digits.target
    X_validation, y_validation = X[validation], y[validation]
    validation_copy = X_validation.copy()
    subsets = nested_training_subsets(training, y)
    rows, class_counts = [], {}
    previous = np.array([], dtype=int)
    print(f"Fixed k={full_model.n_neighbors}; same {len(validation)} clean validation examples.")
    print("Training fraction  Training count  Correct/total  Accuracy")
    for fraction, pool in subsets.items():
        assert np.isin(pool, training).all() and np.isin(previous, pool).all()
        assert not np.isin(pool, np.concatenate([validation, splits["test"]])).any()
        assert len(pool) >= full_model.n_neighbors
        counts = np.bincount(y[pool], minlength=10)
        assert (counts > 0).all()
        if fraction == 1.0:
            np.testing.assert_array_equal(pool, training)
            predictions = full_predictions
        else:
            model = KNeighborsClassifier(**selection["model_parameters"])
            model.fit(X[pool], y[pool])
            predictions = model.predict(X_validation)
        correct = int(np.count_nonzero(predictions == y_validation))
        row = {"fraction": fraction, "training_count": len(pool), "correct": correct,
               "total": len(validation), "accuracy": correct / len(validation)}
        rows.append(row)
        class_counts[f"train_{round(100 * fraction)}pct"] = counts.tolist()
        print(f"{fraction:>17.0%}  {len(pool):>14}  {correct:>7}/{len(validation)}  {row['accuracy']:>8.2%}")
        print(f"  Class counts (0..9): {counts.tolist()}")
        previous = pool

    np.testing.assert_array_equal(X_validation, validation_copy)
    np.testing.assert_array_equal(full_model.predict(X_validation), full_predictions)
    saved_subsets = {f"train_{round(100 * fraction)}pct": pool for fraction, pool in subsets.items()}
    np.savez(artifacts / "training-subsets.npz", **saved_subsets)
    with np.load(artifacts / "training-subsets.npz", allow_pickle=False) as saved:
        for key, pool in saved_subsets.items():
            np.testing.assert_array_equal(saved[key], pool)
    with (artifacts / "training-size-results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "partition": "validation", "selection": selection, "subset_seed": SEED,
        "sampling": "one NumPy default_rng permutation per class; nested prefixes; original training order",
        "rounding": "nearest integer per class (ties to even), minimum one",
        "label_order": list(range(10)), "class_counts": class_counts, "results": rows,
        "limitation": "one seeded nested subset sequence; no uncertainty estimate or k retuning",
    }
    (artifacts / "training-size-report.json").write_text(json.dumps(report, indent=2) + "\n")

    figure, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    counts = [row["training_count"] for row in rows]
    scores = [100 * row["accuracy"] for row in rows]
    ax.plot(counts, scores, "o-", color="#2563eb")
    for row, score in zip(rows, scores):
        ax.annotate(f"{row['correct']}/{row['total']}\n{score:.2f}%",
                    (row["training_count"], score), xytext=(0, -16),
                    textcoords="offset points", ha="center", va="top", fontsize=10)
    ax.set(title=f"More training examples | fixed k={full_model.n_neighbors}",
           xlabel="Number of training images", ylabel="Validation accuracy (%)",
           xticks=counts, ylim=(0, 105), xlim=(0, max(counts) * 1.10))
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", alpha=0.25)
    figure.savefig(artifacts / "learning-curve.png", dpi=160)
    plt.close(figure)
    print("Saved subsets, class counts, results, and learning-curve.png.")
    print("The full-training model is unchanged. No test predictions or evaluation.")


if __name__ == "__main__":
    main()
