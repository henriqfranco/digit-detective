import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_errors import load_selected_experiment
from experiment import SEED, trace_knn_prediction


NOISE_LEVELS = (1, 2, 4)
DIRECTIONS = ("left", "right", "up", "down")


def add_noise(images, standard_noise, sigma):
    return np.clip(images.astype(float) + sigma * standard_noise, 0, 16)


def shift_one_pixel(images, direction):
    shifted = np.zeros_like(images)
    if direction == "left":
        shifted[:, :, :-1] = images[:, :, 1:]
    elif direction == "right":
        shifted[:, :, 1:] = images[:, :, :-1]
    elif direction == "up":
        shifted[:, :-1, :] = images[:, 1:, :]
    elif direction == "down":
        shifted[:, 1:, :] = images[:, :-1, :]
    else:
        raise ValueError("Direction must be left, right, up, or down")
    return shifted


def main():
    artifacts = Path(__file__).resolve().parent / "artifacts"
    digits, splits, selection, model, clean_predictions = load_selected_experiment(artifacts)
    training, validation = splits["train"], splits["validation"]
    clean = digits.images[validation].astype(float)
    clean_copy = clean.copy()
    labels = digits.target[validation].copy()
    original_labels = labels.copy()
    X_train, y_train = digits.data[training], digits.target[training]
    standard_noise = np.random.default_rng(SEED).normal(size=clean.shape)
    images = {"clean": clean}
    images.update({f"noise_{sigma}": add_noise(clean, standard_noise, sigma) for sigma in NOISE_LEVELS})
    images.update({f"shift_{direction}": shift_one_pixel(clean, direction) for direction in DIRECTIONS})
    groups = {"noise": ["clean"] + [f"noise_{sigma}" for sigma in NOISE_LEVELS],
              "translation": ["clean"] + [f"shift_{direction}" for direction in DIRECTIONS]}
    predictions, results, cases = {}, {}, []
    clean_correct = clean_predictions == labels
    print(f"Fixed k={model.n_neighbors}; original training pool; {len(validation)} validation queries.")
    print("Condition       Correct/total  Accuracy  Newly wrong  Recovered")
    for condition, batch in images.items():
        assert batch.shape == clean.shape and np.isfinite(batch).all()
        assert 0 <= batch.min() <= batch.max() <= 16
        predictions[condition] = model.predict(batch.reshape(len(validation), 64))
        correct_mask = predictions[condition] == labels
        newly_wrong = np.flatnonzero(clean_correct & ~correct_mask)
        correct = int(correct_mask.sum())
        results[condition] = {
            "condition": condition, "correct": correct, "total": len(validation),
            "accuracy": correct / len(validation), "errors": len(validation) - correct,
            "newly_wrong": len(newly_wrong), "recovered": int(np.count_nonzero(~clean_correct & correct_mask)),
        }
        row = results[condition]
        assert correct == int(clean_correct.sum()) - row["newly_wrong"] + row["recovered"]
        print(f"{condition:<15} {correct:>7}/{len(validation)} {row['accuracy']:>9.2%} {row['newly_wrong']:>12} {row['recovered']:>10}")
        if condition == "clean":
            continue
        wrong_rows = np.flatnonzero(~correct_mask)
        candidates = newly_wrong if len(newly_wrong) else wrong_rows
        if not len(candidates):
            continue
        query_row = int(candidates[0])
        case = trace_knn_prediction(model, batch[query_row].reshape(64), X_train, y_train, training)
        assert case["prediction"] == predictions[condition][query_row]
        assert all(neighbor["dataset_index"] in training for neighbor in case["neighbors"])
        case.update({"condition": condition, "validation_row": query_row,
                     "query_dataset_index": int(validation[query_row]), "true_label": int(labels[query_row]),
                     "clean_prediction": int(clean_predictions[query_row]), "newly_wrong": bool(clean_correct[query_row])})
        cases.append(case)

    np.testing.assert_array_equal(clean, clean_copy)
    np.testing.assert_array_equal(digits.images[validation], clean_copy)
    np.testing.assert_array_equal(labels, original_labels)
    np.testing.assert_array_equal(digits.target[validation], original_labels)
    np.testing.assert_array_equal(predictions["clean"], clean_predictions)
    np.testing.assert_array_equal(model.predict(clean.reshape(len(validation), 64)), clean_predictions)
    for group, conditions in groups.items():
        with (artifacts / f"robustness-{group}-results.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=results["clean"].keys())
            writer.writeheader()
            writer.writerows([results[condition] for condition in conditions])
    archive = {"validation_indices": validation, "labels": labels, "standard_noise": standard_noise,
               **images, **{f"{key}_predictions": value for key, value in predictions.items()}}
    np.savez_compressed(artifacts / "robustness-inputs.npz", **archive)
    with np.load(artifacts / "robustness-inputs.npz", allow_pickle=False) as saved:
        for key, value in archive.items():
            np.testing.assert_array_equal(saved[key], value)
    report = {
        "partition": "validation", "selection": selection, "noise_seed": SEED,
        "noise": "clip(clean + sigma * shared standard normal array, 0, 16); sigma=1,2,4",
        "translation": "independent one-pixel shifts of clean images; zero fill; crop outgoing pixels",
        "case_selection": "first newly wrong validation row per condition; otherwise first error, if any",
        "results": list(results.values()), "failure_cases": cases,
        "limitation": "one noise realization; intended labels retained even if corruption harms legibility",
    }
    (artifacts / "robustness-report.json").write_text(json.dumps(report, indent=2) + "\n")

    conditions = list(images)
    figure, axes = plt.subplots(2, len(conditions), figsize=(16, 4.4), layout="constrained")
    figure.suptitle("Same validation images | clean, noise, and independent shifts | intended labels retained")
    for query_row in range(2):
        for column, condition in enumerate(conditions):
            ax = axes[query_row, column]
            ax.imshow(images[condition][query_row], cmap="gray", vmin=0, vmax=16, interpolation="nearest")
            ax.set_title(f"{condition}\nSample {validation[query_row]} | true {labels[query_row]}\nPredicted {predictions[condition][query_row]}", fontsize=9)
            ax.set_axis_off()
    figure.savefig(artifacts / "robustness-examples.png", dpi=160)
    plt.close(figure)

    for group, group_conditions in groups.items():
        selected_cases = [case for case in cases if case["condition"] in group_conditions]
        columns = 2 + model.n_neighbors
        figure, axes = plt.subplots(max(1, len(selected_cases)), columns,
                                    figsize=(3.2 * columns, 2.5 * max(1, len(selected_cases))),
                                    squeeze=False, layout="constrained")
        figure.suptitle(f"Validation {group} failures | k={model.n_neighbors} | first newly wrong case per condition, if available")
        if not selected_cases:
            axes[0, 0].text(0.5, 0.5, "No errors in these conditions", ha="center")
        for row, case in enumerate(selected_cases):
            query_row, condition = case["validation_row"], case["condition"]
            panels = [(clean[query_row], f"Clean sample {case['query_dataset_index']}\nTrue {case['true_label']} | predicted {case['clean_prediction']}"),
                      (images[condition][query_row], f"{condition} | intended label {case['true_label']}\nPredicted {case['prediction']} | votes {case['votes']}")]
            panels.extend((digits.images[neighbor["dataset_index"]],
                           f"Voting training sample {neighbor['dataset_index']}\nLabel {neighbor['label']} | distance {neighbor['distance']:.2f}")
                          for neighbor in case["neighbors"])
            for column, (image, title) in enumerate(panels):
                axes[row, column].imshow(image, cmap="gray", vmin=0, vmax=16, interpolation="nearest")
                axes[row, column].set_title(title, fontsize=10)
        for ax in axes.flat:
            ax.set_axis_off()
        figure.savefig(artifacts / f"robustness-{group}-failures.png", dpi=160)
        plt.close(figure)
    print("Saved separate result tables, input/prediction archive, examples, neighbor galleries, and report.")
    print("Clean inputs and predictions are unchanged. No refitting on corruptions or test evaluation.")


if __name__ == "__main__":
    main()
