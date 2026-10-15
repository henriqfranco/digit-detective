from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_errors import load_selected_experiment
from viewer import PredictionViewer
from final_evaluate import load_test_predictions


def check_viewer():
    artifacts = Path(__file__).resolve().parent / "artifacts"
    digits, splits, selection, model, predictions = load_selected_experiment(artifacts)
    viewer = PredictionViewer(digits, splits, model, predictions)
    assert viewer.last_trace["query_dataset_index"] == splits["validation"][0]
    viewer.on_key(SimpleNamespace(key="right"))
    assert viewer.last_trace["query_dataset_index"] == splits["validation"][1]
    viewer.on_key(SimpleNamespace(key="left"))
    assert viewer.position == 0
    viewer.on_key(SimpleNamespace(key="m"))
    assert len(viewer.rows) == 4
    visited = []
    for _ in range(len(viewer.rows)):
        trace = viewer.last_trace
        query_index = trace["query_dataset_index"]
        visited.append(query_index)
        assert trace["prediction"] != digits.target[query_index]
        np.testing.assert_array_equal(viewer.axes[0].images[0].get_array(), digits.images[query_index])
        for column, neighbor in enumerate(trace["neighbors"], start=1):
            assert neighbor["dataset_index"] in splits["train"]
            np.testing.assert_array_equal(viewer.axes[column].images[0].get_array(), digits.images[neighbor["dataset_index"]])
        viewer.move(1)
    assert visited == [794, 1038, 683, 899] and viewer.position == 0
    viewer.move(-1)
    assert viewer.last_trace["query_dataset_index"] == 899

    assert [label.get_text() for label in viewer.partition_selector.labels] == ["validation"]
    if (artifacts / "final-test-results.json").exists():
        test_predictions = load_test_predictions(artifacts, selection, splits)
        np.testing.assert_array_equal(test_predictions, model.predict(digits.data[splits["test"]]))
        final_viewer = PredictionViewer(digits, splits, model, predictions, test_predictions)
        final_viewer.partition_selector.set_active(1)
        assert final_viewer.partition == "test" and final_viewer.position == 0
        assert final_viewer.last_trace["query_dataset_index"] == splits["test"][0]
        assert "Test sample" in final_viewer.axes[0].get_title()
        final_viewer.only_mistakes.set_active(0, state=True)
        wrong = np.flatnonzero(test_predictions != digits.target[splits["test"]])
        assert final_viewer.rows.tolist() == wrong.tolist()
        for row in wrong:
            trace = final_viewer.last_trace
            assert trace["query_dataset_index"] == splits["test"][row]
            assert trace["prediction"] == test_predictions[row]
            assert trace["true_label"] != trace["prediction"]
            np.testing.assert_array_equal(final_viewer.axes[0].images[0].get_array(), digits.images[splits["test"][row]])
            assert all(neighbor["dataset_index"] in splits["train"] for neighbor in trace["neighbors"])
            final_viewer.move(1)
        assert final_viewer.position == 0
        final_viewer.partition_selector.set_active(0)
        assert final_viewer.partition == "validation" and len(final_viewer.rows) == 4
        assert final_viewer.last_trace["query_dataset_index"] == 794
        plt.close(final_viewer.figure)

    viewer.predictions = digits.target[splits["validation"]].copy()
    viewer.only_mistakes.set_active(0, state=False)
    viewer.only_mistakes.set_active(0, state=True)
    assert len(viewer.rows) == 0 and viewer.last_trace is None
    assert all(len(ax.images) == 0 for ax in viewer.axes)
    viewer.move(1)
    assert viewer.position == 0
    plt.close(viewer.figure)
    print("Viewer check passed: navigation, wrapping, filtering, partition gating/switching, matching image IDs, and empty mistakes.")


if __name__ == "__main__":
    check_viewer()
