from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_errors import load_selected_experiment
from viewer import PredictionViewer


def check_viewer():
    artifacts = Path(__file__).resolve().parent / "artifacts"
    digits, splits, _, model, predictions = load_selected_experiment(artifacts)
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

    viewer.predictions = digits.target[splits["validation"]].copy()
    viewer.only_mistakes.set_active(0, state=False)
    viewer.only_mistakes.set_active(0, state=True)
    assert len(viewer.rows) == 0 and viewer.last_trace is None
    assert all(len(ax.images) == 0 for ax in viewer.axes)
    viewer.move(1)
    assert viewer.position == 0
    plt.close(viewer.figure)
    print("Viewer check passed: navigation, wrapping, filtering, matching image IDs, and empty mistakes.")


if __name__ == "__main__":
    check_viewer()
