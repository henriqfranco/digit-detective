from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, RadioButtons

from analyze_errors import load_selected_experiment
from experiment import trace_knn_prediction
from final_evaluate import load_test_predictions


class PredictionViewer:
    def __init__(self, digits, splits, model, predictions, test_predictions=None):
        self.digits, self.splits, self.model, self.predictions = digits, splits, model, predictions
        self.partition = "validation"
        self.partition_predictions = {"validation": predictions}
        if test_predictions is not None:
            self.partition_predictions["test"] = test_predictions
        self.X_train, self.y_train = digits.data[splits["train"]], digits.target[splits["train"]]
        self.rows = np.arange(len(splits["validation"]))
        self.position = 0
        self.last_trace = None
        self.figure, axes = plt.subplots(1, model.n_neighbors + 1, figsize=(9, 5.5), squeeze=False)
        self.axes = axes[0]
        self.figure.subplots_adjust(left=0.05, right=0.95, bottom=0.25, top=0.72, wspace=0.18)
        self.title = self.figure.suptitle("Digit Detective", fontsize=14)
        self.votes = self.figure.text(0.5, 0.19, "", ha="center", fontsize=11)
        self.figure.text(0.5, 0.02, "Left/right arrows: move   |   M: toggle mistakes   |   Votes are agreement, not confidence", ha="center", fontsize=10)
        self.previous = Button(self.figure.add_axes([0.16, 0.06, 0.18, 0.09]), "Previous")
        self.next = Button(self.figure.add_axes([0.38, 0.06, 0.18, 0.09]), "Next")
        self.only_mistakes = CheckButtons(self.figure.add_axes([0.61, 0.055, 0.29, 0.10]), ["Mistakes only"], [False])
        self.previous.on_clicked(lambda event: self.move(-1))
        self.next.on_clicked(lambda event: self.move(1))
        self.only_mistakes.on_clicked(self.update_filter)
        self.partition_selector = RadioButtons(self.figure.add_axes([0.06, 0.86, 0.22, 0.10]),
                                                tuple(self.partition_predictions))
        self.partition_selector.on_clicked(self.select_partition)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.render()

    def move(self, step):
        if len(self.rows):
            self.position = (self.position + step) % len(self.rows)
            self.render()

    def update_filter(self, _label):
        if self.only_mistakes.get_status()[0]:
            self.rows = np.flatnonzero(self.predictions != self.digits.target[self.splits[self.partition]])
        else:
            self.rows = np.arange(len(self.splits[self.partition]))
        self.position = 0
        self.render()

    def select_partition(self, partition):
        self.partition = partition
        self.predictions = self.partition_predictions[partition]
        self.update_filter(None)

    def on_key(self, event):
        if event.key in ("left", "right"):
            self.move(-1 if event.key == "left" else 1)
        elif event.key == "m":
            self.only_mistakes.set_active(0)

    def render(self):
        for ax in self.axes:
            ax.clear()
            ax.set_axis_off()
        if not len(self.rows):
            self.last_trace = None
            self.title.set_text(f"Digit Detective | {self.partition} | k={self.model.n_neighbors}\nNo mistakes to display")
            self.votes.set_text(f"Uncheck Mistakes only to browse all {self.partition} images.")
            self.figure.canvas.draw_idle()
            return
        row = self.rows[self.position]
        query_index = int(self.splits[self.partition][row])
        trace = trace_knn_prediction(self.model, self.digits.data[query_index], self.X_train, self.y_train, self.splits["train"])
        assert trace["prediction"] == self.predictions[row], "Viewer and recorded prediction disagree"
        trace["query_dataset_index"] = query_index
        trace["true_label"] = int(self.digits.target[query_index])
        self.last_trace = trace
        mode = "Mistakes only" if self.only_mistakes.get_status()[0] else f"All {self.partition}"
        self.title.set_text(f"Digit Detective | {self.partition} | k={self.model.n_neighbors}\n{mode}: {self.position + 1} / {len(self.rows)}")
        correct = trace["prediction"] == trace["true_label"]
        self.axes[0].imshow(self.digits.images[query_index], cmap="gray", vmin=0, vmax=16, interpolation="nearest")
        self.axes[0].set_title(f"{self.partition.capitalize()} sample {query_index}\nTrue {trace['true_label']} | predicted {trace['prediction']}\n{'Correct' if correct else 'Incorrect'}", fontsize=12)
        for column, neighbor in enumerate(trace["neighbors"], start=1):
            self.axes[column].imshow(self.digits.images[neighbor["dataset_index"]], cmap="gray", vmin=0, vmax=16, interpolation="nearest")
            self.axes[column].set_title(f"Voting neighbor {column}\nDataset sample {neighbor['dataset_index']} | label {neighbor['label']}\nTraining row {neighbor['training_row']}\nDistance {neighbor['distance']:.2f}", fontsize=11)
        self.votes.set_text(f"Votes: {trace['votes']}  |  One final prediction: {trace['prediction']}")
        self.figure.canvas.draw_idle()


def main():
    plt.switch_backend("WebAgg")
    matplotlib.rcParams["webagg.address"] = "127.0.0.1"
    matplotlib.rcParams["webagg.open_in_browser"] = False
    artifacts = Path(__file__).resolve().parent / "artifacts"
    digits, splits, selection, model, predictions = load_selected_experiment(artifacts)
    test_predictions = None
    if (artifacts / "final-test-results.json").exists():
        test_predictions = load_test_predictions(artifacts, selection, splits)
    viewer = PredictionViewer(digits, splits, model, predictions, test_predictions)
    viewer.figure.savefig(artifacts / "viewer-validation-example.png", dpi=160)
    viewer.only_mistakes.set_active(0, state=True)
    if viewer.last_trace is not None:
        viewer.figure.savefig(artifacts / "viewer-validation-mistake.png", dpi=160)
    viewer.only_mistakes.set_active(0, state=False)
    if test_predictions is not None:
        viewer.partition_selector.set_active(1)
        viewer.figure.savefig(artifacts / "viewer-test-example.png", dpi=160)
        viewer.partition_selector.set_active(0)
    plt.show()


if __name__ == "__main__":
    main()
