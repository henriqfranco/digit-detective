from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.datasets import load_digits

digits = load_digits()
images = digits.images
X = digits.data
y = digits.target
labels, counts = np.unique(y, return_counts=True)

# Dataset checks run each time this inspection script is executed.
assert images.shape == (1797, 8, 8), "Unexpected image dimensions"
assert X.shape == (len(images), 64), "Unexpected feature dimensions"
assert y.shape == (len(images),), "Images and labels must have equal sample counts"
assert np.isfinite(images).all() and np.isfinite(X).all() and np.isfinite(y).all(), "Nonfinite data"
assert images.min() == X.min() == 0 and images.max() == X.max() == 16, "Unexpected pixel range"
assert np.array_equal(labels, np.arange(10)), "Expected labels 0 through 9"
assert counts.sum() == len(y), "Class counts must cover every sample"

print(f"Dataset loaded successfully: {len(y)} labeled digit images.")
print("\nArray        Shape             Dtype")
print(f"images       {str(images.shape):<17} {images.dtype}")
print(f"X (features) {str(X.shape):<17} {X.dtype}")
print(f"y (labels)   {str(y.shape):<17} {y.dtype}")
print(f"\nPixel range: {X.min():g} to {X.max():g}")
print("All image, feature, and label values are finite.")
print(f"Distinct labels: {labels}")

print("\nLabel  Count  Share")
for label, count in zip(labels, counts):
    print(f"{label:>5}  {count:>5}  {count / len(y):>6.2%}")
print(f"Total: {counts.sum()}")

print("\nSample 0: the image, feature row, and label share the same index.")
print("Image pixel array:")
print(images[0])
print("Feature row:")
print(X[0])
print(f"Label: {y[0]}")
print("\nDataset checks passed.")

# Each image row contributes eight consecutive features, left to right.
sample_index = 0
image = images[sample_index]
vector = image.flatten()
reconstructed = vector.reshape(8, 8)
row, column = 2, 5
feature_index = 8 * row + column

assert np.array_equal(images.reshape(len(images), 64), X), "Image/feature order mismatch"
assert vector.shape == (64,) and np.array_equal(vector, X[sample_index]), "Flattening mismatch"
assert np.array_equal(reconstructed, image), "Reconstruction changed pixel values"
assert image[row, column] == vector[feature_index], "Pixel/feature mapping mismatch"

print("\nImage-to-vector trace:")
print(f"Sample {sample_index}, label {y[sample_index]}: {image.shape} -> {vector.shape} -> {reconstructed.shape}")
print(f"Feature index = 8 * {row} + {column} = {feature_index}")
print(f"images[{sample_index}, {row}, {column}] = X[{sample_index}, {feature_index}] = {vector[feature_index]:g}")
print("Flattening matches every dataset feature row; reconstruction preserves every pixel.")

artifacts = Path(__file__).resolve().parent / "artifacts"
artifacts.mkdir(exist_ok=True)
display = dict(cmap="gray", vmin=0, vmax=16, interpolation="nearest", origin="upper")

# Three fixed examples per class make handwriting differences visible.
gallery, axes = plt.subplots(6, 5, figsize=(10, 11), layout="constrained")
gallery.suptitle("Three examples of each digit: true labels, not predictions", fontsize=15)
for label in labels:
    sample_indices = np.flatnonzero(y == label)[:3]
    assert len(sample_indices) == 3 and np.all(y[sample_indices] == label)
    for rank, index in enumerate(sample_indices):
        ax = axes[3 * (label // 5) + rank, label % 5]
        ax.imshow(images[index], **display)
        ax.set_title(f"Label {label} | sample {index}", fontsize=11)
        ax.set_axis_off()
gallery.savefig(artifacts / "class-gallery.png", dpi=160)
plt.close(gallery)

trace = plt.figure(figsize=(10, 6), layout="constrained")
trace.suptitle(f"Same pixels, different arrangement: sample {sample_index}, label {y[sample_index]}", fontsize=15)
layout = trace.add_gridspec(2, 3, height_ratios=[3, 1])
accent = "#6366F1"
for position, (pixels, title) in enumerate([
    (image, "Original image (8, 8)"),
    (image, "Pixel brightness values"),
    (reconstructed, "Reconstructed image (8, 8)"),
]):
    ax = trace.add_subplot(layout[0, position])
    ax.imshow(pixels, **display)
    ax.set_title(title, fontsize=12)
    ax.add_patch(Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=accent, linewidth=2.5))
    if position == 1:
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xlabel("Pixel column index")
        ax.set_ylabel("Pixel row index")
        for r in range(8):
            for c in range(8):
                ax.text(c, r, f"{image[r, c]:g}", ha="center", va="center", fontsize=10,
                        color="white" if image[r, c] < 8 else "black")
    else:
        ax.set_axis_off()

ax = trace.add_subplot(layout[1, :])
# The extra axis is only for displaying a one-dimensional vector with imshow.
ax.imshow(vector[np.newaxis, :], aspect="auto", **display)
ax.add_patch(Rectangle((feature_index - 0.5, -0.5), 1, 1, fill=False, edgecolor=accent, linewidth=2.5))
ax.set_title(f"Flattened vector (64,): highlighted feature {feature_index} has brightness {vector[feature_index]:g}", fontsize=12)
ax.set_xticks([0, 8, 16, 21, 24, 32, 40, 48, 56, 63])
ax.set_yticks([])
ax.set_xlabel("Feature index: each group of eight came from one image row")
trace.savefig(artifacts / "image-vector-trace.png", dpi=160)
plt.close(trace)

print(f"\nSaved gallery: {artifacts / 'class-gallery.png'}")
print(f"Saved trace: {artifacts / 'image-vector-trace.png'}")
print("Step 2 checks passed.")
