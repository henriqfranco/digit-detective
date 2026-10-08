# Digit Detective

An educational project for recognizing handwritten digits and explaining predictions through their nearest training examples.

## Step 0: define the task

The input is an 8x8 grayscale image represented by pixel intensities. The output is one category from 0 through 9.

A **sample** is one image. A **label** is the known digit associated with that image. A labeled example pairs the image with its answer. Later we will describe each image using numerical **features**: its pixel intensities.

This is **supervised learning** because training examples include known answers. During training, a classifier receives inputs paired with labels. During prediction, it receives the input without its correct label; we can compare the prediction with a known label afterward to evaluate it.

This is **multiclass classification** because the output is one of ten categories. Although those categories are written as numbers, they are class identifiers. Predicting 8 instead of 7 is not automatically a smaller error than predicting 2 instead of 7. A regression task would instead predict a numerical quantity, such as a temperature or a price.

For this project, examples are supplied by scikit-learn's built-in load_digits dataset. This is the small 8x8 educational dataset, not 28x28 MNIST. The script now performs loading, inspection, and Step 2 visualization.

## Run the inspection script

From the project folder:

```sh
.venv/bin/python inspect_digits.py
```

Verified Step 0 output (the script now prints inspection/trace results and saves two figures as well):

```text
Dataset loaded successfully: 1797 labeled digit images.
```

The original Step 0 script had three statements:

```python
from sklearn.datasets import load_digits

digits = load_digits()

print(f"Dataset loaded successfully: {len(digits.target)} labeled digit images.")
```

- The import makes the dataset-loading function available in this script.
- Parentheses call that function; assignment binds its returned dataset object to the name `digits`.
- `digits.target` accesses the labels, `len(...)` counts them, and the f-string inserts that count into the printed message.

Loading data is not fitting a model. No classifier or predictions have been created yet.

## Environment and why it exists

The project uses a local virtual environment in `.venv`. It provides an isolated installation location for project packages while using the existing Python interpreter. Calling `.venv/bin/python` selects that environment directly, so activation is optional.

| Tool | Tested version | Role |
| --- | --- | --- |
| Python | 3.14.4 | Runs our program |
| NumPy | 2.5.3 | Numerical arrays and calculations |
| Matplotlib | 3.11.2 | Images, plots, and later the viewer |
| scikit-learn | 1.9.1 | Dataset loading, classifiers, and evaluation |

`requirements.txt` records exact installed versions of the three libraries and their supporting dependencies. Those additional packages are dependencies installed by the libraries, not extra application frameworks. Version recording helps explain or reproduce behavior later; it is not a guarantee of identical results on every platform.

Python on this machine lacks `ensurepip`, the helper normally used to install pip into a new virtual environment. We created the environment without it and used the existing pip to target that environment. This follows pip's documented `--python` option and did not install the project libraries into the system Python.

To recreate the environment if it is absent, using the tested Python version:

```sh
python3 -m venv --without-pip .venv
python3 -m pip --python .venv install --only-binary=:all: pip
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
.venv/bin/python inspect_digits.py
.venv/bin/python -m pip check
```

Prebuilt packages avoid compiling numerical libraries locally. Dependency checks and fresh-process dataset loading passed. Importing NumPy, Matplotlib's plotting module, and scikit-learn also passed. Visible plot verification belongs to the later plotting step.

## Step 0 understanding checkpoint

1. What two pieces of information make up one labeled training example?
2. Why is predicting a digit classification even though the labels are numbers?
3. Should the correct label be supplied to the classifier during prediction? Why?

Step 0 is complete: setup verified and learning checkpoint discussed. Classification means selecting a discrete category; supplying the correct label during prediction would leak the answer and make evaluation misleading. Continue with [the plan](tasks/plan.md) and [the checklist](tasks/todo.md).

## Step 1: inspect the data and understand X and y

The inspection starts with:

```python
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
images = digits.images
X = digits.data
y = digits.target
```

`np` is a short alias for NumPy. The dataset contains NumPy arrays: numerical collections with dimensions and a common element type. These assignments give names to existing arrays; they do not create new copies, normalize values, or train a model.

`X` conventionally names the feature matrix; `y` names the label array. Those names have no special Python behavior. Scikit-learn supplies the image and feature representations, so we do not need to build a conversion ourselves in this step.

### Shapes and axes

An array's `shape` is a tuple giving the length of each axis. Verified values:

| Array | Shape | Meaning of axes | Dtype |
| --- | --- | --- | --- |
| images | (1797, 8, 8) | sample, pixel row, pixel column | float64 |
| X | (1797, 64) | sample, feature position | float64 |
| y | (1797,) | sample | int64 |

Every image has 8 times 8 = 64 pixels, so its feature representation has 64 entries. The two forms represent the same collection of images. The explicit image/vector conversion and visual gallery belong to Step 2.

The first axis always selects a sample. `X[0]` has shape `(64,)`, `images[0]` has shape `(8, 8)`, and `y[0]` is one scalar label. Python indexing starts at zero. Therefore `X[12, 20]` accesses feature index 20 of sample index 12: the 21st feature of the 13th sample. It is one intensity, not a label, entire image, or model prediction.

The comma in `(1797,)` denotes a one-element tuple. This is a one-dimensional array with 1,797 entries, not a two-dimensional array with a missing column count. There is one label per sample, so `y` does not need ten columns. Ten is the number of available categories, not the number of labels assigned to each image.

### Values versus storage types

The observed intensity range is 0 through 16. Low intensities represent dark background in the usual grayscale display; high intensities represent brighter stroke pixels. Intensities describe brightness, not digit categories. A pixel equal to 8 does not tell us the image's label is 8.

`float64` means pixels use a 64-bit floating-point representation. The values can still be whole numbers such as `5.0`; using floating-point storage does not imply the dataset contains fractional intensities. `int64` means the labels use 64-bit integers. Neither storage choice determines whether the task is classification or regression: that depends on what the target means.

The script checks finite values using `np.isfinite(...).all()`. `isfinite` produces a Boolean result for each entry, and `all()` asks whether every result is true. This rules out NaN (not-a-number) and positive/negative infinity. Such entries can invalidate numerical operations. It is a basic validity check, not proof that every image or annotation is perfect.

### Labels and class counts

```python
labels, counts = np.unique(y, return_counts=True)
```

This returns the distinct labels and their occurrence counts. The two outputs are assigned to two names in the same statement. `labels[j]` corresponds to `counts[j]`; the default result is sorted by label.

Measured results:

| Label | Count | Share |
| --- | --- | --- |
| 0 | 178 | 9.91% |
| 1 | 182 | 10.13% |
| 2 | 177 | 9.85% |
| 3 | 183 | 10.18% |
| 4 | 181 | 10.07% |
| 5 | 182 | 10.13% |
| 6 | 181 | 10.07% |
| 7 | 179 | 9.96% |
| 8 | 174 | 9.68% |
| 9 | 180 | 10.02% |

The counts sum to 1,797. Classes are nearly balanced, not exactly equal. The share is count divided by total sample count, displayed as a percentage. This describes class representation; it is not a model accuracy result.

The output loop uses `zip(labels, counts)` to pair each label with its count. Formatting such as `:>5` aligns table columns, and `.2%` displays a fraction as a percentage with two decimal places. Formatting affects presentation, not the data.

### Image/label alignment

`images[i]`, `X[i]`, and `y[i]` refer to the same sample. The script displays sample 0's full pixel array, feature row, and label; its label is 0.

If an image of a seven is paired with another sample's label of two, a learning algorithm receives an incorrect image/answer pair. Equal array lengths would not reveal that error. Our arrays come from the loader's corresponding records and we have not reordered them. Later, any shuffle or split must use the same sample indices across all three arrays. The present assertions check structural consistency, not independent correctness of the dataset's annotations.

### Checks and understanding checkpoint

The script's assertions check documented dimensions, consistent sample counts, finite values, intensity endpoints, expected categories, and complete class counts. An assertion stops ordinary execution with an explanation if its condition is false. These are educational self-checks; run the script normally, without Python's assertion-disabling optimization option.

Fresh-process execution passed all checks. Step 1 is complete: the learner demonstrated sample/feature indexing, the distinction between X and y, and the need to preserve image/label alignment. The checkpoint questions were:

1. In `X.shape == (1797, 64)`, what does each number represent, and why is the second number 64 rather than 10?
2. What exactly does `X[12, 20]` select? What does `y[12]` select?
3. Why would shuffling X without applying the same reordering to y break training, even if their shapes stayed unchanged?

## Step 2: see the images and trace their feature vectors

Running the same script also generates:

- `artifacts/class-gallery.png`: three examples for each label, with original sample indices.
- `artifacts/image-vector-trace.png`: sample 0 as an image, a numeric pixel grid, a feature strip, and a reconstructed image.

These figures display dataset labels, not classifier predictions. The gallery takes the first three samples of each class for a deterministic illustration; it does not estimate performance. Images with the same label have different stroke shapes and intensities, showing why a digit is not represented by one universal pixel pattern.

### The small operation behind flattening

```python
image = images[0]
vector = image.flatten()
reconstructed = vector.reshape(8, 8)
```

An 8x8 image has 64 intensities. `flatten()` places them into a one-dimensional array in the default row-major order: first image row left to right, then the next row, until all eight rows have been read. It returns a new flattened copy. A vector is simply an ordered one-dimensional collection of numbers here.

For a smaller example:

```text
Grid:                 Vector:
1  2                  [1, 2, 3, 4]
3  4
```

The actual sample's verified shapes are:

```text
(8, 8) -> (64,) -> (8, 8)
 image    vector    reconstructed image
```

`reshape(8, 8)` arranges the same 64 values back into eight rows and eight columns. Flattening and reshaping do not average, resize, normalize, or erase pixel intensities. With the shape and ordering known, this conversion is reversible. Each sample still contributes one row to X; the eight image rows are pixel rows within that single sample, not eight separate training examples.

### Follow one pixel

The purple outline follows sample 0's pixel at row index 2, column index 5. Those indices refer to the third pixel row and sixth pixel column because indexing starts at zero. Its measured brightness is 11.

Each complete image row contributes eight vector entries:

| Image row index | Vector feature indices |
| --- | --- |
| 0 | 0 through 7 |
| 1 | 8 through 15 |
| 2 | 16 through 23 |
| 3 | 24 through 31 |
| 4 | 32 through 39 |
| 5 | 40 through 47 |
| 6 | 48 through 55 |
| 7 | 56 through 63 |

Two complete rows contribute 16 entries before the selected row begins. Moving five positions along that row gives index 16 + 5 = 21. More generally, for an eight-column image:

```python
feature_index = 8 * row + column
```

Verified identity:

```text
images[0, 2, 5] = X[0, 21] = 11
```

`images[0, 2, 5]` selects a sample, then its pixel row and pixel column. `X[0, 21]` selects the same sample, then the pixel's position in its flattened list. `y[0]` remains the category label 0. Position 21 and brightness 11 are not category labels.

### Why these display choices?

Every plot uses `cmap="gray"`, fixed intensity limits `vmin=0, vmax=16`, and `interpolation="nearest"`. A given value therefore has the same displayed brightness in every image. Enlargement shows the original pixels as blocks rather than smoothing or inventing intermediate visual detail. `origin="upper"` places image row 0 at the top, matching the printed grid.

The vector strip uses an extra axis solely because `imshow` displays a two-dimensional array; the actual vector still has shape `(64,)`. The rectangles highlight existing pixels without modifying the arrays. Saving figures makes the results inspectable and reproducible in this environment, whose Matplotlib backend renders image files.

### What the representation does and does not provide

Flattening preserves all pixel information and a consistent position order. It does not construct features such as stroke count, loop count, or digit meaning. Later, our nearest-neighbor model will compare these ordered intensities. Its comparison rule does not automatically treat image geometry like a human does; we will investigate that when testing translations.

### Verification and learning checkpoint

The script checks that all 1,797 flattened images match the corresponding supplied X rows, that the selected vector has 64 entries, that the selected pixel mapping is correct, and that reconstructing sample 0 preserves every pixel exactly. It also checks the selected gallery labels. All checks passed, and both saved figures were visually inspected.

Step 2 and the data-understanding checkpoint are complete. The learner distinguished sample count from feature count, explained preservation of pixel brightness through flattening and reshaping, and correctly calculated that pixel row 1, column 3 maps to feature index 11. Step 3 will establish reproducible training, validation, and test partitions.

## Step 3: preserve training, validation, and test partitions

Run the new experiment script from the project folder:

```sh
.venv/bin/python experiment.py
```

The first run creates and saves the split. Later runs load the saved arrays without regenerating or rewriting them. Split setup itself fits no model; the script now also performs the Step 4 baseline described below. The inspection script stays focused on its original dataset exploration and figures.

### Why three groups?

| Partition | Role | Measured count |
| --- | --- | --- |
| Training | Supply the examples and labels used to fit the model | 1,077 |
| Validation | Evaluate development candidates and choose settings | 360 |
| Test | Evaluate the frozen choice after development | 360 |

Think of training as practice material, validation as a mock exam that helps you adjust your approach, and test as the reserved final exam. A model can perform well on examples it already fitted on without performing well on new examples. The ability to perform on unseen examples from the relevant population is called generalization.

Validation examples are excluded from fitting, but their results influence choices. This is why a separate test partition remains useful: the winning validation score helped select the model and can be optimistic. Consulting test results to choose settings would turn the test partition into additional development data. Repeating a frozen evaluation for reproducibility is different from changing the model in response to test outcomes.

### Why approximately 60/20/20?

This gives most examples to training while retaining useful evaluation groups. The proportions are an educational starting tradeoff, not a universally optimal choice. Whole samples cannot be divided, so actual percentages are 59.93% training and 20.03% each for validation and test.

We reserve 20% test first: 360 samples, leaving 1,437. Then 25% of the remainder becomes validation: another 360. The rest, 1,077, is training. The second fraction is 25% because 25% of the remaining 80% is 20% of the original total. Choosing 20% of the remainder instead would allocate only about 16% of the original total to validation.

### Split sample positions once

The creation branch uses:

```python
indices = np.arange(len(y))
remaining, test = train_test_split(
    indices, test_size=0.20, random_state=SEED, stratify=y
)
train, validation = train_test_split(
    remaining, test_size=0.25, random_state=SEED, stratify=y[remaining]
)
```

`indices` contains original sample positions 0 through 1,796. Each index identifies an entire image, not one pixel. `train_test_split` shuffles and divides this list. A sample remains intact, so its 64 features remain together.

In the second call, `y[remaining]` supplies the labels corresponding to exactly the remaining indices, in their current order. Using the full y array there would mismatch the input membership and label lengths.

Once indices are chosen, select corresponding feature and label rows together:

```python
X_train = X[splits["train"]]
y_train = y[splits["train"]]
```

NumPy accepts an array of indices to select multiple rows. Both selections use exactly the same original sample positions and order. Therefore the image/label pairing stays intact. The verified shapes are X_train=(1077,64), y_train=(1077,), X_validation=(360,64), y_validation=(360,). Splitting changes sample counts, not the 64-feature representation.

### Stratification and seed

`stratify=y` tells the splitter to preserve class proportions approximately. Labels are used to allocate examples; this is not classifier training. During later prediction, held-out labels will be used after predictions to calculate results, not supplied as prediction inputs.

Verified class allocation:

| Label | Training | Validation | Test |
| --- | --- | --- | --- |
| 0 | 106 | 36 | 36 |
| 1 | 109 | 37 | 36 |
| 2 | 107 | 35 | 35 |
| 3 | 109 | 37 | 37 |
| 4 | 109 | 36 | 36 |
| 5 | 109 | 36 | 37 |
| 6 | 109 | 36 | 36 |
| 7 | 107 | 36 | 36 |
| 8 | 104 | 35 | 35 |
| 9 | 108 | 36 | 36 |

Every partition includes all ten classes. Rounding means allocation is approximate rather than identical fractional counts.

`SEED=42` provides a repeatable pseudo-random allocation in the recorded environment. The number has no special statistical merit. We do not search for a seed that produces flattering accuracy. Saving indices preserves exact membership and order directly, even if the generation procedure changes later. All candidate settings will use these same validation examples so a score comparison does not also change the evaluation sample.

### Files and functions

- `experiment.py`: one split creation/loading function and a main program that reports allocation.
- `artifacts/splits.npz`: NumPy archive containing named train, validation, and test arrays of original sample indices.
- `artifacts/split-metadata.json`: dataset identity, seed, intended proportions, stratification choice, and creation-time Python/NumPy/scikit-learn versions.
- `check_splits.py`: one runnable correctness check, using temporary files and the existing libraries.

The metadata includes a digest of the ordered dataset: a compact fingerprint that changes if its numerical data or row order changes. This prevents applying old sample positions to different input data. It is an identity check, not learned preprocessing. The script rejects a partial saved pair rather than silently replacing it.

`def` defines a function that runs when called. The final `if __name__ == "__main__": main()` runs the main workflow when executing experiment.py directly. Importing its split function from the check script does not create project artifacts or run the reporting workflow.

### Verification and boundaries

Run the persistent check with:

```sh
.venv/bin/python check_splits.py
```

The check passed for intended sample counts, complete coverage with no duplicates or overlaps, all classes present, approximately preserved class proportions, exact saved membership/order, and unchanged files on reload. It also confirmed rejection of changed source data, overlapping saved indices, and an incomplete file pair. Those negative cases only alter temporary test artifacts. Actual project creation and a separate fresh-process reload both succeeded.

For this step, test labels are only counted to verify allocation. Individual test images, predictions, accuracy, and errors are not examined. Earlier steps explored the full dataset's structure and a small gallery before this split existed; we record that honestly. From now on, development analysis uses training and validation, and the existing broad inspection gallery must not be used to investigate selected test examples.

If learned preprocessing is introduced later, it must be fitted on training data only and applied unchanged elsewhere. A random sample split also does not establish an unseen-writer evaluation: writer identities would be needed for a writer-separated split. The current benchmark evaluation concerns held-out samples from this dataset.

Step 3 is complete: saved partition checks passed and the learner explained training-score optimism, the distinction between validation-guided selection and an independent test evaluation, the second-stage 25% calculation, and seed repeatability. Step 4 will establish a training-majority baseline and evaluate it on validation. Test prediction and evaluation remain reserved.

## Step 4: majority-class baseline

A baseline is a simple reference against which to compare later models. Our rule counts training labels, chooses the most frequent class, and predicts that same class for every query. It does not inspect pixel brightness or digit shapes. Its chosen class still comes from training data: even a very simple fitted rule can have data-dependent state.

### Count training labels and choose the constant

The manual calculation in `evaluate_majority_baseline` begins with:

```python
labels, counts = np.unique(y_train, return_counts=True)
majority_label = int(labels[counts.argmax()])
manual_predictions = np.full(len(X_validation), majority_label, dtype=y_train.dtype)
```

`np.unique` returns sorted labels and corresponding occurrence counts. `argmax()` returns the position of the largest count, not the count itself or the class label. We use that position to select the corresponding label. `int` converts the NumPy integer to an ordinary Python integer. `np.full` creates an array filled with the chosen label, one entry per validation image.

The actual training counts have a tie: classes 1,3,4,5,6 each occur 109 times. The rule chooses the lowest tied label, 1. Sorted labels and argmax's first-maximum behavior implement that convention; the installed DummyClassifier uses the same choice. This rule does not consult validation outcomes to break ties.

The name majority-class baseline conventionally refers to the most frequent class; a class does not have to exceed half the dataset. Here several classes tie for the largest share, around 10.12% each of training.

### Prediction and scoring are separate

Every validation prediction is 1. A query with actual label 1 is correct; every other query is incorrect. The model's choice comes from y_train. Only after constructing the predictions do we compare them with y_validation:

```python
correct = int(np.count_nonzero(manual_predictions == y_validation))
manual_accuracy = correct / len(y_validation)
```

Elementwise equality produces a Boolean array, with True for a matching prediction and False otherwise. `np.count_nonzero` counts the True entries. Because prediction and target arrays use the same validation order, each comparison concerns the same sample.

Measured validation results:

| Rule | Correct | Total | Incorrect | Accuracy |
| --- | --- | --- | --- | --- |
| Always predict 1 | 37 | 360 | 323 | 10.28% |

There are 37 actual ones among the 360 validation examples. Accuracy is 37/360 = 0.102777... as a fraction, or approximately 10.28% as a percentage. The training count of 109 determines the chosen rule; it is not the numerator of validation accuracy.

### Verify the library version of the same rule

```python
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
library_predictions = baseline.predict(X_validation)
library_accuracy = accuracy_score(y_validation, library_predictions)
```

`strategy="most_frequent"` explicitly selects the rule we derived. `fit` establishes the class information from training labels. X_train is supplied to match the estimator API, but this strategy does not learn from its pixel values. `predict` returns labels for the validation inputs without receiving their correct answers. `accuracy_score` compares the predictions with those answers afterward.

Assertions confirm that the entire manual prediction array matches the library array and the manual accuracy matches the library score. This links an understandable calculation to the reusable library API.

### Why this result is useful

A roughly balanced ten-class dataset gives a constant-class rule roughly one-tenth accuracy. We predicted that rough scale and then measured 10.28%; the score comes from the actual partition, not a guarantee of exactly 10%.

Future classifiers will be measured on the same validation examples. Their added complexity should produce useful behavior relative to this simple rule. Beating one baseline alone does not establish that a model is reliable or useful on other distributions; error analysis and the final test still matter. There is no required accuracy target for finishing this learning project.

Overall accuracy must be interpreted with class composition. In a hypothetical evaluation group where 95% of examples belong to the class selected by training, always predicting that class achieves 95% accuracy while missing every example of the remaining classes. A high aggregate score can therefore hide failure on minority classes. This is why we inspected counts and will later inspect per-class errors.

### Saved result, checks, and next learning questions

`artifacts/validation-results.csv` records the rule, correct count, denominator, and full-precision fractional accuracy as the first row of our comparison table. The experiment script regenerates the results it currently computes; future comparison stages will extend that computation and table together.

Run:

```sh
.venv/bin/python experiment.py
.venv/bin/python check_baseline.py
.venv/bin/python check_splits.py
```

The baseline check uses small labeled arrays to establish training-only choice even when validation favors a different class, unchanged predictions when query brightness changes, deterministic handling of tied nonconsecutive labels, and known accuracy values. Both checks passed. Actual benchmark execution matched the library, and byte fingerprints confirmed the existing split and metadata files were unchanged. No test predictions or accuracy were produced.

Step 4 is complete: baseline predictions and score are verified, and the learner explained how high aggregate accuracy can hide zero recognition of another class and why the constant prediction must come from training frequencies. Matching the library verifies the calculation; the scientific reason for training-only choice is to keep fitting separate from held-out evaluation. Step 5 will derive tiny-vector distances and neighbor votes before using KNN on digit images.

## Step 5: tiny-vector distances and votes

We calculated distances and votes by hand before expressing them in Python. `neighbor_math.py` is a self-contained educational example with runnable assertions. It uses no digit images, saved partitions, or benchmark results.

```sh
.venv/bin/python neighbor_math.py
```

### Distance summarizes feature differences

For query q=[1,2,3] and example a=[2,4,3], subtraction gives [-1,-2,0]. Squaring gives [1,4,0]; summing gives squared distance 5; taking the square root gives Euclidean distance sqrt(5), approximately 2.236.

The learner calculated q minus b=[1,2,5] as [0,0,-2], squared distance 4, and distance 2. Therefore b is nearer because 2 is smaller than 2.236. A distance need not occur as a value inside the query vector. A feature value describes one coordinate; a distance summarizes differences across corresponding coordinates. Numerical equality between a feature value and a distance is coincidental.

Squaring prevents positive and negative feature differences from cancelling. Identical vectors have distance zero. Smaller Euclidean distance means nearer under this numerical comparison; it does not guarantee matching labels or human-perceived similarity.

The calculation is:

```python
def squared_distance(a, b):
    return np.sum((a - b) ** 2)
```

Array subtraction operates on corresponding entries. `** 2` squares each difference; `np.sum` adds them. `np.sqrt` converts the squared distance into Euclidean distance. This helper is only an arithmetic teaching example; later digit classification will use the library's neighbor implementation.

Square root preserves the order of nonnegative numbers, so sorting squared distances selects the same neighbors as sorting Euclidean distances. They are different numerical quantities; replacing distance with squared distance would change inverse-distance weights if those were introduced. Our initial voting uses equal weights.

### One query, several votes, one prediction

For a new tiny training set and the same query q:

| Training vector | Label | Squared distance | Euclidean distance |
| --- | --- | --- | --- |
| [1,2,4] | 7 | 1 | 1 |
| [1,4,3] | 3 | 4 | 2 |
| [4,2,3] | 3 | 9 | 3 |
| [1,2,7] | 7 | 16 | 4 |

`k` is the number of nearest training examples selected to vote. At k=1, the selected label is [7] and the prediction is 7. At k=3, the selected labels are [7,3,3]: class 3 has two votes and class 7 one, so the single prediction is 3. The closest neighbor can lose the vote. The learner correctly predicted 7 for the alternative selected labels [7,7,3].

Neighbor labels are inputs to the vote, not three final predictions for the same query. Each selected neighbor contributes one vote. Class labels are categories, so the classifier counts occurrences rather than averaging their numerical identifiers. The winning class has the most votes; with more than two categories it need not exceed half of k.

`np.argsort` returns sample positions in distance order. The script takes the first k positions, retrieves their labels, counts them with np.unique, and picks the largest count. The toy examples have distinct distances, so neighbor membership is unambiguous.

### Check the library and discuss ties

The script fits KNeighborsClassifier on the four tiny examples with explicit Euclidean distance, uniform votes, and exact brute-force search. It verifies neighbor indices, distances, and the final predictions for k=1 and k=3. `query.reshape(1,3)` supplies one query row with three features, as required by the batch-oriented prediction API. All checks passed.

Two kinds of ties are different:

- Vote tie: neighbor labels [3,1,2] each have one vote at k=3, despite distinct distances. Odd k does not prevent every multiclass tie. The installed uniform-vote classifier returns the smallest class label, 1, in this case. That is a deterministic convention, not evidence that class 1 is more likely to be correct. The nearest label here is 3, so the convention is not simply choosing the nearest vote.
- Distance tie at the selection boundary: two candidates are equally distant, but only one remaining slot is available. The demonstration uses two examples at distance 1 and k=1; reversing their training order changes the installed classifier's prediction from 7 to 3. This illustrates the documented possibility of training-order dependence. Do not assume every implementation or search method resolves equally distant candidates identically.

Step 5 is complete: arithmetic and library checks passed. The learner calculated distance, interpreted smaller-distance ordering, produced one prediction from neighbor votes, and identified the equal-vote tie for [1,2,3]. The deterministic vote convention and the separate issue of equal distances at the neighbor-selection boundary were discussed. Step 6 will fit and trace a digit classifier using only the saved training and validation partitions.

## References

- [Scikit-learn installation guidance](https://scikit-learn.org/stable/install.html)
- [Python virtual environments](https://docs.python.org/3/library/venv.html)
- [Pip's interpreter-selection option](https://pip.pypa.io/en/stable/topics/python-option/)
- [Dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [NumPy array shape](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html)
- [Distinct values and counts with NumPy](https://numpy.org/doc/stable/reference/generated/numpy.unique.html)
- [NumPy flattening](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html)
- [NumPy reshaping](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)
- [Matplotlib image display](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html)
- [Scikit-learn splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [NumPy array archives](https://numpy.org/doc/stable/reference/generated/numpy.savez.html)
- [Scikit-learn data leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html)
- [Scikit-learn majority-class baseline](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
- [Scikit-learn accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
- [NumPy maximum-position selection](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html)
- [Scikit-learn nearest-neighbor concepts](https://scikit-learn.org/stable/modules/neighbors.html)
- [Scikit-learn KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
