# Digit Detective: learning and implementation plan

Status: Steps 0 through 4 and Checkpoint A are complete. The baseline is verified at 37/360 validation accuracy (10.28%), and the learner explained training-derived prediction choice and how class imbalance can hide failure on another class. Step 5 is next; KNN fitting, hyperparameter comparison, and test evaluation have not started.

## Objective and learning contract

Build a program that displays a handwritten digit, predicts its class, and shows the training neighbors whose votes produced the prediction. Finish with a failure gallery and a one-page explanation. The purpose is understanding the complete machine-learning workflow, not reaching an accuracy target.

We will work sequentially, one small lesson and implementation slice at a time. Before each slice: explain the question, define new terms, predict the outcome, and justify the decision. Then write a small amount of code, run it, inspect the actual output, and explain it back in ordinary language. A working result without an explanation is an unfinished learning step. An unexpected result becomes something to investigate, not something to hide.

For each decision, distinguish:

- A requirement of this problem, such as keeping labels aligned with images.
- A scientific safeguard, such as holding out a test partition.
- A reasonable starting choice, such as a 60/20/20 split.
- A convenience, such as the numeric seed 42.
- A hypothesis to test, such as whether a larger k improves validation accuracy.

The plan is deliberately sequential. Outsourcing the reasoning or implementing the entire project in advance would defeat the learning objective.

## Starting decisions

1. Use Python, NumPy, Matplotlib, and scikit-learn on CPU. Introduce Python syntax when needed; no preliminary Python course.
2. Start with ordinary Python scripts and saved plots. A notebook is an alternative presentation format, not a prerequisite. Prefer a Matplotlib viewer with a sample selector or previous/next controls, using the existing plotting dependency.
3. Use scikit-learn's load_digits dataset, not MNIST. Its documented structure is 1,797 images of 8x8 pixels, 64 features, ten classes, and integer-valued intensities from 0 to 16. Inspect actual array dtypes rather than assuming integer storage.
4. Start with raw pixels, Euclidean distance, and equal neighbor votes. State these settings explicitly. Do not mechanically add standardization, PCA, augmentation, or distance weighting.
5. Use a fixed stratified 60% training / 20% validation / 20% test split. Save original dataset indices, the seed, and relevant version information. The default seed is 42; the value has no special statistical merit.
6. Compare exactly k=1, 3, 5, and 9 on the same validation examples. Choose highest clean validation accuracy; break exact score ties with the smaller k. Record this rule before observing scores.
7. Keep the chosen model fitted on the original training partition for the primary final test. This keeps the validation and test evaluation tied to the same fitted model. A train-plus-validation refit is optional later, must be decided before seeing test results, and must be reported as a different training procedure.
8. Treat robustness and reduced-training experiments as diagnostics of the fixed chosen configuration. Do not repeatedly retune on their outcomes during the primary study.
9. During development, the viewer and mistake gallery use validation examples only. Unlock test examples after the final configuration is frozen and the final evaluation is complete.

## Step 0: frame the problem and prepare a small environment

**Question:** What precisely is the program trying to predict?

**Learn:** Supervised learning means using examples paired with known answers. A sample is one image; a feature is one numerical input; a label is its intended digit. This is multiclass classification: choose one of ten discrete categories. The label 8 is a class identifier, not a numerical quantity whose error should be measured by subtraction from 7.

**Build:** Verify the available Python environment, create a project-local environment if needed, install only the four requested packages, and record tested versions. Create one runnable inspection script. Explain imports, indexing, function calls, and array operations only when first used.

**Why:** An isolated environment makes dependencies reproducible and avoids modifying unrelated projects. A script provides an explicit execution order; hidden notebook state is avoidable. A model need not understand digits symbolically to learn useful statistical patterns.

**Output:** A successful import/load check and a short statement of inputs, outputs, and scope.

**Verification:** Run the script from a fresh process and confirm imports and dataset loading work. No accuracy measurement yet.

**Explain-back checkpoint:** What is one training example? Why is this classification rather than regression? What does the label tell us that the pixels do not?

**Dependencies / likely files:** None / a requirements file and one inspection script; small scope.

## Step 1: inspect the dataset and define X and y

**Question:** What does the loaded data actually contain?

**Learn:** Array axes, shape, dtype, indexing, sample count, feature count, and class frequency. Conventionally X contains inputs and y contains targets. Their row order must agree: X[i] and y[i] describe the same sample.

**Build:** Load the default ten-class dataset; print the shapes of images, data, and target; inspect min/max values, dtypes, distinct labels, missing/nonfinite values, and class counts. Show one sample's image array, feature row, and label. Count labels with NumPy rather than adding a dataframe dependency.

**Expected structure:** images=(1797, 8, 8), X=(1797, 64), y=(1797,). These are expectations to verify, not substitutes for inspecting the output.

**Why:** Many ML bugs are alignment or shape bugs. Nearly balanced classes help interpret accuracy but do not mean class counts are exactly equal. Dataset-wide descriptive counts are allowed; choosing a model or fitting preprocessing from held-out outcomes is not.

**Output:** Dataset summary and class-count table.

**Verification:** Counts sum to 1,797; arrays have equal sample counts; labels are 0 through 9; intensities are finite and in the expected range.

**Explain-back checkpoint:** In X.shape, which axis represents images? What does X[12, 20] mean? Why does randomly shuffling X without shuffling y break training?

**Dependencies / likely files:** Step 0 / the inspection script; small scope.

## Step 2: see every class and trace image-to-vector conversion

**Question:** How does an image become numerical model input?

**Learn:** An 8x8 image is a two-dimensional array; flattening creates a 64-element row. It preserves pixel values and their ordered positions but does not explicitly encode two-dimensional neighborhood relationships. Reshape is not resizing or extracting new features.

**Build:** Display examples of all ten classes, preferably several per class to reveal handwriting variation. Trace one pixel at row r, column c to feature index 8*r+c using zero-based, row-major indexing. Show the numeric image, flattened vector, and reconstructed image.

**Why:** Every pixel position must mean the same thing in every sample. Fixed display limits 0 and 16 prevent separate plots from silently using different brightness scales. Nearest-pixel display interpolation avoids visually inventing smooth detail in an 8x8 image.

**Output:** Class gallery and an image/vector trace figure.

**Verification:** Flattened images match corresponding rows of the supplied data matrix; reconstructing the vector produces the original image exactly.

**Explain-back checkpoint:** Why 64 features rather than 1,797? Did flattening discard pixel values? What image structure does ordinary Euclidean comparison fail to explicitly model?

**Dependencies / likely files:** Step 1 / the inspection script and generated plots; small scope.

## Checkpoint A: data understanding

Explain X, y, all array shapes, image/label alignment, and the image-to-vector mapping without relying on memorized code.

## Step 3: create and preserve train/validation/test partitions

**Question:** How do we find out whether the model works on examples it did not fit on?

**Learn:** Training data is used by fit; validation data supports development choices; test data estimates performance after those choices are fixed. Generalization is performance on unseen examples from the relevant population. Stratification preserves class proportions approximately; shuffling removes dependence on dataset order.

**Build:** Split original dataset indices, not images and labels independently. Reserve 20% for test first. Split the remaining 80% so 25% of that remainder becomes validation, yielding approximately 60/20/20 overall. Stratify both operations with the appropriate labels and fixed seed. Expected counts for this procedure are 1,077 training, 360 validation, and 360 test; verify them when implemented. Save indices in a NumPy archive and metadata in a small JSON file. Later runs reload the saved split.

**Why:** Training needs most examples, while validation and test need enough cases from each class to be useful. These ratios are a starting tradeoff, not a universal optimum. Saving indices preserves membership, ordering, and neighbor provenance more concretely than saving a seed alone. A fixed seed makes an experiment repeatable; it does not establish that one split is representative.

**Rules:** After allocation checks, do not display individual test images, score test predictions, inspect test errors, or use test labels to guide development. Basic overlap/count/class checks do not constitute model evaluation. Dataset exploration already described the full dataset; record that honestly and do no targeted test analysis thereafter.

**Leakage lesson:** Leakage includes fitting a scaler or feature selector on the entire dataset before splitting. If learned preprocessing is later tested, fit it on training only and apply that fitted transformation unchanged elsewhere. A random sample split evaluates held-out samples, not necessarily unseen writers; do not claim writer independence without writer IDs and a group-aware split.

**Output:** Saved split indices, split metadata, and partition/class-count table.

**Verification:** Pairwise-disjoint index sets; complete coverage; no duplicate indices; correct counts; all classes represented; corresponding images/features/labels remain aligned; reload returns the same indices.

**Explain-back checkpoint:** Why would test-based tuning corrupt the evaluation? Why is validation 25% of the remainder? Is seed 42 better than another seed? Why is sample separation not automatically writer separation?

**Dependencies / likely files:** Step 2 / experiment script and saved split artifacts; small scope.

## Step 4: build the majority-class baseline

**Question:** How much performance can we get without inspecting the image?

**Learn:** A baseline is a simple reference. Accuracy is correct predictions divided by total predictions. A majority-class rule always predicts the most common training label. Baseline choice uses training labels, not the most frequent validation or test label.

**Build:** Count training labels and calculate the rule manually, then compare it with DummyClassifier(strategy='most_frequent'). Evaluate on validation; record correct/total as well as percentage.

**Why:** A sophisticated model must justify added work relative to an inexpensive rule. On a roughly balanced ten-class dataset, around 10% is a useful expectation, not a guaranteed measured result. High accuracy may be misleading in an imbalanced problem; inspect the distribution and which classes the rule neglects.

**Output:** Baseline class and validation result in the experiment table.

**Verification:** All predictions equal the training-majority class; manual accuracy matches library accuracy.

**Explain-back checkpoint:** Is the most frequent training label learned from data? Why not select it from validation labels? What would this baseline achieve on a dataset with 95% of examples in one class?

**Dependencies / likely files:** Step 3 / experiment script and metrics output; small scope.

## Step 5: derive distance and voting by hand

**Question:** What exactly makes one example a neighbor?

**Learn:** Vectors, componentwise subtraction, squares, summation, square root, sorting, plurality voting, and ties.

For q=(1,2,3) and a=(2,4,3):

    q-a=(-1,-2,0)
    squared distance=1+4+0=5
    Euclidean distance=sqrt(5)

In general d(q,a)=sqrt(sum_j((q_j-a_j)^2)). Squared distances give the same neighbor ordering because square root is increasing on nonnegative values. They are not numerically the same distances and are not interchangeable in distance-weighted voting.

Use a tiny labeled training table:

| Vector | Label | Squared distance to q |
| --- | --- | --- |
| (1,2,4) | 7 | 1 |
| (1,4,3) | 3 | 4 |
| (4,2,3) | 3 | 9 |
| (1,2,7) | 7 | 16 |

At k=1 the prediction is 7. At k=3 the nearest labels are 7,3,3 and the prediction is 3. Calculate this on paper before expressing it with NumPy and checking the library.

**Why:** The model's prediction should follow understandable arithmetic. Equal voting chooses the class with the most votes; in multiclass problems this may be a plurality rather than more than half the votes. Odd k does not prevent every multiclass tie: labels 1,2,3 tie at k=3. Distinguish a tied vote from tied neighbor distances at the k boundary. Inspect and document the installed library's tie behavior rather than adding a private conflicting voting rule.

**Output:** Worked arithmetic and a tiny runnable check.

**Verification:** Manual and NumPy distances agree; library predictions on the untied toy example are 7 and 3 respectively; tie examples are explicitly discussed.

**Explain-back checkpoint:** Why square before summing? Why can the nearest example lose the vote? Does being visually similar guarantee being close in raw pixel distance?

**Dependencies / likely files:** Step 4 / one small math/check script; small scope.

## Step 6: fit the first digit classifier and trace inference

**Question:** What happens during fit, and what happens during prediction?

**Learn:** Estimator API, training versus inference, single-sample batch shape (1,64), non-parametric instance-based classification, fitted state versus hyperparameters, and feature-space assumptions.

**Build:** Fit KNeighborsClassifier on training X/y with provisional k=3, metric='euclidean', weights='uniform', and algorithm='brute' for an explicit exact-search starting point. Predict one validation image. Retrieve actual selected neighbors and distances; map returned training-row indices back to original dataset indices; print neighbor labels and vote counts. Compute validation accuracy and compare to baseline.

**Why:** k=3 is a starting illustration, not the final choice. Exact brute-force search is adequate for this dataset and easy to reason about; no tree-search optimization is needed. KNN primarily stores examples and labels instead of optimizing a small coefficient vector. This is still a fitted model: its state changes with training examples. k, distance, and vote weighting are chosen hyperparameters. Prediction uses pixels, not the query's true label; that label is consulted only to evaluate the answer.

**Scaling lesson:** All pixel positions share an intensity scale. A common positive scaling of every coordinate preserves uniform-vote neighbor ordering; per-feature standardization changes the distance geometry. It could help or hurt and needs evidence. It is not a required ritual.

**Output:** One complete prediction trace and first KNN validation score.

**Verification:** Neighbor IDs belong to training only; labels align with the stored training order; distances agree with NumPy within floating-point tolerance; the shown votes account for the library prediction. Explain any tie rather than concealing it.

**Explain-back checkpoint:** What has KNN learned or retained? What is chosen rather than learned? Why can evaluating a stored training sample, especially at k=1, be misleading? Why does predict need a two-dimensional batch?

**Dependencies / likely files:** Step 5 / experiment script; small scope.

## Checkpoint B: honest evaluation and prediction mechanics

Explain the three data roles, reproduce a toy vote, distinguish fitted state from hyperparameters, and trace a validation prediction back to its training examples.

## Step 7: compare k fairly and freeze the selected setting

**Question:** Which neighborhood size works best under our stated selection rule?

**Learn:** Hyperparameter selection, controlled comparisons, validation overfitting, bias/variance intuition, and the distinction between a hypothesis and an observed result.

**Build:** Fit k=1,3,5,9 on identical training indices with all other choices fixed. Measure clean validation accuracy, error count, and optionally training accuracy as a diagnostic. Save the result table. Select highest validation accuracy, then smaller k on an exact tie. Record the winning configuration and original-training-only final-fit policy before accessing test results.

**Why:** Changing one factor makes results interpretable. Smaller k can be sensitive to individual examples; larger k can smooth over useful local distinctions. These are tendencies, not a guaranteed monotonic accuracy curve. Training score does not replace held-out evaluation. One additional correct prediction among 360 validation cases changes accuracy by about 0.28 percentage points; small differences may reflect sampling variation.

**Output:** Four-row comparison table and written selection rationale.

**Verification:** All runs share exactly the same partitions, feature representation, metric, and weighting; selected k follows the declared rule; no test metric has been produced.

**Explain-back checkpoint:** Why not try thousands of settings? How can repeated tuning overfit validation? What can and cannot be concluded when one setting wins by one example?

**Dependencies / likely files:** Step 6 / experiment script and metrics output; small scope.

## Step 8: inspect confusion matrices and build the failure gallery

**Question:** Which kinds of mistakes does overall accuracy hide?

**Learn:** True versus predicted class, confusion matrix axes, per-class support, recall, precision, and observations versus explanations. Introduce precision and recall through one actual class's counts before discussing additional metrics.

**Build:** Produce validation confusion matrices with explicit class order 0..9: raw counts and row-normalized fractions. Find incorrect predictions with y_pred != y_true. Inspect six or more distinct failures if available; if fewer exist, inspect all and report the actual count. Include the query, true/predicted labels, training-neighbor thumbnails, distances, and vote tally. Include successful cases for comparison. Selection is deterministic or explicitly documented, not cherry-picked to flatter the model.

**Why:** Rows represent actual classes, columns predicted classes. Cell [3,8] counts true threes predicted as eights. Row normalization reveals the fraction of each actual class going to each prediction; overall percentages can hide class-specific weaknesses. A story such as 'this stroke looks like an eight' is a visual hypothesis, not proof of a causal mechanism. The selected neighbor votes are the direct mechanism.

**Output:** Confusion plots, per-class counts/recall table, and validation failure gallery.

**Verification:** Matrix total equals validation sample count; diagonal total equals correct predictions; off-diagonal total equals errors; each displayed query is genuinely wrong for the stated run; every displayed neighbor is from that model's fitted training pool. If no errors occur, state that rather than inventing mistakes. Separate robustness failures from clean-data failures.

**Explain-back checkpoint:** Read a matrix cell, calculate one class's recall and precision, identify the main observed confusion, and distinguish a vote explanation from a visual hypothesis.

**Dependencies / likely files:** Step 7 / experiment script, plotting helpers only if reuse warrants them, and generated plots; small scope.

## Step 9: turn the trace into a prediction-and-neighbor viewer

**Question:** Can someone inspect a prediction without reading the implementation?

**Build:** Add a simple Matplotlib sample selector or previous/next controls. Display the query image, true label, predicted label, correctness, selected training neighbors, neighbor labels/distances/original IDs, and vote counts. Offer all-validation and errors-only browsing. Display the exact k used in the prediction. Label extra contextual neighbors separately if ever added; only selected voting neighbors count toward the explanation.

**Why:** This viewer is an inspection tool, not a second prediction algorithm. Reuse the same fitted model, sample identity mapping, and plotting routines. A vote share such as 2/3 is neighborhood agreement, not a calibrated probability that the answer is correct. Avoid a 'confidence' label that promises more than the calculation supports.

**Output:** Usable viewer plus saved example figures for the report.

**Verification:** Browse multiple samples and errors; confirm labels, votes, distances, and IDs update together; handle an empty errors-only collection; verify a usable visible plot window. Development browsing remains validation-only.

**Explain-back checkpoint:** Trace an unfamiliar displayed prediction from feature vector to selected neighbors to final vote.

**Dependencies / likely files:** Step 8 / viewer script and existing experiment functionality; small scope. Reuse through a small shared function/module only when two callers need it.

## Checkpoint C: model selection and interpretation

Justify selected k, interpret main confusions, and explain both a correct prediction and a mistake using actual neighbors. Confirm the test partition remains unevaluated.

## Step 10: measure the effect of fewer training examples

**Question:** How much does performance depend on the number of examples available?

**Learn:** Learning curves, data coverage, controlled subsampling, and the limits of a single realization.

**Build:** Keep chosen k fixed. Use approximately 10%,25%,50%,100% of the training partition, with exact sample/class counts reported. Shuffle indices within each class once using a fixed seed, then take nested fractions so smaller pools are contained in larger ones. Fit a separate model for each pool and evaluate on the unchanged clean validation partition. Preserve the full-training chosen model for the final evaluation. Plot accuracy versus actual training count.

**Why:** Taking the first N dataset rows can distort class composition. Stratified nested sampling reduces changing composition and membership as confounders. It does not make one curve a statistical guarantee. A smaller sample may occasionally score better; investigate instead of forcing a monotonic plot. Do not retune k separately at each size in this first controlled experiment.

**Output:** Training-size table and simple learning-curve plot.

**Verification:** Every subset is training-only; nesting and class coverage hold; sizes exceed chosen k; validation membership is unchanged; final model still uses the full original training pool.

**Explain-back checkpoint:** Why hold k fixed? Does a curve prove that more data always helps? What changed and what remained controlled?

**Dependencies / likely files:** Step 9 / experiment script and generated results; small scope.

## Step 11: test robustness on separate corrupted copies

**Question:** What happens when input pixels change without changing the intended digit?

**Learn:** Controlled perturbation, noise, clipping, translations, distribution shift, and the limits of label-preservation assumptions.

**Build, experiment A:** Make float copies of validation images; add zero-mean Gaussian noise at standard deviations 1,2,4 in the native 0..16 units, with a fixed randomness seed. Clip to 0..16. Reuse one standardized noise array across severities to make the comparison paired. Report accuracy at each level with the clean score alongside it.

**Build, experiment B:** Independently shift clean images one pixel left, right, up, or down; fill exposed pixels with zero and discard pixels that leave the frame. Do not wrap pixels around to the opposite edge. Do not combine translation and noise in this first study.

**Why:** One pixel is one-eighth of an image dimension, a substantial perturbation at this resolution. Translation changes pixel positions, so a semantically similar image may become distant under raw pixel distance. Zero filling matches this dataset's background scale. Severe noise or cropping may destroy legibility, so unchanged labels encode our intended transformation, not a guarantee that every corrupted image remains unambiguous.

**Output:** Separate noise/translation result tables, before/after examples, and clearly labeled robustness failures with neighbors.

**Verification:** Original validation arrays are unchanged; shapes and labels remain aligned; pixels are finite/in-range; shifts have zero fill rather than wrapping; model is not refitted on validation; repeating the experiment reproduces corruptions and scores.

**Explain-back checkpoint:** Why is this not the original clean validation evaluation? Why is a one-pixel shift large here? Does poor corruption performance prove poor clean benchmark performance? Would validation-based augmentation be a training-data leak?

**Dependencies / likely files:** Step 10 / experiment script, tiny corruption helpers if needed, and results; small scope.

## Step 12: evaluate once on test and write the one-page explanation

**Question:** How well does the frozen procedure perform on the reserved examples?

**Before accessing test predictions:** Save the final k, raw-pixel representation, Euclidean distance, uniform voting, search setting, original training pool, baseline, split identity, selection rule, and experimental version. Confirm no preprocessing or robustness change has silently replaced the selected procedure.

**Build:** Evaluate the original-training-fitted baseline and chosen KNN on test. Save correct/total, accuracy, class supports, confusion matrix, and per-class recall. Only after metrics are fixed, inspect test mistakes and neighbors, clearly separated from development failures. No new k selection follows from test outcomes. If results prompt a later redesign, mark that future work as a new study; the already-inspected test set cannot regain untouched status.

**Why:** The held-out evaluation estimates performance for this data and split, not all handwriting or all writers. Report error counts and denominators so small score changes remain interpretable. Re-running the frozen procedure for reproducibility is fine; tuning it after examining test results is the problem.

**One-page explanation:** Problem and dataset; X/y representation; split and seed; majority baseline; manual-to-library KNN mechanism; four-k validation comparison and choice; final test result; main confusions; reduced-data and robustness observations; limitations; exact reproduction instructions and recorded versions. Fill measured numbers only after running the experiments. Link larger tables/galleries rather than forcing them onto the page.

**Output:** Final result artifacts, final failure gallery, viewer with labeled validation/test browsing, and concise report.

**Verification:** Rerun the frozen experiment in a fresh process with saved indices and seeds; confirm matching predictions/results in the recorded environment; verify report numbers match artifacts and gallery entries; ensure the viewer exposes the same final model's neighbors. One small runnable check should cover split integrity, toy arithmetic, image/vector mapping, and corruption invariants; avoid an elaborate test framework.

**Explain-back checkpoint:** Explain why the test result is more appropriate for final reporting than the best validation score, while also explaining its sampling and distribution limits.

**Dependencies / likely files:** Step 11 / experiment/viewer scripts, results, one small check, and report; medium scope at most.

## Optional Step 13: persistence and personal drawings

Begin only after the core workflow and explanation are understood.

**Save/reload:** Preserve the fitted model and versions, reload it, and check identical predictions and neighbor identity. Only load trusted serialized model artifacts. Retain training examples or their recoverable IDs so explanations remain possible.

**Drawing challenge:** First ingest a few labeled personal drawings as a separate external challenge set; only then build a drawing interface if useful. Specify grayscale conversion, foreground/background polarity, centering, stroke thickness, aspect ratio/padding, 8x8 resizing, and mapping into the 0..16 range. Preview the original, processed image, and vector. Personal handwriting is a different distribution, not another draw from the benchmark split. If adapting preprocessing on these drawings, separate development drawings from fresh evaluation drawings and avoid claiming benchmark accuracy transfers.

**Why optional:** Saving is useful for reuse, and drawing is useful for investigating deployment inputs. Neither is necessary to understand the first modeling workflow. A polished drawing UI can obscure an input mismatch, so it comes last.

**Verification:** Reload equivalence for persistence; shape/range/polarity checks and labeled external results for drawings.

## Minimal project structure, added only as needed

- tasks/plan.md and tasks/todo.md: this roadmap and progress record.
- A dependency record and short README with the actual commands once implemented.
- An inspection script: dataset, class gallery, image/vector trace.
- An experiment script: saved splits, baseline, k comparison, diagnostics, final evaluation.
- A viewer script: interactive prediction and neighbor display.
- One small runnable correctness check.
- An artifacts directory for splits, metadata, result tables, and figures.
- A one-page report.

Names and factoring are implementation conveniences, not a fixed architecture. Extract shared functionality only when actual reuse appears. No service layer, API, database, web framework, GPU pipeline, large hyperparameter-search system, or experiment-tracking platform is needed.

## Risks and safeguards

| Risk | Safeguard |
| --- | --- |
| Producing working code without understanding | Explain-back checkpoints and manual calculations before library use |
| Test contamination | Validation-only development and explicit final configuration before test evaluation |
| Image/label or neighbor ID mismatch | Split indices once and maintain mappings to original dataset rows |
| Misleading high training score | Report held-out validation and test performance |
| Overinterpreting tiny score differences | Show correct/total and discuss finite sample size |
| Cherry-picking failure explanations | Document deterministic selection and report all failures if few exist |
| Mutating the original validation data | Corrupt copies and assert original data remains unchanged |
| Translation wraps pixels | Explicit zero-filled shifts with a runnable invariant check |
| Treating neighbor agreement as certainty | Display votes; avoid uncalibrated confidence claims |
| Making unsupported real-world claims | Separate benchmark, corruption, and personal-drawing evaluations |

## Ready-to-advance criteria

You can explain X and y; interpret shapes; trace a pixel and prediction; justify the split; calculate a tiny distance and vote; distinguish fitted state from chosen hyperparameters; justify k from validation; read main confusions; explain data-volume and robustness experiments; reproduce the final experiment; and describe why benchmark performance does not guarantee performance on drawings. No particular accuracy threshold is required.

## Official references

- Dataset structure: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
- Fixed and stratified splitting: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- Majority-class baseline: https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
- Neighbors, voting, and distance ties: https://scikit-learn.org/stable/modules/neighbors.html
- Confusion matrix definition: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
- Leakage and preprocessing: https://scikit-learn.org/stable/common_pitfalls.html
