# Digit Detective — final experiment

**Question and data.** Predict a handwritten digit from scikit-learn's `load_digits`: 1,797 labeled 8×8 grayscale images, ten classes, intensities 0–16. This is not MNIST. `X` has shape (1797,64); each row contains an image's pixels in row-major order. `y[i]` is that image's digit. Flattening preserves brightness values and their order.

**Evaluation design.** Seed-42 stratified splits contain 1,077 training, 360 validation, and 360 test images. Saved indices keep images and labels aligned. Training supplies examples; validation chooses settings; test measures the fixed procedure afterward. The final fit uses only the original training partition. No scaler was added: pixels already share a brightness scale. [Recorded protocol](artifacts/final-protocol.json).

**Mechanism and selection.** KNN retains labeled training vectors. Euclidean distance is the square root of the sum of squared feature differences: differences (−1,−2,0) give √5. The closest k labels vote; k is a chosen hyperparameter. Exact brute-force search and uniform votes were fixed. For k=1,3,5,9, validation correct counts were 356,353,353,351 out of 360. The declared rule selected highest accuracy, then smaller k on an exact tie: **k=1**, 98.89%. The baseline always predicts training-majority label 1; tied training counts use the lowest label.

| Final test procedure | Correct | Errors | Accuracy |
| --- | ---: | ---: | ---: |
| Majority baseline: always 1 | 36/360 | 324 | 10.00% |
| Selected KNN: k=1 | 355/360 | 5 | 98.61% |

**Mistakes and explanation.** Three actual eights were called ones; one nine was called four and another eight. Recall was 32/35=91.43% for eight, 34/36=94.44% for nine, and 100% for the remaining classes. For example, test sample 1553 (eight) copies training sample 1178's label one at distance 26.12. These neighbor traces explain the arithmetic, not human understanding of handwriting. [All five failures and one success](artifacts/test-failure-gallery.png), [confusion matrix](artifacts/test-confusion-counts.png), [class metrics](artifacts/test-per-class.csv).

**Diagnostics and limits.** With fixed k=1, nested training pools of 109/268/537/1,077 scored 331/345/352/356 correct on unchanged validation images. Gaussian noise at strengths 1/2/4 scored 355/354/342; independent one-pixel shifts scored 61.11–69.44%. These were separate diagnostics, without retuning. One shift is 1/8 of an image dimension; raw pixel matching is sensitive to alignment and cropping. A single split, subset sequence, and noise realization do not establish universal performance. Samples were held out, but writer independence was not established. Personal drawings can differ in inversion, centering, stroke width, and resizing. Test results were inspected only after freezing the procedure; subsequent tuning would require a new study.

**Reproduce.** Use Python 3.14.4, NumPy 2.5.3, scikit-learn 1.9.1, and Matplotlib 3.11.2; full pins are in `requirements.txt`. From the project root, with the environment prepared as in [README](README.md):

```sh
.venv/bin/python final_evaluate.py
.venv/bin/python check_final.py
.venv/bin/python viewer.py
```

The evaluator verifies the frozen protocol and reproduces exact test predictions. The viewer labels validation/test collections and shows their voting training neighbors. [Full diagnostic commands and explanations](README.md).
