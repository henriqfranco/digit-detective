# Digit Detective progress

Step 0 is complete: setup verified and understanding checkpoint discussed. Classification selects discrete categories; a label supplied during prediction would leak the answer and undermine evaluation. Each step includes its acceptance criteria, verification, dependencies, and likely files in [plan.md](plan.md). Work sequentially and complete the explain-back checkpoint before advancing.

Step 1 is complete: inspection verified expected shapes and dtypes, finite values, pixel range 0..16, labels 0..9, counts totaling 1,797, and one corresponding image/feature/label record. Class counts range from 174 to 183. Through a smaller visual example, the learner demonstrated sample/feature indexing, the distinction between X and y, and why image/label pairs must stay aligned.

Step 2 and Checkpoint A are complete: gallery and conversion checks passed and both figures were visually inspected. The learner distinguished 64 features per image from 1,797 samples, explained unchanged brightness through flattening/reshaping, and correctly mapped pixel row 1, column 3 to feature 11. X/y roles and alignment were demonstrated in Step 1.

Step 3 is complete: saved split integrity and reload checks passed. The learner explained why familiar training examples can yield optimistic scores, why settings chosen with test results turn test into validation, why 25% of the 80% remainder yields 20% overall, and why seed 42 provides repeatability rather than guaranteed performance. Only allocation checks used the test partition; no predictions or model evaluation.

Step 4 is complete: manual and library validation predictions agree (37/360, 10.28%), results are saved, and checks passed without changing split files. The learner identified 0% recognition of sevens despite 95% overall accuracy in the imbalanced example, and correctly chose training-majority label 7 despite validation favoring 2. Training labels 1,3,4,5,6 tie at 109 each in the real split, so the lowest-label rule predicts 1. Test prediction/evaluation remains reserved.

Step 5 is complete: tiny-vector checks passed, and the learner calculated squared distance 4, identified smaller distances as nearer, chose a single prediction from repeated neighbor labels, and recognized equal votes for [1,2,3]. The library's lowest-label vote convention and the separate issue of equally distant selection candidates were discussed. No digit partitions or benchmark results were accessed by neighbor_math.py.

Step 6 and Checkpoint B are complete: initial KNN and its saved trace passed verification. The learner identified retained labeled examples as fitted state, k=3 as a chosen hyperparameter, and the need for all 64 image features. Query shapes were clarified: (1,64) means one image with 64 features; (64,1) means 64 samples with one feature each. Validation remains 353/360 (98.06%); split files are unchanged and test prediction/evaluation remains reserved.

Step 7 is complete: comparison and saved selection passed verification; k=1 wins at 356/360 (98.89%, 4 errors). The learner explained why changing validation groups confounds k comparisons, why a small lead does not guarantee future superiority, and that validation results influence k selection while fit still uses training examples only. The frozen fitting policy remains original-training-only, and test evaluation remains reserved.

- [x] 0. Define the prediction task and verify a small reproducible Python environment.
- [x] 1. Inspect shapes, dtypes, ranges, labels, alignment, and class counts.
- [x] 2. Display every class and verify the image/vector round trip.
- [x] Checkpoint A: explain data representation and alignment.
- [x] 3. Save fixed stratified train/validation/test indices and verify integrity.
- [x] 4. Calculate and evaluate the training-majority baseline on validation.
- [x] 5. Derive toy distances and votes manually, then verify them in code.
- [x] 6. Fit initial KNN and trace a validation prediction to training neighbors.
- [x] Checkpoint B: explain data roles, distances, votes, and fitted state.
- [x] 7. Compare k=1,3,5,9 fairly and freeze the selected configuration.
- [ ] 8. Inspect validation confusions and actual mistakes with their neighbors.
- [ ] 9. Build and visibly verify the prediction-and-neighbor viewer.
- [ ] Checkpoint C: justify selection and explain correct and incorrect predictions.
- [ ] 10. Compare fixed-k performance with nested stratified training subsets.
- [ ] 11. Evaluate separate noise and zero-filled translation experiments on copies.
- [ ] 12. Evaluate the frozen model on test; finish gallery, report, and fresh-process reproduction.
- [ ] Final learning checkpoint: meet the ready-to-advance criteria without an accuracy target.

## Optional, after the core workflow

- [ ] Save/reload and verify prediction and neighbor equivalence.
- [ ] Evaluate labeled personal drawings as a separate external challenge.
- [ ] Add a drawing interface after its input-processing contract is understood.
