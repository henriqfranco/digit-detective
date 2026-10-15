# Digit Detective progress

Current status: Steps 0–12 and all core learning checkpoints are complete. Test evaluation is complete: k=1 scores 355/360 (98.61%), baseline 36/360 (10.00%). Earlier step notes describe which data was available at those stages; test remained reserved until Step 12. Optional persistence and personal-drawing work have not started.

Step 0 is complete: setup verified and understanding checkpoint discussed. Classification selects discrete categories; a label supplied during prediction would leak the answer and undermine evaluation. Each step includes its acceptance criteria, verification, dependencies, and likely files in [plan.md](plan.md). Work sequentially and complete the explain-back checkpoint before advancing.

Step 1 is complete: inspection verified expected shapes and dtypes, finite values, pixel range 0..16, labels 0..9, counts totaling 1,797, and one corresponding image/feature/label record. Class counts range from 174 to 183. Through a smaller visual example, the learner demonstrated sample/feature indexing, the distinction between X and y, and why image/label pairs must stay aligned.

Step 2 and Checkpoint A are complete: gallery and conversion checks passed and both figures were visually inspected. The learner distinguished 64 features per image from 1,797 samples, explained unchanged brightness through flattening/reshaping, and correctly mapped pixel row 1, column 3 to feature 11. X/y roles and alignment were demonstrated in Step 1.

Step 3 is complete: saved split integrity and reload checks passed. The learner explained why familiar training examples can yield optimistic scores, why settings chosen with test results turn test into validation, why 25% of the 80% remainder yields 20% overall, and why seed 42 provides repeatability rather than guaranteed performance. Only allocation checks used the test partition; no predictions or model evaluation.

Step 4 is complete: manual and library validation predictions agree (37/360, 10.28%), results are saved, and checks passed without changing split files. The learner identified 0% recognition of sevens despite 95% overall accuracy in the imbalanced example, and correctly chose training-majority label 7 despite validation favoring 2. Training labels 1,3,4,5,6 tie at 109 each in the real split, so the lowest-label rule predicts 1. Test prediction/evaluation remains reserved.

Step 5 is complete: tiny-vector checks passed, and the learner calculated squared distance 4, identified smaller distances as nearer, chose a single prediction from repeated neighbor labels, and recognized equal votes for [1,2,3]. The library's lowest-label vote convention and the separate issue of equally distant selection candidates were discussed. No digit partitions or benchmark results were accessed by neighbor_math.py.

Step 6 and Checkpoint B are complete: initial KNN and its saved trace passed verification. The learner identified retained labeled examples as fitted state, k=3 as a chosen hyperparameter, and the need for all 64 image features. Query shapes were clarified: (1,64) means one image with 64 features; (64,1) means 64 samples with one feature each. Validation remains 353/360 (98.06%); split files are unchanged and test prediction/evaluation remains reserved.

Step 7 is complete: comparison and saved selection passed verification; k=1 wins at 356/360 (98.89%, 4 errors). The learner explained why changing validation groups confounds k comparisons, why a small lead does not guarantee future superiority, and that validation results influence k selection while fit still uses training examples only. The frozen fitting policy remains original-training-only, and test evaluation remains reserved.

Step 8 is complete: selected-model analysis is verified, and the learner interpreted true/predicted matrix directions, correctly calculated recall and precision after review, and explained one-neighbor and multi-neighbor votes. Earlier artifacts remain unchanged. Test evaluation remains reserved.

Step 9 and Checkpoint C are complete: viewer.py uses the shared selected-experiment loader and existing trace function; all-validation/mistakes filtering, Previous/Next, arrow keys, and M work. Checks cover wrapping, matching images/IDs, and no mistakes. Correct/mistake snapshots and a live screenshot are saved. WebAgg requires the recorded Tornado dependency; it binds to localhost. The learner explained that navigation uses the existing fitted model; selection and interpretation were covered in earlier checkpoints.

Step 10 is complete: seed-42 nested class-stratified pools contain 109, 268, 537, and 1,077 original training IDs. With fixed k=1 and unchanged clean validation queries, they score 331, 345, 352, and 356 correct out of 360. Saved indices, exact class counts, table, and plot are reproducible. Toy and real checks cover nesting, training-only membership, ordering, class coverage, unchanged inputs/full-model predictions, and saved-index reload. The learner explained why changing two factors confounds the comparison and why more training examples do not guarantee better predictions. Earlier selection/split/result records are unchanged; test evaluation remains reserved.

Step 11 is complete: the frozen full-training k=1 model scores 356/360 clean, 355/354/342 with shared seed-42 Gaussian noise at sigma 1/2/4, and 224/220/250/244 with separate left/right/up/down one-pixel zero-filled shifts. Originals and intended labels are preserved; no transformed data is fitted. Separate tables, exact inputs/predictions, examples, seven deterministic failure traces with neighbors, and galleries are saved. All runnable checks pass; fresh-process results reproduce; figures are visually inspected; prior artifacts remain unchanged. The learner explained how shifting pixels changes neighbors, why originals must remain untouched, and why shifted-image results do not invalidate the clean score. Test evaluation remained reserved through this step.

Step 12 and the final learning checkpoint are complete: final-protocol.json records the original training-only procedure before test prediction; final-test-results.json and test-predictions.npz preserve exact baseline/KNN outcomes. There are five test errors: 8→1 three times, 9→4 once, 9→8 once. Class metrics, both confusion figures, all failure traces/gallery, report.md, and a live viewer screenshot are saved. Validation/Test browsing uses the same fitted model and recorded predictions; test access checks protocol/archive/partition identity. The combined check_final.py suite passes, a fresh-process run reproduces results, figures/live controls are verified, and earlier artifacts are unchanged. The learner identified test-based k adjustment as validation feedback and explained why personal drawings need not match benchmark accuracy; we clarified that new examples can share a distribution while drawing inputs can differ systematically. Earlier checkpoints covered the other ready-to-advance concepts. No test-driven retuning occurred. File organization remains flat after the discarded src reorganization.

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
- [x] 8. Inspect validation confusions and actual mistakes with their neighbors.
- [x] 9. Build and visibly verify the prediction-and-neighbor viewer.
- [x] Checkpoint C: justify selection and explain correct and incorrect predictions.
- [x] 10. Compare fixed-k performance with nested stratified training subsets.
- [x] 11. Evaluate separate noise and zero-filled translation experiments on copies.
- [x] 12. Evaluate the frozen model on test; finish gallery, report, and fresh-process reproduction.
- [x] Final learning checkpoint: meet the ready-to-advance criteria without an accuracy target.

## Optional, after the core workflow

- [ ] Save/reload and verify prediction and neighbor equivalence.
- [ ] Evaluate labeled personal drawings as a separate external challenge.
- [ ] Add a drawing interface after its input-processing contract is understood.
