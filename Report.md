# Week 10 Report — Improving Deep Neural Networks

## 1. Project Overview

The objective of Week 10 was to investigate overfitting in the existing neural network classifier and evaluate several techniques for improving its generalization performance.

The main quest focused on **Improving Deep Neural Networks**, while the side quest focused on **Improve NN I**:

- Diagnose overfitting using training and validation behavior.
- Experiment with stronger regularization.
- Compare different weight initialization methods.
- Evaluate the impact of these changes through controlled experiments.
- Select a final model using scene-based cross-validation.
- Evaluate the selected model on a previously unseen test scene.

The experiments continued the handcrafted-feature classification pipeline developed in previous weeks.

---

## 2. Dataset and Experimental Setup

### 2.1 Dataset

The experiments used the handcrafted feature archive generated during Week 7.

The feature archive contained:

| Feature | Shape | Data type |
|---|---:|---|
| RGB color histogram | 2700 × 4096 | float32 |
| HOG | 2700 × 1764 | float32 |
| Combined features | 2700 × 5860 | float32 |
| Labels | 2700 | int64 |

For this week's neural network experiments, the RGB color histogram features were selected.

The classification task included nine YCB object classes:

1. Master Chef Can
2. Cracker Box
3. Sugar Box
4. Tomato Soup Can
5. Tuna Fish Can
6. Potted Meat Can
7. Pitcher Base
8. Bleach Cleanser
9. Power Drill

### 2.2 Scene-Based Data Split

To evaluate generalization across different scenes, Scene 59 was kept completely separate from model development.

| Dataset | Scenes | Samples |
|---|---|---:|
| Development | 48–58 | 2325 |
| Final test | 59 | 375 |
| Total | 48–59 | 2700 |

The development data was evaluated using Leave-One-Scene-Out Cross-Validation (LOSO-CV).

In each fold:

- One complete scene was used for validation.
- The remaining development scenes were used for training.
- StandardScaler was fitted only on the training portion of that fold.
- The validation scene was transformed using the fitted scaler.

This approach was designed to evaluate generalization to unseen scenes rather than relying on random image-level splits.

**Important:** Scene 59 was excluded from all model selection, hyperparameter comparisons, and regularization experiments. It was used only for the final evaluation after the model had been selected and trained.

### 2.3 Reproducibility

The controlled experiments used:

- Random seed: 42
- Python random seed
- NumPy random seed
- PyTorch random seed
- Fold-specific DataLoader generator seeds
- CPU execution
- Batch size: 64
- Learning rate: 0.001
- Training duration: 15 epochs during cross-validation

The same scene-based folds and preprocessing procedure were used for the controlled comparisons.

---

## 3. Baseline Neural Network

The baseline classifier was a Multi-Layer Perceptron (MLP).

### Architecture

| Layer | Configuration |
|---|---|
| Input | 4096 features |
| Fully connected | 256 neurons |
| Activation | ReLU |
| Dropout | 0.3 |
| Fully connected | 128 neurons |
| Activation | ReLU |
| Dropout | 0.2 |
| Output | 9 neurons |

Total trainable parameters: **1,082,889**

### Training Configuration

| Parameter | Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Baseline weight decay | 0.0001 |
| Loss function | CrossEntropyLoss |
| Epochs | 15 |

The model was trained using the same architecture throughout the regularization and initialization experiments unless otherwise specified.

---

## 4. Baseline Overfitting Diagnosis

The initial baseline experiment achieved:

| Metric | Result |
|---|---:|
| OOF Accuracy | 84.69% |
| OOF Macro-F1 | 79.12% |
| Training time | 114.03 seconds |

However, this initial run did not use the explicit fold-specific DataLoader generator used in the later controlled experiments.

Therefore, it was treated as an initial reference rather than the main comparison baseline.

### 4.1 Controlled Baseline

The baseline was subsequently rerun using the same reproducibility protocol as the other experiments.

| Metric | Controlled Baseline |
|---|---:|
| OOF Accuracy | 83.57% |
| OOF Macro-F1 | 77.49% |
| Training time | 89.78 seconds |

All subsequent experiment comparisons used this controlled baseline.

### 4.2 Evidence of Overfitting

The training curves revealed a substantial difference between training and validation performance.

By the early epochs:

- Training accuracy approached 100%.
- Training loss decreased rapidly toward zero.
- Validation accuracy remained substantially lower.
- Validation loss remained considerably higher and sometimes increased during later epochs.

At epoch 15, the mean gaps across the 11 validation folds were:

| Metric | Controlled Baseline |
|---|---:|
| Training loss | 0.0028 |
| Validation loss | 2.2505 |
| Loss gap | 2.2477 |
| Training accuracy | 99.88% |
| Validation accuracy | 76.45% |
| Accuracy gap | 23.42 percentage points |

These observations indicate that the model fitted the training data very effectively but did not consistently generalize to unseen scenes.

The validation loss was also unstable across folds, suggesting substantial variation in scene difficulty.

---

## 5. Regularization Experiments

Two regularization experiments were conducted to investigate whether stronger regularization could reduce overfitting.

### 5.1 Increased Dropout

The dropout probabilities were increased:

| Layer | Baseline | Increased Dropout |
|---|---:|---:|
| First hidden layer | 0.3 | 0.5 |
| Second hidden layer | 0.2 | 0.4 |

The architecture and other training parameters remained unchanged.

Results:

| Metric | Controlled Baseline | Increased Dropout |
|---|---:|---:|
| OOF Accuracy | 83.57% | 84.17% |
| OOF Macro-F1 | 77.49% | 76.45% |
| Training time | 89.78 s | 89.72 s |

The increased dropout experiment achieved the highest OOF accuracy among the tested configurations.

However, its macro-F1 decreased by approximately 1.04 percentage points compared with the controlled baseline.

At epoch 15, it also achieved smaller overfitting gaps:

| Metric | Controlled Baseline | Increased Dropout |
|---|---:|---:|
| Loss gap | 2.2477 | 1.8123 |
| Accuracy gap | 23.42% | 19.88% |

This suggests that stronger dropout reduced the measured training-validation gap.

Nevertheless, a smaller gap did not translate into better macro-F1 performance.

**Conclusion:** Increased dropout reduced the observed overfitting gap but did not improve the overall macro-F1 score.

### 5.2 Increased Weight Decay

The Adam optimizer's weight decay was increased from:

- Baseline: 0.0001
- Experiment: 0.001

The network architecture and remaining training settings were kept unchanged.

Results:

| Metric | Controlled Baseline | Increased Weight Decay |
|---|---:|---:|
| OOF Accuracy | 83.57% | 84.00% |
| OOF Macro-F1 | 77.49% | 78.21% |
| Training time | 89.78 s | 90.63 s |

The increased weight decay experiment achieved the highest OOF macro-F1 among all tested configurations.

The improvement over the controlled baseline was:

- Accuracy: +0.43 percentage points.
- Macro-F1: +0.72 percentage points.

At selected training epochs, the weight decay experiment also showed improved validation behavior.

At epoch 10:

| Metric | Increased Weight Decay |
|---|---:|
| Training loss | 0.0037 |
| Validation loss | 1.1520 |
| Loss gap | 1.1482 |
| Training accuracy | 99.93% |
| Validation accuracy | 81.32% |
| Accuracy gap | 18.60% |

However, by epoch 15, the loss gap increased to 2.2805 and the accuracy gap increased to 26.57%.

This indicates that the benefit was not consistent throughout training.

**Conclusion:** Increased weight decay provided the best overall cross-validation macro-F1, although the improvement was relatively small and the model still showed evidence of overfitting.

---

## 6. Weight Initialization Experiments

Two initialization strategies were evaluated while keeping the network architecture and training protocol unchanged.

### 6.1 Kaiming Normal Initialization

Kaiming Normal initialization was applied to the linear layers using:

- `mode="fan_in"`
- `nonlinearity="relu"`

Biases were initialized to zero.

Results:

| Metric | Controlled Baseline | Kaiming Normal |
|---|---:|---:|
| OOF Accuracy | 83.57% | 83.35% |
| OOF Macro-F1 | 77.49% | 76.77% |
| Training time | 89.78 s | 94.80 s |

Kaiming initialization did not improve the overall classification performance.

### 6.2 Xavier Uniform Initialization

Xavier Uniform initialization was applied to the linear layers, with biases initialized to zero.

Results:

| Metric | Controlled Baseline | Xavier Uniform |
|---|---:|---:|
| OOF Accuracy | 83.57% | 81.59% |
| OOF Macro-F1 | 77.49% | 75.97% |
| Training time | 89.78 s | 93.59 s |

Xavier Uniform performed worse than the controlled baseline in both metrics.

### 6.3 Initialization Comparison

Neither initialization strategy improved the overall results.

The experiments suggest that changing initialization alone was insufficient to address the generalization limitations observed in this particular network and dataset.

---

## 7. Overall Experiment Comparison

The following table summarizes all five controlled experiments.

| Experiment | OOF Accuracy | OOF Macro-F1 |
|---|---:|---:|
| Increased Weight Decay | 84.00% | 78.21% |
| Controlled Baseline | 83.57% | 77.49% |
| Kaiming Normal | 83.35% | 76.77% |
| Increased Dropout | 84.17% | 76.45% |
| Xavier Uniform | 81.59% | 75.97% |

### Main Observations

1. Increased weight decay achieved the highest OOF macro-F1.
2. Increased dropout achieved the highest OOF accuracy.
3. Kaiming Normal did not improve performance over the controlled baseline.
4. Xavier Uniform produced the weakest overall results.
5. The improvements from regularization were relatively small.
6. Performance varied considerably between validation scenes.

Based on macro-F1, increased weight decay (`weight_decay=0.001`) was selected as the final candidate.

This selection was based exclusively on development cross-validation results.

---

## 8. Training Curve Analysis

Training and validation curves were generated for all five experiments.

The following figures were saved:

- `results/all_experiments_loss_curves.png`
- `results/all_experiments_accuracy_curves.png`

The plots show the mean training and validation metrics across the 11 LOSO folds, with standard deviation bands.

### Observations

- Training loss decreased rapidly across all experiments.
- Training accuracy approached 100% early in training.
- Validation performance remained lower and varied substantially between scenes.
- Increased dropout reduced the average training-validation gap.
- Increased weight decay showed relatively favorable validation behavior around epoch 10.
- Kaiming and Xavier initialization did not consistently improve validation performance.
- The standard deviation bands highlighted substantial differences in scene difficulty.

The results indicate that overfitting remained a challenge even after applying regularization.

---

## 9. Final Model Selection and Training

The increased weight decay experiment was selected because it achieved the highest OOF macro-F1.

### 9.1 Epoch Selection

The best validation epoch was recorded for each of the 11 LOSO folds.

| Validation Scene | Best Epoch |
|---|---:|
| 48 | 13 |
| 49 | 14 |
| 50 | 12 |
| 51 | 15 |
| 52 | 3 |
| 53 | 1 |
| 54 | 6 |
| 55 | 3 |
| 56 | 4 |
| 57 | 2 |
| 58 | 2 |

The median of these selected epochs was **4**.

Therefore, the final model was trained for four epochs on the complete development dataset.

This provided a practical epoch-selection strategy based on the development folds without using the final test scene.

### 9.2 Final Training Configuration

| Parameter | Value |
|---|---:|
| Architecture | Baseline MLP |
| Input features | 4096 |
| Output classes | 9 |
| Training samples | 2325 |
| Epochs | 4 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Weight decay | 0.001 |
| Optimizer | Adam |
| Device | CPU |

### 9.3 Final Training Results

| Epoch | Training Loss | Training Accuracy |
|---|---:|---:|
| 1 | 0.4026 | 89.63% |
| 2 | 0.0174 | 99.48% |
| 3 | 0.0071 | 99.87% |
| 4 | 0.0045 | 99.96% |

The final model, fitted feature scaler, and training history were saved:

- `models/best_model.pth`
- `models/feature_scaler.joblib`
- `results/final_training_history.csv`

The final training accuracy should not be interpreted as an estimate of generalization performance.

---

## 10. Final Evaluation on Scene 59

After model selection and final training were completed, the model was evaluated on Scene 59.

This scene had not been used during training, cross-validation, preprocessing fitting, or model selection.

### 10.1 Overall Performance

| Metric | Result |
|---|---:|
| Test samples | 375 |
| Accuracy | 64.80% |
| Macro-F1 (all 9 classes) | 37.13% |
| Macro-F1 (5 present classes) | 66.83% |

The difference between the two macro-F1 scores is due to the fact that only five of the nine object classes were present in Scene 59.

The all-class macro-F1 assigns zero F1 to the four absent classes under the selected reporting convention.

The present-class macro-F1 provides a more representative summary of performance on the classes actually present in the final test scene.

### 10.2 Per-Class Performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Master Chef Can | 0.0000 | 0.0000 | 0.0000 | 0 |
| Cracker Box | 0.9868 | 1.0000 | 0.9934 | 75 |
| Sugar Box | 0.0000 | 0.0000 | 0.0000 | 0 |
| Tomato Soup Can | 0.8052 | 0.8267 | 0.8158 | 75 |
| Tuna Fish Can | 1.0000 | 0.1733 | 0.2955 | 75 |
| Potted Meat Can | 0.2338 | 0.2400 | 0.2368 | 75 |
| Pitcher Base | 0.0000 | 0.0000 | 0.0000 | 0 |
| Bleach Cleanser | 0.0000 | 0.0000 | 0.0000 | 0 |
| Power Drill | 1.0000 | 1.0000 | 1.0000 | 75 |

### 10.3 Interpretation

The model performed very well on three of the five present classes:

- Cracker Box: 99.34% F1-score.
- Tomato Soup Can: 81.58% F1-score.
- Power Drill: 100% F1-score.

However, performance was substantially weaker on:

- Tuna Fish Can: 29.55% F1-score.
- Potted Meat Can: 23.68% F1-score.

This uneven performance explains why the overall accuracy and macro-F1 differ considerably.

---

## 11. Failure Analysis

The confusion matrix revealed several important classification errors.

The most frequent misclassifications were:

| True Class | Predicted Class | Incorrect Samples |
|---|---|---:|
| Potted Meat Can | Sugar Box | 57 |
| Tuna Fish Can | Potted Meat Can | 47 |
| Tuna Fish Can | Tomato Soup Can | 15 |
| Tomato Soup Can | Potted Meat Can | 12 |
| Tomato Soup Can | Cracker Box | 1 |

### 11.1 Potted Meat Can

The model correctly classified only 18 out of 75 Potted Meat Can samples.

Most of the remaining samples were incorrectly classified as Sugar Box.

This was the largest source of error in the final test set.

### 11.2 Tuna Fish Can

Only 13 out of 75 Tuna Fish Can samples were correctly classified.

The model predicted:

- Potted Meat Can for 47 samples.
- Tomato Soup Can for 15 samples.
- Tuna Fish Can for 13 samples.

Its precision was 100%, but its recall was only 17.33%.

This means that the model's Tuna Fish Can predictions were correct, but it failed to identify most actual Tuna Fish Can samples.

### 11.3 Tomato Soup Can

Tomato Soup Can achieved a comparatively strong result, with 62 correct predictions out of 75.

Most errors involved confusion with Potted Meat Can.

### 11.4 Strongly Recognized Classes

Cracker Box and Power Drill were classified correctly in all 75 samples each.

These results demonstrate that the model could generalize effectively to certain classes in the unseen scene, even though performance remained uneven across classes.

### 11.5 Potential Causes

The observed errors may be related to:

- Similarity between handcrafted color distributions.
- Limited discriminative information in the selected RGB histogram representation.
- Differences in object appearance, illumination, or background between scenes.
- Scene-dependent changes in the feature distribution.

These are possible explanations rather than experimentally confirmed causes.

Further experiments using image-based neural networks, combined RGB-HOG features, or improved feature representations would be needed to investigate them.

---

## 12. Limitations

Several limitations should be considered when interpreting these results.

1. The experiments used handcrafted RGB histogram features rather than an end-to-end image-based neural network.
2. The final test set consisted of a single held-out scene.
3. The model showed substantial variation across validation scenes.
4. The regularization improvements were relatively small.
5. The experiments were conducted with a single main random seed, so the stability of small improvements has not been established through repeated-seed experiments.
6. Scene 59 contained only five of the nine target classes, making the all-class macro-F1 less representative of the classes present in that scene.
7. The final model continued to show a substantial generalization gap between training and unseen-scene performance.

The final test results should therefore be interpreted as performance on this particular held-out scene, not as a guarantee of performance across all possible YCB-Video scenes.

---

## 13. Conclusion

Week 10 focused on diagnosing and reducing overfitting in a neural network classifier using controlled experiments.

The main findings were:

- The baseline MLP showed clear signs of overfitting.
- Increasing dropout reduced the measured training-validation gap but did not improve macro-F1.
- Increasing weight decay to 0.001 produced the best cross-validation macro-F1, reaching 78.21%.
- Kaiming Normal and Xavier Uniform initialization did not improve the baseline results.
- The selected weight-decay model was trained on all 2325 development samples for four epochs.
- The final evaluation on the unseen Scene 59 achieved 64.80% accuracy and 66.83% macro-F1 across the five present classes.
- The main classification weaknesses were Potted Meat Can and Tuna Fish Can.

The difference between cross-validation and final test performance highlights the difficulty of generalizing across scenes.

Although increased weight decay provided a modest improvement during development, it did not eliminate overfitting or guarantee strong performance on the unseen scene.

**Overall, this week's experiments demonstrated that regularization must be evaluated through scene-based validation and independent testing, rather than relying only on training accuracy or a smaller training-validation gap.**

---

## 14. Generated Artifacts

The following files were generated during the experiments.

### Models

- `models/best_model.pth`
- `models/feature_scaler.joblib`

### Training and Evaluation Results

- `results/experiment_comparison.csv`
- `results/per_scene_weight_decay_comparison.csv`
- `results/overfitting_gap_analysis.csv`
- `results/final_overfitting_comparison.csv`
- `results/weight_decay_best_epochs.csv`
- `results/final_training_history.csv`
- `results/scene59_predictions.csv`
- `results/scene59_classification_report.csv`
- `results/scene59_misclassifications.csv`

### Visualizations

- `results/baseline_loss_curves.png`
- `results/baseline_accuracy_curves.png`
- `results/training_curves_loss.png`
- `results/training_curves_accuracy.png`
- `results/all_experiments_loss_curves.png`
- `results/all_experiments_accuracy_curves.png`
- `results/scene59_confusion_matrix.png`

---

**Week 10 Status: Completed**

The regularization and initialization experiments, overfitting analysis, final model selection, and independent Scene 59 evaluation were completed.
