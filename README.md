# ML Internship — Week 10
## Improving Deep Neural Networks

This project investigates overfitting and generalization in a Multi-Layer Perceptron (MLP) classifier using handcrafted features extracted from the YCB-Video dataset.

The main objective was to evaluate regularization and weight initialization techniques through controlled experiments and scene-based cross-validation.

## Project Objectives

- Diagnose overfitting using training and validation curves.
- Evaluate stronger dropout regularization.
- Investigate the effect of increased weight decay.
- Compare Kaiming Normal and Xavier Uniform initialization.
- Select a final model using Leave-One-Scene-Out Cross-Validation.
- Evaluate generalization on a completely unseen scene.

## Dataset

The experiments use handcrafted RGB color histogram features generated during Week 7.

| Dataset | Scenes | Samples |
|---|---|---:|
| Development | 48–58 | 2,325 |
| Final Test | 59 | 375 |
| Total | 48–59 | 2,700 |

**Feature dimensions:** 4,096 RGB histogram features per sample.

**Number of classes:** 9 YCB object categories.

Scene 59 was kept completely separate during model development and used only for final evaluation.

## Model Architecture

The classifier is a Multi-Layer Perceptron (MLP).

| Layer | Configuration |
|---|---|
| Input | 4,096 features |
| Hidden Layer 1 | 256 neurons + ReLU |
| Dropout 1 | 0.3 (baseline) |
| Hidden Layer 2 | 128 neurons + ReLU |
| Dropout 2 | 0.2 (baseline) |
| Output | 9 classes |

**Trainable parameters:** 1,082,889

### Training Configuration

| Parameter | Value |
|---|---:|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Baseline Weight Decay | 0.0001 |
| Loss Function | CrossEntropyLoss |
| Cross-Validation Epochs | 15 |
| Final Training Epochs | 4 |
| Random Seed | 42 |
| Device | CPU |

## Experiments

Five controlled configurations were evaluated using the same 11-fold Leave-One-Scene-Out Cross-Validation protocol.

1. Controlled Baseline
2. Increased Dropout (0.5 and 0.4)
3. Increased Weight Decay (0.001)
4. Kaiming Normal Initialization
5. Xavier Uniform Initialization

### Cross-Validation Results

| Experiment | OOF Accuracy | OOF Macro-F1 |
|---|---:|---:|
| Increased Weight Decay | **84.00%** | **78.21%** |
| Controlled Baseline | 83.57% | 77.49% |
| Kaiming Normal | 83.35% | 76.77% |
| Increased Dropout | 84.17% | 76.45% |
| Xavier Uniform | 81.59% | 75.97% |

Increased weight decay achieved the highest OOF Macro-F1 and was selected for final training.

The improvement over the controlled baseline was modest:

- Accuracy: +0.43 percentage points.
- Macro-F1: +0.72 percentage points.

## Overfitting Analysis

The training curves showed that training loss decreased rapidly and training accuracy approached 100%, while validation performance remained lower and varied considerably across scenes.

At epoch 15, the controlled baseline showed:

- Training Accuracy: 99.88%
- Validation Accuracy: 76.45%
- Accuracy Gap: 23.42 percentage points
- Loss Gap: 2.2477

Increased dropout reduced the measured training-validation gap, while increased weight decay showed more favorable validation behavior around epoch 10.

However, neither technique completely eliminated overfitting.

## Final Model

The increased weight decay configuration was selected based on its cross-validation Macro-F1.

The median best epoch across the 11 validation folds was 4.

The selected model was retrained on all 2,325 development samples for four epochs.

### Final Test Results — Scene 59

| Metric | Result |
|---|---:|
| Test Samples | 375 |
| Accuracy | **64.80%** |
| Macro-F1 (all 9 classes) | 37.13% |
| Macro-F1 (5 present classes) | **66.83%** |

### Per-Class F1-Score

| Class | F1-Score |
|---|---:|
| Cracker Box | 99.34% |
| Tomato Soup Can | 81.58% |
| Power Drill | 100.00% |
| Tuna Fish Can | 29.55% |
| Potted Meat Can | 23.68% |

The main classification difficulties involved Potted Meat Can and Tuna Fish Can.

The most frequent errors were:

- Potted Meat Can → Sugar Box: 57 samples.
- Tuna Fish Can → Potted Meat Can: 47 samples.
- Tuna Fish Can → Tomato Soup Can: 15 samples.

The difference between cross-validation and final test performance highlights the challenge of generalizing across unseen scenes.

## Project Structure

```text
MLInternship_week10/
│
├── notebooks/
│   └── improving_deep_neural_networks.ipynb
│
├── models/
│   ├── best_model.pth
│   └── feature_scaler.joblib
│
├── results/
│   ├── experiment_comparison.csv
│   ├── per_scene_weight_decay_comparison.csv
│   ├── overfitting_gap_analysis.csv
│   ├── final_overfitting_comparison.csv
│   ├── weight_decay_best_epochs.csv
│   ├── final_training_history.csv
│   ├── scene59_predictions.csv
│   ├── scene59_classification_report.csv
│   ├── scene59_misclassifications.csv
│   ├── all_experiments_loss_curves.png
│   ├── all_experiments_accuracy_curves.png
│   └── scene59_confusion_matrix.png
│
├── Report.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/farahkhoda/MLInternship_week10.git
cd MLInternship_week10
```

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Open the notebook:

```bash
jupyter notebook notebooks/improving_deep_neural_networks.ipynb
```

Run the notebook cells sequentially.

**Dataset requirement:** The notebook loads the handcrafted feature archive and metadata generated during Week 7. Update the dataset paths in the notebook if your local directory structure differs.

## Saved Artifacts

The final model and preprocessing pipeline are saved separately:

- `models/best_model.pth` — trained model weights and configuration.
- `models/feature_scaler.joblib` — fitted StandardScaler.
- `results/` — experiment metrics, predictions, training histories, and visualizations.

## Limitations

- The model uses handcrafted RGB histogram features rather than raw images.
- Only one random seed was used for the main controlled comparisons.
- Validation performance varied substantially across scenes.
- The final test set contains five of the nine target classes.
- Regularization improved cross-validation performance only modestly.
- Generalization to additional unseen scenes remains uncertain.

## Conclusion

Week 10 demonstrated the importance of diagnosing overfitting through scene-based validation.

Increased weight decay achieved the best cross-validation Macro-F1, but the final evaluation on Scene 59 showed that generalization remained challenging.

The experiments provided practical experience with regularization, initialization, controlled comparisons, model selection, and independent evaluation.

---

**Project:** ML Internship — Week 10  
**Focus:** Deep Learning, Regularization, Overfitting, and Generalization  
**Framework:** PyTorch  
**Dataset:** YCB-Video  
**Author:** Farah Khoda
