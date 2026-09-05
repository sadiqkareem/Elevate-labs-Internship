# Task 4: Classification with Logistic Regression

**Internship:** AI & ML Internship – Elevate Labs
**Objective:** Build a binary classifier using logistic regression.
**Tools:** Scikit-learn, Pandas, Matplotlib

## Dataset
Breast Cancer Wisconsin (Diagnostic) Data Set — 569 samples, 30 numeric
features (radius, texture, perimeter, area, smoothness, compactness,
concavity, concave points, symmetry, fractal dimension — each as mean,
standard error, and "worst" value), target: **malignant (0)** vs **benign (1)**.

Loaded via `sklearn.datasets.load_breast_cancer()`, which is the same
UCI Breast Cancer Wisconsin dataset distributed on Kaggle
(https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).

## What was done
1. **Train/test split** — 80/20 split, stratified by class.
2. **Feature standardization** — `StandardScaler` fit on training data only,
   then applied to the test set.
3. **Model** — `LogisticRegression` from scikit-learn.
4. **Evaluation:**
   - Confusion matrix
   - Precision, Recall, F1-score
   - ROC curve and ROC-AUC score
5. **Threshold tuning** — precision/recall recomputed across thresholds
   (0.3–0.7 and a finer 0.05–0.95 sweep) to show the precision/recall
   trade-off as the decision threshold changes.
6. **Sigmoid function** — explained and plotted; shows how logistic
   regression squashes the linear score `z = w·x + b` into a 0–1
   probability.

## Results (test set, threshold = 0.5)
| Metric | Score |
|---|---|
| Precision | 0.986 |
| Recall | 0.986 |
| F1-score | 0.986 |
| ROC-AUC | 0.995 |

Confusion matrix:
```
                 Predicted Malignant  Predicted Benign
Actual Malignant          41                  1
Actual Benign              1                 71
```

## Files
- `task4_logistic_regression.py` — full pipeline script
- `confusion_matrix.png` — confusion matrix heatmap
- `roc_curve.png` — ROC curve with AUC
- `sigmoid_curve.png` — sigmoid function plot
- `precision_recall_vs_threshold.png` — precision & recall vs. decision threshold
- `threshold_tuning_results.csv` — precision/recall/F1 at several thresholds

## How to run
```bash
pip install scikit-learn pandas matplotlib
python task4_logistic_regression.py
```

## Interview Questions — Short Answers

1. **How does logistic regression differ from linear regression?**
   Linear regression predicts a continuous value; logistic regression
   predicts a probability (via the sigmoid function) and is used for
   classification. Its loss function is log-loss, not mean squared error.

2. **What is the sigmoid function?**
   `sigmoid(z) = 1 / (1 + e^-z)`. It maps any real number `z` to a value
   between 0 and 1, so it can represent a probability.

3. **What is precision vs recall?**
   Precision = TP / (TP + FP) — of all predicted positives, how many were
   correct. Recall = TP / (TP + FN) — of all actual positives, how many
   were correctly identified.

4. **What is the ROC-AUC curve?**
   The ROC curve plots True Positive Rate vs False Positive Rate across
   all thresholds. AUC (Area Under Curve) summarizes overall separability;
   1.0 is perfect, 0.5 is random guessing.

5. **What is the confusion matrix?**
   A table showing True Positives, True Negatives, False Positives, and
   False Negatives — the raw counts behind all the classification metrics.

6. **What happens if classes are imbalanced?**
   Accuracy becomes misleading (a model can score high by always
   predicting the majority class). Precision, recall, F1, and ROC-AUC (or
   PR-AUC) become more informative; techniques like class weighting,
   resampling (SMOTE/undersampling), or threshold adjustment help.

7. **How do you choose the threshold?**
   Based on the cost of false positives vs false negatives for the
   specific problem. Tools like the precision-recall curve, ROC curve, or
   business/domain cost analysis guide the choice — e.g., in medical
   diagnosis, thresholds are often lowered to reduce false negatives.

8. **Can logistic regression be used for multi-class problems?**
   Yes — via **One-vs-Rest** (train one binary classifier per class) or
   **multinomial/softmax logistic regression**, which generalizes the
   sigmoid to a softmax function over multiple classes.
