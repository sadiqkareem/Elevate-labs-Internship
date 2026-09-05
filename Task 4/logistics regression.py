import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay, classification_report
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load a binary classification dataset
# ---------------------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="diagnosis")  # 0 = malignant, 1 = benign

print("Dataset shape:", X.shape)
print("Class distribution:\n", y.value_counts().rename({0: "malignant", 1: "benign"}))
print()

# ---------------------------------------------------------------------------
# 2. Train/test split and standardize features
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on training data
X_test_scaled = scaler.transform(X_test)          # apply same transform to test

# ---------------------------------------------------------------------------
# 3. Fit a Logistic Regression model
# ---------------------------------------------------------------------------
model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]  # P(class = benign)

# ---------------------------------------------------------------------------
# 4. Evaluate: confusion matrix, precision, recall, ROC-AUC
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("=" * 60)
print("EVALUATION AT DEFAULT THRESHOLD (0.5)")
print("=" * 60)
print("Confusion Matrix:\n", cm)
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print("\nFull classification report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix (threshold = 0.5)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# ROC curve plot
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(5.5, 4.5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", color="darkorange", lw=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5a. Sigmoid function explanation + plot
# ---------------------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z_vals = np.linspace(-10, 10, 200)
sig_vals = sigmoid(z_vals)

plt.figure(figsize=(5.5, 4))
plt.plot(z_vals, sig_vals, color="teal", lw=2)
plt.axhline(0.5, color="gray", linestyle="--", lw=1)
plt.axvline(0, color="gray", linestyle="--", lw=1)
plt.xlabel("z (linear combination: w·x + b)")
plt.ylabel("sigmoid(z)")
plt.title("Sigmoid Function")
plt.tight_layout()
plt.savefig("sigmoid_curve.png", dpi=150)
plt.close()

print("\n" + "=" * 60)
print("SIGMOID FUNCTION")
print("=" * 60)
print("""
Logistic Regression first computes a linear score:
    z = w1*x1 + w2*x2 + ... + wn*xn + b

This z can be any real number (-inf to +inf), so it is passed through the
SIGMOID (logistic) function to squash it into a probability between 0 and 1:

    sigmoid(z) = 1 / (1 + e^(-z))

  - As z -> +inf,  sigmoid(z) -> 1
  - As z -> -inf,  sigmoid(z) -> 0
  - At z = 0,      sigmoid(z) = 0.5   (the default decision boundary)

The model predicts class 1 (benign) if sigmoid(z) >= threshold (default 0.5),
otherwise it predicts class 0 (malignant).
""")

# ---------------------------------------------------------------------------
# 5b. Threshold tuning
# ---------------------------------------------------------------------------
print("=" * 60)
print("THRESHOLD TUNING")
print("=" * 60)

thresholds_to_try = [0.3, 0.4, 0.5, 0.6, 0.7]
results = []
for t in thresholds_to_try:
    y_pred_t = (y_proba >= t).astype(int)
    p = precision_score(y_test, y_pred_t)
    r = recall_score(y_test, y_pred_t)
    f = f1_score(y_test, y_pred_t)
    results.append((t, p, r, f))
    print(f"Threshold={t:.1f} | Precision={p:.3f} | Recall={r:.3f} | F1={f:.3f}")

results_df = pd.DataFrame(results, columns=["threshold", "precision", "recall", "f1"])
results_df.to_csv("threshold_tuning_results.csv", index=False)

# Plot precision/recall vs threshold across a finer range
fine_thresholds = np.linspace(0.05, 0.95, 50)
precisions, recalls = [], []
for t in fine_thresholds:
    yp = (y_proba >= t).astype(int)
    precisions.append(precision_score(y_test, yp, zero_division=0))
    recalls.append(recall_score(y_test, yp, zero_division=0))

plt.figure(figsize=(6, 4.5))
plt.plot(fine_thresholds, precisions, label="Precision", color="crimson")
plt.plot(fine_thresholds, recalls, label="Recall", color="royalblue")
plt.axvline(0.5, color="gray", linestyle="--", lw=1, label="Default threshold (0.5)")
plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Precision & Recall vs. Threshold")
plt.legend()
plt.tight_layout()
plt.savefig("precision_recall_vs_threshold.png", dpi=150)
plt.close()

print("""
Why tune the threshold?
- Lowering the threshold (e.g. 0.3) makes the model predict "benign" more
  readily -> increases RECALL for benign, but risks more false negatives
  for malignant cases (dangerous in a medical context).
- Raising the threshold (e.g. 0.7) makes the model more conservative about
  predicting "benign" -> increases PRECISION, catches more malignant cases,
  but may flag more benign cases as malignant (more false alarms).
- In cancer diagnosis, missing a malignant case (false negative) is usually
  far costlier than a false alarm, so thresholds are often tuned LOWER than
  0.5 (to trigger "malignant" more easily) or metrics like recall on the
  malignant class are prioritized over overall accuracy.
""")

print("All plots saved: confusion_matrix.png, roc_curve.png, sigmoid_curve.png, precision_recall_vs_threshold.png")
print("Threshold results saved: threshold_tuning_results.csv")
