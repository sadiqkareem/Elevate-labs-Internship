
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))
print(f"Decision Tree (full depth) test accuracy: {dt_acc:.4f}")

plt.figure(figsize=(22, 12))
plot_tree(dt, feature_names=X.columns, class_names=["No Disease", "Disease"],
          filled=True, max_depth=3, fontsize=8)
plt.title("Decision Tree (top 3 levels shown, fully grown model)")
plt.savefig("decision_tree.png", dpi=150, bbox_inches="tight")
plt.close()

depths = range(1, 21)
train_scores, test_scores = [], []
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
    test_scores.append(accuracy_score(y_test, clf.predict(X_test)))

plt.figure(figsize=(8, 5))
plt.plot(depths, train_scores, marker="o", label="Train accuracy")
plt.plot(depths, test_scores, marker="o", label="Test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Overfitting vs max_depth")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("overfitting_depth.png", dpi=150, bbox_inches="tight")
plt.close()

best_depth = depths[int(np.argmax(test_scores))]
print(f"Best max_depth by test accuracy: {best_depth} "
      f"(test acc={max(test_scores):.4f})")

dt_pruned = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_pruned.fit(X_train, y_train)
pruned_acc = accuracy_score(y_test, dt_pruned.predict(X_test))
print(f"Pruned Decision Tree (max_depth={best_depth}) test accuracy: {pruned_acc:.4f}")

# ---------- 4. Random Forest ----------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest test accuracy: {rf_acc:.4f}")

print("\nComparison (single tree vs random forest):")
print(f"  Decision Tree (full):    {dt_acc:.4f}")
print(f"  Decision Tree (pruned):  {pruned_acc:.4f}")
print(f"  Random Forest:           {rf_acc:.4f}")

print("\nClassification report - Random Forest:")
print(classification_report(y_test, rf.predict(X_test)))

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importances (Random Forest):")
print(importances)

plt.figure(figsize=(8, 6))
importances.sort_values().plot(kind="barh", color="steelblue")
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

dt_cv = cross_val_score(DecisionTreeClassifier(max_depth=best_depth, random_state=42), X, y, cv=5)
rf_cv = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=42), X, y, cv=5)

print(f"\n5-Fold CV - Decision Tree (max_depth={best_depth}): "
      f"mean={dt_cv.mean():.4f}, std={dt_cv.std():.4f}")
print(f"5-Fold CV - Random Forest: mean={rf_cv.mean():.4f}, std={rf_cv.std():.4f}")

with open("results_summary.txt", "w") as f:
    f.write("TASK 5 RESULTS SUMMARY\n")
    f.write("======================\n\n")
    f.write(f"Decision Tree (full depth) test accuracy: {dt_acc:.4f}\n")
    f.write(f"Best max_depth (least overfitting): {best_depth}\n")
    f.write(f"Decision Tree (max_depth={best_depth}) test accuracy: {pruned_acc:.4f}\n")
    f.write(f"Random Forest test accuracy: {rf_acc:.4f}\n\n")
    f.write("Feature importances (Random Forest):\n")
    f.write(importances.to_string() + "\n\n")
    f.write(f"5-Fold CV Decision Tree: mean={dt_cv.mean():.4f}, std={dt_cv.std():.4f}\n")
    f.write(f"5-Fold CV Random Forest: mean={rf_cv.mean():.4f}, std={rf_cv.std():.4f}\n")

print("\nDone. Files saved: decision_tree.png, overfitting_depth.png, "
      "feature_importance.png, results_summary.txt")
