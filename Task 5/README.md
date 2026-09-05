# Task 5: Decision Trees and Random Forests

**Dataset:** Heart Disease Dataset (`heart.csv`, 1025 rows, 13 features, binary `target`)

## What I did
1. Loaded the dataset and split into train/test (80/20, stratified).
2. Trained a `DecisionTreeClassifier` and visualized it with `plot_tree` (`decision_tree.png`, top 3 levels shown for readability).
3. Analyzed overfitting by sweeping `max_depth` from 1–20 and plotting train vs test accuracy (`overfitting_depth.png`). Picked the depth with best test accuracy.
4. Trained a `RandomForestClassifier` (200 trees) and compared accuracy against the single tree.
5. Extracted and plotted feature importances from the Random Forest (`feature_importance.png`).
6. Evaluated both models with 5-fold cross-validation for a more robust accuracy estimate.

## Results
| Model | Test Accuracy | 5-Fold CV Mean |
|---|---|---|
| Decision Tree (full depth) | 0.9854 | – |
| Decision Tree (max_depth=9, best) | 0.9854 | 0.9980 |
| Random Forest (200 trees) | 1.0000 | 0.9971 |

Top features by importance: `cp` (chest pain type), `thalach` (max heart rate), `ca` (number of major vessels), `oldpeak`, `thal`.

Full numeric output: see `results_summary.txt`.

## Files
- `task5.py` — full script
- `heart.csv` — dataset
- `decision_tree.png` — tree visualization
- `overfitting_depth.png` — train vs test accuracy across depths
- `feature_importance.png` — Random Forest feature importances
- `results_summary.txt` — all metrics in text form

## How to run
```bash
pip install scikit-learn pandas matplotlib
python task5.py
```

## Interview Q&A (brief)

**1. How does a decision tree work?**
It recursively splits the data on the feature/threshold that best separates the classes, forming a tree of if/else rules until a stopping criterion (pure leaf, max depth, min samples) is met.

**2. What is entropy and information gain?**
Entropy measures impurity/disorder in a node (0 = pure, max when classes are evenly mixed). Information gain is the reduction in entropy after a split — the tree picks the split that maximizes it.

**3. How is random forest better than a single tree?**
It trains many trees on bootstrapped samples with random feature subsets, then averages/votes their predictions. This reduces variance and overfitting compared to one deep tree.

**4. What is overfitting and how do you prevent it?**
Overfitting is when a model learns noise/specifics of training data and fails to generalize. Prevent it via pruning (max_depth, min_samples_leaf), cross-validation, ensembling, or more data.

**5. What is bagging?**
Bootstrap Aggregating: train multiple models on random samples (with replacement) of the data and combine their outputs (majority vote/average) to reduce variance. Random Forest is bagging + random feature selection.

**6. How do you visualize a decision tree?**
`sklearn.tree.plot_tree` (matplotlib) or export to Graphviz with `export_graphviz` and render as an image/PDF.

**7. How do you interpret feature importance?**
For tree models, importance reflects how much a feature reduces impurity (weighted by number of samples) across all splits where it's used — higher value means more influence on predictions.

**8. Pros/cons of random forests?**
Pros: high accuracy, robust to overfitting, handles non-linear relationships, gives feature importance, needs little preprocessing.
Cons: less interpretable than a single tree, slower to train/predict, larger memory footprint.
