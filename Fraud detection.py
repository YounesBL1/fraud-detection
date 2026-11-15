import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import (ParameterGrid, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.preprocessing import RobustScaler

SEED = 42

# Step 1: Load data
df = pd.read_csv("C:/Users/youne/OneDrive/Desktop/project/Fraud detection.py")
print(df.shape, "| missing:", df.isnull().sum().sum(), "| duplicates:", df.duplicated().sum())

# Step 2: EDA
print(df["Class"].value_counts())
df["Class"].value_counts().plot(kind="bar", logy=True, title="Class distribution (log scale)")
plt.show()
print(df.corr()["Class"].drop("Class").abs().sort_values(ascending=False).head(8))

# Step 3: Clean
df = df.drop_duplicates().reset_index(drop=True)

# Step 4: Split first, then scale (fit on train only)
X, y = df.drop("Class", axis=1), df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

scaler = RobustScaler()
cols = ["Time", "Amount"]
X_train, X_test = X_train.copy(), X_test.copy()
X_train[cols] = scaler.fit_transform(X_train[cols])
X_test[cols] = scaler.transform(X_test[cols])

# Validation split (from train) for comparing models; test set stays untouched
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.25, stratify=y_train, random_state=SEED)

# Steps 5-7: Handle imbalance with class weights, train baseline + stronger models
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                            class_weight="balanced_subsample",
                                            n_jobs=-1, random_state=SEED),
    "Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced",
                                                        random_state=SEED),
}

# Step 8: Compare models (accuracy is misleading here, so use PR-AUC)
val_scores = {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_val)[:, 1]
    val_scores[name] = average_precision_score(y_val, p)
    print(f"{name:20s} PR-AUC={val_scores[name]:.4f}  ROC-AUC={roc_auc_score(y_val, p):.4f}")
best_name = max(val_scores, key=val_scores.get)
print("Best on validation:", best_name)

# Step 10: Tune the best model from Step 8
grids = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
    "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 10, 20],
                      "min_samples_leaf": [1, 2, 5], "max_features": ["sqrt", 0.3]},
    "Gradient Boosting": {"max_iter": [100, 200, 300], "max_depth": [3, 5, 7, None],
                          "learning_rate": [0.03, 0.05, 0.1, 0.2],
                          "min_samples_leaf": [10, 20, 40], "l2_regularization": [0, 0.5, 1, 2]},
}
search = RandomizedSearchCV(
    models[best_name], grids[best_name],
    n_iter=min(8, len(ParameterGrid(grids[best_name]))), scoring="average_precision",
    cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
    random_state=SEED, verbose=1)
search.fit(X_train, y_train)
model = search.best_estimator_
print("Best params:", search.best_params_, "| CV PR-AUC: %.4f" % search.best_score_)

# Step 9: Pick the decision threshold from out-of-fold predictions on train
oof = cross_val_predict(model, X_train, y_train, method="predict_proba",
                        cv=StratifiedKFold(3, shuffle=True, random_state=SEED))[:, 1]
prec, rec, thr = precision_recall_curve(y_train, oof)
f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
threshold = thr[np.argmax(f1)]
print("Chosen threshold: %.3f" % threshold)

# Step 8 (final): Evaluate once on the test set
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= threshold).astype(int)
print("Test PR-AUC: %.4f | ROC-AUC: %.4f" % (average_precision_score(y_test, proba),
                                              roc_auc_score(y_test, proba)))
print(classification_report(y_test, pred, target_names=["Legit", "Fraud"], digits=4))
print(confusion_matrix(y_test, pred))

# Step 11: Feature importance
idx = np.concatenate([y_test[y_test == 1].index,
                      y_test[y_test == 0].sample(min(3000, (y_test == 0).sum()), random_state=SEED).index])
perm = permutation_importance(model, X_test.loc[idx], y_test.loc[idx],
                              scoring="average_precision", n_repeats=3, random_state=SEED)
imp = pd.Series(perm.importances_mean, index=X.columns).sort_values()
imp.tail(15).plot(kind="barh", title="Permutation importance")
plt.tight_layout()
plt.show()

# Step 12: Save model
joblib.dump({"model": model, "scaler": scaler, "cols": cols, "threshold": threshold},
            "fraud_model.joblib")
print("Saved fraud_model.joblib")