"""Bagging vs boosting on Telco churn: Random Forest vs AdaBoost, Gradient Boosting, XGBoost.

For every model:
  1. Default hyperparameters  -> 5-fold stratified CV ROC-AUC on the training set
  2. RandomizedSearchCV (30 settings x 5 folds, ROC-AUC) -> best hyperparameters
  3. Decision threshold chosen on out-of-fold training predictions (max F1) - never on the test set
  4. One final evaluation on the untouched 20% test set
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import prepare

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 42
N_ITER = 30
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def make_pipeline(model, numeric, categorical):
    # Tree models need no scaling; categories are one-hot encoded.
    pre = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def model_specs(pos_weight):
    return {
        "Random Forest (bagging)": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {"clf__n_estimators": randint(200, 800),
             "clf__max_depth": [None, 4, 6, 8, 10, 12, 16],
             "clf__min_samples_leaf": randint(1, 30),
             "clf__max_features": ["sqrt", "log2", 0.3, 0.5],
             "clf__class_weight": [None, "balanced"]}),
        "AdaBoost (boosting)": (
            AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), random_state=SEED),
            {"clf__n_estimators": randint(50, 600),
             "clf__learning_rate": loguniform(0.01, 1.0),
             "clf__estimator__max_depth": [1, 2, 3]}),
        "Gradient Boosting (boosting)": (
            GradientBoostingClassifier(random_state=SEED),
            {"clf__n_estimators": randint(100, 600),
             "clf__learning_rate": loguniform(0.01, 0.2),
             "clf__max_depth": randint(2, 6),
             "clf__subsample": uniform(0.6, 0.4),
             "clf__min_samples_leaf": randint(1, 50)}),
        "XGBoost (boosting)": (
            XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss"),
            {"clf__n_estimators": randint(100, 800),
             "clf__learning_rate": loguniform(0.01, 0.2),
             "clf__max_depth": randint(2, 7),
             "clf__subsample": uniform(0.6, 0.4),
             "clf__colsample_bytree": uniform(0.5, 0.5),
             "clf__min_child_weight": randint(1, 10),
             "clf__gamma": uniform(0, 5),
             "clf__reg_lambda": loguniform(0.1, 10),
             "clf__scale_pos_weight": [1, pos_weight]}),
    }


def best_f1_threshold(y, proba):
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def main():
    RESULTS.mkdir(exist_ok=True)
    df = prepare.load_clean()
    X, y, numeric, categorical = prepare.split_columns(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pos_weight = round(float((y_tr == 0).sum() / (y_tr == 1).sum()), 2)
    print(f"train {len(X_tr)} | test {len(X_te)} | churn rate {y.mean():.1%} | pos_weight {pos_weight}")

    rows, best_params, fitted = [], {}, {}
    for name, (model, space) in model_specs(pos_weight).items():
        pipe = make_pipeline(model, numeric, categorical)

        default_cv = cross_val_score(pipe, X_tr, y_tr, cv=CV, scoring="roc_auc", n_jobs=-1)

        t0 = time.time()
        search = RandomizedSearchCV(pipe, space, n_iter=N_ITER, cv=CV, scoring="roc_auc",
                                    n_jobs=-1, random_state=SEED, refit=True)
        search.fit(X_tr, y_tr)
        tune_seconds = time.time() - t0
        best = search.best_estimator_
        fitted[name] = best
        best_params[name] = {k.replace("clf__", ""): (v if not isinstance(v, (np.floating, np.integer)) else v.item())
                             for k, v in search.best_params_.items()}

        # Threshold from out-of-fold predictions on TRAIN only
        oof = cross_val_predict(best, X_tr, y_tr, cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
        thr = best_f1_threshold(y_tr, oof)

        p_te = best.predict_proba(X_te)[:, 1]
        p_tr = best.predict_proba(X_tr)[:, 1]
        pred = (p_te >= thr).astype(int)
        rows.append({
            "model": name,
            "cv_auc_default": default_cv.mean(),
            "cv_auc_tuned": search.best_score_,
            "cv_auc_tuned_std": search.cv_results_["std_test_score"][search.best_index_],
            "train_auc": roc_auc_score(y_tr, p_tr),
            "test_auc": roc_auc_score(y_te, p_te),
            "test_pr_auc": average_precision_score(y_te, p_te),
            "threshold": thr,
            "test_precision": precision_score(y_te, pred),
            "test_recall": recall_score(y_te, pred),
            "test_f1": f1_score(y_te, pred),
            "test_accuracy": accuracy_score(y_te, pred),
            "tuning_seconds": tune_seconds,
        })
        print(f"{name:30s} CV AUC default {default_cv.mean():.4f} -> tuned {search.best_score_:.4f} | "
              f"test AUC {rows[-1]['test_auc']:.4f} | F1 {rows[-1]['test_f1']:.3f} | {tune_seconds:.0f}s")

    res = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    res.round(4).to_csv(RESULTS / "model_comparison.csv", index=False)
    (RESULTS / "best_params.json").write_text(json.dumps(best_params, indent=2))
    print("\n" + res.round(3).to_string(index=False))

    # Permutation importance of the best model on the test set (model-agnostic, uses original columns)
    top_name = res.iloc[0]["model"]
    imp = permutation_importance(fitted[top_name], X_te, y_te, scoring="roc_auc",
                                 n_repeats=10, random_state=SEED, n_jobs=-1)
    imp_df = (pd.DataFrame({"feature": X_te.columns, "auc_drop": imp.importances_mean})
              .sort_values("auc_drop", ascending=False))
    imp_df.round(4).to_csv(RESULTS / "permutation_importance.csv", index=False)
    print(f"\nTop features for {top_name} (drop in test ROC-AUC when shuffled):")
    print(imp_df.head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
