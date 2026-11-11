# Telco Customer Churn: Bagging vs Boosting with Hyperparameter Tuning

Which customers will cancel their telecom subscription? This project compares one **bagging** model
(Random Forest) with three **boosting** models (AdaBoost, Gradient Boosting, XGBoost) on the IBM Telco
Customer Churn dataset (7,043 customers, 26.5% churn), first with default settings and then after
hyperparameter tuning.

## What the data says (EDA)

| Segment | Churn rate |
|---|---|
| Month-to-month contract | **42.7%** (one-year 11.3%, two-year 2.8%) |
| First 12 months of tenure | **47.4%** (after 4 years: 9.5%) |
| Fibre-optic internet | 41.9% (DSL 19.0%) |
| Electronic-check payment | 45.3% (automatic card/bank payments ~15–17%) |

Median tenure is 10 months for churners vs 38 for customers who stayed.

## Method

1. **Cleaning:** dropped `customerID`. 11 blank `TotalCharges` all belong to tenure-0 customers, so they were set to 0.
2. **Features added:** number of add-on services, average monthly spend (`TotalCharges / tenure`),
   `charge_increase` (current minus average monthly charge), `is_new_customer` (tenure ≤ 12).
3. **Split:** stratified 80/20 train/test (5,634 / 1,409). The test set is used only once, at the end.
4. **Preprocessing:** one-hot encoding of 16 categorical columns inside a scikit-learn `Pipeline`
   (no scaling needed for tree models).
5. **Tuning:** `RandomizedSearchCV`, 30 settings × 5-fold stratified CV per model, optimising ROC-AUC.
6. **Threshold:** chosen to maximise F1 on **out-of-fold training predictions**, never on the test set.
7. **Explainability:** permutation importance on the test set for the best model.

## Results

| Model | CV ROC-AUC, default | CV ROC-AUC, tuned (± sd) | Train AUC | **Test ROC-AUC** | Test PR-AUC | Test recall | Test F1 | Tuning time* |
|---|---|---|---|---|---|---|---|---|
| Gradient Boosting | 0.845 | 0.850 ± 0.012 | 0.862 | **0.846** | 0.664 | 71.9% | 0.637 | 270 s |
| XGBoost | 0.822 | **0.850** ± 0.012 | 0.862 | **0.846** | **0.664** | 73.0% | 0.638 | **24 s** |
| Random Forest (bagging) | 0.823 | 0.848 ± 0.009 | 0.889 | 0.844 | 0.653 | 73.5% | 0.643 | 196 s |
| AdaBoost | 0.846 | 0.849 ± 0.011 | 0.856 | 0.841 | 0.651 | 70.3% | 0.632 | 249 s |

\*on a 2-core machine; faster on a typical laptop.

**Findings**
- **Tuning mattered most for XGBoost (+0.029 CV AUC) and Random Forest (+0.026).** Their defaults
  (deep trees) overfit. AdaBoost and Gradient Boosting were already well configured by default (+0.003–0.004).
- **After tuning, all four models land within 0.005 test ROC-AUC (0.841–0.846)**, smaller than the CV
  standard deviation (~0.01), so no model is meaningfully better on this dataset.
- **Tuning picked shallow trees (depth 2) with a small learning rate** for both Gradient Boosting and XGBoost:
  the churn signal is simple and mostly additive, so complex interactions mainly add noise.
- **Bagging vs boosting:** Random Forest fits training data more tightly (train AUC 0.889 vs 0.86 for the
  boosters) but generalises equally well, because averaging many deep trees cancels their variance.
- **XGBoost tuned ~10× faster** than scikit-learn's Gradient Boosting for the same accuracy, so it's the
  practical choice to deploy and retrain.
- **Top drivers (permutation importance):** tenure and contract type dominate (shuffling them drops test
  AUC by 0.056 and 0.047), followed by internet service, online security and tech support.

## Business takeaways

- At the chosen threshold, the model catches **~73% of churners** in the test set, with about 57% of
  flagged customers actually churning (2.2× the 26.5% base rate).
- Retention spend should target **new, month-to-month, fibre, electronic-check customers**. Offers that
  move customers to one- or two-year contracts, or bundle security and tech-support add-ons, address the
  strongest churn drivers.

## Run it

```bash
pip install -r requirements.txt
python src/download_data.py
python src/compare_models.py     # writes results/model_comparison.csv, best_params.json, permutation_importance.csv
```

Data: IBM Telco Customer Churn sample dataset
([IBM GitHub](https://github.com/IBM/telco-customer-churn-on-icp4d)).
