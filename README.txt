RAIN PREDICTION IN AUSTRALIA — END-TO-END DATA SCIENCE PROJECT
================================================================

An end-to-end machine learning project on the "Rain in Australia"
(weatherAUS) dataset, predicting whether it will rain tomorrow from
daily weather observations.


BUSINESS CONTEXT
----------------
Scenario: Agribusiness Operations Management (planting, harvesting,
pesticide application, irrigation planning).

Missing a rainy day (false negative) is more expensive than issuing a
false alarm (false positive). Therefore, RECALL on the "rain" class
was chosen as the primary evaluation metric.


DATASET
-------
- Source: Kaggle — "Rain in Australia" (jsphyg/weather-dataset-rattle-package)
- Origin: Australian Bureau of Meteorology (BOM)
- Size: 145,460 rows x 23 columns
- Target: RainTomorrow (Yes/No) — imbalanced (~78% no / ~22% yes)


PROJECT STRUCTURE
-----------------
- README.txt              This file
- notebook_english.ipynb  Full analysis notebook
- weatherAUS.csv          Dataset (download from Kaggle)


METHODOLOGY (5 stages)
----------------------
1. EDA driven by business questions
2. Preprocessing: feature engineering (Season), one-hot encoding,
   median imputation, class balancing
3. Supervised modeling: 5 models compared (Logistic Regression,
   Decision Tree, Random Forest, XGBoost, LightGBM)
4. Hyperparameter tuning: Grid Search + Optuna
5. Unsupervised comparison: K-Means (k=2) vs supervised model


WINNING MODEL: XGBoost (optimized with Optuna)
----------------------------------------------
- Recall (rain):    0.7533  (captures ~75% of rainy days)
- Precision (rain): 0.6124  (61% of alerts are confirmed)
- F1 (rain):        0.6756
- ROC-AUC:          0.8999
- Accuracy:         0.8378  (baseline is 78%)


KEY INSIGHT — UNSUPERVISED COMPARISON
-------------------------------------
K-Means did NOT reproduce the rain/no-rain split.
- Silhouette score: 0.035 (weather data is a continuum, not clusters)
- Only 5 of 15 top features overlap between K-Means and XGBoost
- K-Means separated "hot vs cold days" (dominated by temperature)
- XGBoost captured "rain vs no rain" (dominated by humidity, clouds,
  pressure)

Conclusion: the problem is NOT naturally separable by raw distance.
The supervised model adds real value beyond blind grouping.


TECHNOLOGIES
------------
Python 3.10+, pandas, numpy, matplotlib, seaborn, scikit-learn,
xgboost, lightgbm, shap, optuna.


HOW TO RUN
----------
1. Install dependencies:
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   lightgbm shap optuna

2. Download the dataset from Kaggle and place weatherAUS.csv in the
   project root.

3. Open notebook_english.ipynb in Jupyter or Google Colab.


LICENSE
-------
MIT License — feel free to use as reference.
