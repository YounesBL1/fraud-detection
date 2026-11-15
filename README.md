# Credit Card Fraud Detection

Machine learning project that detects fraudulent credit card transactions
in a highly imbalanced dataset (492 frauds out of 284,807 transactions, 0.17%).

## Dataset
Kaggle "Credit Card Fraud Detection" (ULB). Download `creditcard.csv` from
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it in the project folder.
The file is not included in this repo because of its size.

## Approach
1. Load data and run exploratory analysis
2. Remove duplicate rows
3. Stratified train/test split, then scale Time and Amount (fit on train only)
4. Handle class imbalance with class weights
5. Compare Logistic Regression, Random Forest, and Gradient Boosting on a validation set
6. Tune the best model with RandomizedSearchCV (scored on PR-AUC)
7. Choose the decision threshold using out-of-fold predictions
8. Evaluate once on the untouched test set
9. Analyze feature importance and save the model

## Results
| Metric | Value |
|---|---|
| PR-AUC | (your result) |
| ROC-AUC | (your result) |
| Fraud precision | (your result) |
| Fraud recall | (your result) |

Accuracy is not reported as the main metric because predicting "no fraud"
every time already gives 99.8%.

## How to run
pip install -r requirements.txt
python fraud_detection.py

## Limitations
The test set contains only 95 fraud cases, so scores can vary noticeably
between runs. Features V1-V28 are anonymized, which limits interpretation.