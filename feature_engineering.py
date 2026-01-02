# feature_engineering.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        bill_cols=None,
        pay_cols=None,
        limit_col="LIMIT_BAL",
        age_col="AGE"
    ):
        self.bill_cols = bill_cols or ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3"]
        self.pay_cols = pay_cols or ["PAY_0", "PAY_2", "PAY_3"]
        self.limit_col = limit_col
        self.age_col = age_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["avg_bill_amt_3m"] = X[self.bill_cols].mean(axis=1)
        X["credit_utilization"] = X["avg_bill_amt_3m"] / (X[self.limit_col] + 1)

        X["num_missed_payments"] = (X[self.pay_cols] > 0).sum(axis=1)
        X["max_delay_last_6_months"] = X[self.pay_cols].max(axis=1)

        X["debt_to_limit"] = X["avg_bill_amt_3m"] / (X[self.limit_col] + 1)

        X["age_bin"] = pd.cut(
            X[self.age_col],
            bins=[18, 25, 35, 50, 100],
            labels=["18-25", "26-35", "36-50", "50+"]
        )

        return X
