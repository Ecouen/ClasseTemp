import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class AutoML():
    def __init__(self, csv, target):
        self.df = pd.read_csv(csv)
        self.target = target
        self.fitted = False
        self.models = {}
        self.models['Linear Regression'] = LinearRegression()

    def preprocessing(self):
        dimension = self.df.shape
        # suppression des colonnes avec nans
        for column in self.df.columns:
            if self.df[column].isnull().sum()/dimension[0] > 0.1:
                self.df.drop([column], inplace=True, axis=1)

        # drop nans
        self.df.dropna(axis=0, inplace=True)

        # variables x et y
        X = self.df.drop([self.target], axis=1)
        y = self.df[self.target]

        # get dummies
        for column in X.columns:
            try:
                pd.to_numeric(X[column])
            except ValueError:
                dummies = pd.get_dummies(X[column], drop_first=True)
                new_columns = dummies.columns
                X[column + '_' + new_columns] = dummies[new_columns]
                X.drop([column], inplace=True, axis=1)

        return train_test_split(X, y, test_size=0.25, random_state=0)

    def fit_mdl(self, X, y):
        for model in self.models:
            self.models[model].fit(X,y)
        self.fitted = True

    def prediction(self, X):
        if self.fitted:
            predict = {}
            for model in self.models:
                predict[model] = self.models[model].predict(X)
            return predict
        else:
            print("fit the models before trying to predict")

data = AutoML('train.csv','SalePrice')
X_train, X_test, y_train, y_test = data.preprocessing()
print(X_train.shape)
data.fit_mdl(X_train, y_train)
y_pred = data.prediction(X_test)
print(r2_score(y_pred['Linear Regression'], y_test))