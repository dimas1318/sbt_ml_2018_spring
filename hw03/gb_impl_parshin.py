#coding=utf-8

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
import numpy as np


# Параметрами с которыми вы хотите обучать деревья
TREE_PARAMS_DICT = {'max_depth': 9, 'max_features': 5, 'random_state': 13}
# Параметр tau (learning_rate) для вашего GB
TAU = 0.08051


class SimpleGB(BaseEstimator):
    def __init__(self, tree_params_dict, iters, tau):
        self.tree_params_dict = tree_params_dict
        self.iters = iters
        self.tau = tau
        
    def fit(self, X_data, y_data):
        self.base_algo = LogisticRegression(C=0.00001).fit(X_data, y_data)
        self.estimators = []
        # Sigmoid func
        # yp = 1 / (1 + exp(-y_pred) (transform prediction into probability)
        # y_pred = -ln (1 / (yp - 1))
        curr_pred = - np.log(1. / self.base_algo.predict_proba(X_data)[:, 1] - 1)
        
        for iter_num in range(self.iters):
            # Нужно посчитать градиент функции потерь
            yp = 1. / (1 + np.exp(-curr_pred))
            # yp' = -exp(y_pred) / (1 + exp(-y_pred))**2 = -yp(1 - yp)
            # log_loss = y*log(yp) + (1-y)*log(1 - yp)    y = 0 or 1
            # d/d(yp) (log_loss) = yp'(y / yp - (1 - y) / (1 - yp))
            grad = -yp * (1. - yp) * (y_data / yp - (1. - y_data) / (1. - yp))
            # Нужно обучить DecisionTreeRegressor предсказывать антиградиент
            # Не забудьте про self.tree_params_dict
            algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, - grad)

            self.estimators.append(algo)
            # Обновите предсказания в каждой точке
            curr_pred += self.tau * algo.predict(X_data)
        return self
    
    def predict(self, X_data):
        # Предсказание на данных
        res = -np.log(1. / self.base_algo.predict_proba(X_data)[:, 1] - 1)
        for estimator in self.estimators:
            res += self.tau * estimator.predict(X_data)
        # Задача классификации, поэтому надо отдавать 0 и 1
        return res > 0.12
