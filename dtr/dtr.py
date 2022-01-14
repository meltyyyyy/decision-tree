import os

import mglearn.datasets
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def execute():
    ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))
    data_train = ram_prices[ram_prices.date < 2000]
    data_test = ram_prices[ram_prices.date >= 2000]

    X_train = data_train.date[:, np.newaxis]
    y_train = np.log(data_train.price)

    tree = DecisionTreeRegressor().fit(X_train, y_train)
    linear_reg = LinearRegression().fit(X_train, y_train)

    X_all = ram_prices.date[:, np.newaxis]

    pred_tree = tree.predict(X_all)
    pred_lr = linear_reg.predict(X_all)

    price_tree = np.exp(pred_tree)
    price_lr = np.exp(pred_lr)

    fig = plt.figure()
    plt.semilogy(data_train.date, data_train.price, label="Training data")
    plt.semilogy(data_test.date, data_test.price, label="Test data")
    plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
    plt.semilogy(ram_prices.date, price_lr, label="Linear prediction")
    plt.legend()
    fig.savefig("dtr/dtr.png")
