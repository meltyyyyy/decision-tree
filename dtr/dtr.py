import os

import mglearn.datasets
import pandas as pd
from matplotlib import pyplot as plt


def execute():
    ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))
    fig = plt.figure()
    plt.semilogy(ram_prices.date, ram_prices.price)
    plt.xlabel("Year")
    plt.ylabel("Price in $/Mbyte")
    fig.savefig("dtr/dtr.png")
