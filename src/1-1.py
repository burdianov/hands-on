import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data_path = "../data/lifesat.csv"
lifesat = pd.read_csv(data_path)
