import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# Download and prepare the data
data_path = "../data/lifesat.csv"
lifesat = pd.read_csv(data_path)

X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# Visualize the data
lifesat.plot(
    kind="scatter",
    grid=True,
    x="GDP per capita (USD)",
    y="Life satisfaction",
)
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# Select a linear model
model_linear = LinearRegression()
model_kn = KNeighborsRegressor(n_neighbors=3)

# Train the model
model_linear.fit(X, y)
model_kn.fit(X, y)

# Make a prediction for Cyprus
X_new = [[37_655.2]]

print(model_linear.predict(X_new))
print(model_kn.predict(X_new))
