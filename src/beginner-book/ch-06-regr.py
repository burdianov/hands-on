import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# lINEAR REGRESSION
# Step 1 - Load the data
housing = pd.read_csv("../../data/housing.csv")

# Step 2 - Examine the data
housing.head()
housing.info()
housing.isnull().sum()

# Step 3 - Split the dataset
X = housing[["Floor Area (sqft)"]]
y = housing[["Value ($1000)"]]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4 - Visualizing the data

plt.scatter(X_train, y_train, s=10)
plt.title("Housing Price vs Floor Area")
plt.xlabel("Floor Area (sqft)")
plt.ylabel("Price in $1000s")

# Step 5 - Preprocessing
"""
There is no need to perform data preprocessing for this example as our dataset does not contain any missing value or textual data. In addition, we do not need to do feature scaling as there is only one feature in the dataset.
"""

# Step 6 - Train the Model
lr = LinearRegression()
lr.fit(X_train.values, y_train.values)
print(lr.intercept_)
print(lr.coef_)

predicted_price = lr.predict([[250], [300]])
print(predicted_price)

plt.scatter(X_train, y_train, s=10)
y_pred = lr.predict(X_train.values)
plt.plot(X_train, y_pred)

plt.title("Housing Price vs Floor Area")
plt.xlabel("Floor Area (sqft)")
plt.ylabel("Price in $1000s")

# Step 7 - Evaluate the Model
RMSE = mean_squared_error(y_train, y_pred, squared=False)
r2 = r2_score(y_train, y_pred)
print(RMSE)
print(r2)

y_pred_test = lr.predict(X_test.values)
RMSE = mean_squared_error(y_test, y_pred_test, squared=False)
r2 = r2_score(y_test, y_pred_test)
print(RMSE)
print(r2)
