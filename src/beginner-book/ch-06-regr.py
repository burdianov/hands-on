import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# lINEAR REGRESSION
## Step 1 - Load the data
housing = pd.read_csv("../../data/housing.csv")

## Step 2 - Examine the data
housing.head()
housing.info()

### Check for missing values
housing.isnull().sum()

## Step 3 - Split the dataset
X = housing[["Floor Area (sqft)"]]
y = housing[["Value ($1000)"]]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

## Step 4 - Visualizing the data
plt.scatter(X_train, y_train, s=10)
plt.title("Housing Price vs Floor Area")
plt.xlabel("Floor Area (sqft)")
plt.ylabel("Price in $1000s")

## Step 5 - Data Preprocessing
"""
There is no need to perform data preprocessing for this example as our dataset does not contain any missing value or textual data. In addition, we do not need to do feature scaling as there is only one feature in the dataset.
"""

## Step 6 - Train the Model
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

## Step 7 - Evaluate the Model
RMSE = mean_squared_error(y_train, y_pred, squared=False)
r2 = r2_score(y_train, y_pred)
print(RMSE)
print(r2)

y_pred_test = lr.predict(X_test.values)
RMSE = mean_squared_error(y_test, y_pred_test, squared=False)
r2 = r2_score(y_test, y_pred_test)
print(RMSE)
print(r2)

# POLYNOMICAL REGRESSION

## Step 1 - Load the data
housing = pd.read_csv("../../data/housing.csv")

## Step 2 - Examine the data
housing.head()
housing.info()

### Check for missing values
housing.isnull().sum()

## Step 3 - Split the dataset
X = housing[["Floor Area (sqft)"]]
y = housing[["Value ($1000)"]]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

## Step 4 - Visualizing the data
plt.scatter(X_train, y_train, s=10)
plt.title("Housing Price vs Floor Area")
plt.xlabel("Floor Area (sqft)")
plt.ylabel("Price in $1000s")

## Step 5 - Data Preprocessing
poly_features = PolynomialFeatures(degree=2)
X_train_transformed = poly_features.fit_transform(X_train.values)

## Step 6 - Train the Model
pr = LinearRegression()
pr.fit(X_train_transformed, y_train)
print(pr.intercept_)
print(pr.coef_)

num_transformed = poly_features.transform([[250]])
predicted_price = pr.predict(num_transformed)
print(predicted_price)

### Plot the curve
plt.scatter(X_train, y_train, s=10)
y_pred = pr.predict(X_train_transformed)
sorted_zip = sorted(zip(X_train["Floor Area (sqft)"], y_pred))
X_train_sorted, y_pred_sorted = zip(*sorted_zip)
plt.plot(X_train_sorted, y_pred_sorted)

### Label the chart
plt.title("Housing Price vs Floor Area")
plt.xlabel("Floor Area (sqft)")
plt.ylabel("Price in $1000s")

## Step 7 - Evaluate the Model
RMSE = mean_squared_error(y_train, y_pred, squared=False)
r2 = r2_score(y_train, y_pred)
print(RMSE)
print(r2)

X_test_transformed = poly_features.transform(X_test.values)
y_pred_test = pr.predict(X_test_transformed)
RMSE = mean_squared_error(y_test, y_pred_test, squared=False)
r2 = r2_score(y_test, y_pred_test)
print(RMSE)
print(r2)

# PIPELINE

## Create the pipeline
pipeline = Pipeline(
    [("poly", PolynomialFeatures(degree=2)), ("model", LinearRegression())]
)

## Train the model
pipeline.fit(X_train.values, y_train.values)

## Make predictions
print(pipeline.predict([[250]]))

## Evaluate the model on the train set
y_pred = pipeline.predict(X_train.values)
print(mean_squared_error(y_train, y_pred, squared=False))
print(r2_score(y_train, y_pred))

## Evaluate model on the test set
y_pred_test = pipeline.predict(X_test.values)
print(mean_squared_error(y_test, y_pred_test, squared=False))
print(r2_score(y_test, y_pred_test))

# CROSS-VALIDATION
scores = cross_val_score(
    pipeline, X_train, y_train, cv=3, scoring="neg_root_mean_squared_error"
)
neg_rmse = scores.mean()
-neg_rmse

scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="r2")
r2 = scores.mean()
r2
