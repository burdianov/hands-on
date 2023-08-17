import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
    MinMaxScaler,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

X = pd.DataFrame({"A": [1, 2, 3, 2], "B": [11, 1, 8, 3]})
scaler = StandardScaler()

scaler.fit(X)
print(scaler.mean_, scaler.var_)

X_scaled = scaler.transform(X)

df = pd.read_csv("../../data/datapreprocessing.csv")

imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp.fit(df[["Years", "Strength", "Height"]])
df[["Years", "Strength", "Height"]] = imp.transform(df[["Years", "Strength", "Height"]])
imp.statistics_

imp.set_params(strategy="most_frequent")
imp.fit(df[["Color"]])
df[["Color"]] = imp.transform(df[["Color"]])
imp.statistics_

le = LabelEncoder()
df["Dangerous"] = le.fit_transform(df["Dangerous"])

oe = OrdinalEncoder(dtype=np.int32)
df[["Weight"]] = oe.fit_transform(df[["Weight"]])

ohe = OneHotEncoder(dtype=np.int32, sparse_output=False, drop="first")
color_encoded = ohe.fit_transform(df[["Color"]])
df2 = pd.DataFrame(color_encoded, columns=ohe.get_feature_names_out())
df = pd.concat((df, df2), axis=1)

mms = MinMaxScaler()
df[["Years", "Strength", "Height"]] = mms.fit_transform(
    df[["Years", "Strength", "Height"]]
)

data = pd.DataFrame([[1], [4], [np.NAN], [8], [11]], columns=["A"])

pl = Pipeline([("imp", SimpleImputer(strategy="mean")), ("scaler", MinMaxScaler())])
print(pl.fit_transform(data))

data = pd.DataFrame([[1], [4], [np.NAN], [8], [11]], columns=["A"])
ct = ColumnTransformer(
    [
        ("imp", SimpleImputer(strategy="mean"), ["A"]),
        ("scaler", MinMaxScaler(), ["A"]),
    ]
)
print(ct.fit_transform(data))

data = pd.DataFrame(
    {
        "A": [1, 2, 3, 4, 5],
        "B": ["Apple", "Orange", "Apple", "Banana", "Apple"],
        "C": [11, 12, 13, 14, 15],
    }
)

ct2 = ColumnTransformer(
    [
        ("encode", OrdinalEncoder(), ["B"]),
        ("normalize", MinMaxScaler(), ["A"]),
    ],
    remainder="passthrough",
)
print(ct2.fit_transform(data))
