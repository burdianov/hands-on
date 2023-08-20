import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("../../data/diabetes.csv")
df.head()
df.isnull().sum()
df.isna().sum()

X = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=["Glucose", "BMI"], class_names=["No", "Yes"])
plt.show()

clf.set_params(max_depth=3)
clf.fit(X_train, y_train)
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=["Glucose", "BMI"], class_names=["No", "Yes"])
plt.show()

clf.predict([[90, 20], [200, 30]])

scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
accuracy = scores.mean()
accuracy

clf_rf = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)

clf_rf.fit(X_train, y_train)
scores = cross_val_score(clf_rf, X_train, y_train, cv=5, scoring="accuracy")
accuracy = scores.mean()
accuracy
