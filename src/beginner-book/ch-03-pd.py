import pandas as pd
import numpy as np

list1 = [1, 2, 3, 4, 5]
print(list1, end="\n\n")

series1 = pd.Series(list1)
print(series1)
print(series1, end="\n\n")

series2 = pd.Series(data=list1, index=["P", "Q", "R", "S", "T"])

myList = [[1, 2, 3], [4, 5, 6]]
df1 = pd.DataFrame(myList)

myDict = {"A": [1, 2, 3], "B": [4, 5, 6]}
df2 = pd.DataFrame(myDict)

myList2 = [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]
df3 = pd.DataFrame(
    data=myList2, index=["A", "B"], columns=["1st", "2nd", "3rd", "4th", "5th"]
)
print(df3)

classData = pd.read_csv("../../data/pandasDemo.csv")
classData.head()
classData.describe()

class_data = classData.rename(
    columns={"TA": "Assistant"}, index={0: "Row Zero", 1: "Row One"}
)

X1 = classData["ModuleID"]
X2 = classData[["ModuleID"]]
X3 = classData[["ModuleID", "Instructor"]]
print(X3)

arr1 = class_data["ModuleID"] == "CS101"
class_data[arr1]
arr1 = class_data[class_data["ModuleID"] == "CS101"]

arr1 = class_data["ModuleID"] == "CS101"
arr2 = class_data["Instructor"] == "Aaron"
class_data[arr1 & arr2]
class_data[(class_data["ModuleID"] == "CS101") & (class_data["Instructor"] == "Aaron")]

class_data.iloc[[0]]
class_data.loc[["Row Zero"]]

class_data.iloc[[0, 3, 4]]
class_data.loc[["Row Zero", 3, 4]]

class_data.iloc[:3]

classData.iloc[:5, :2]

class_data.iloc[:, 2:3]

class_data["AverageGPA"] = [44, 46, 47, 41, 45, 49, 40, 41, 45, 48, 42]

class_data["AverageGPA"] = class_data["AverageGPA"] * 0.1

class_data.sort_values(["Rating", "AverageGPA"], ascending=False)

myData = pd.DataFrame([[None], [np.NaN], [""], ["Apple"]], columns=["A"])
print(myData.isnull())
myData.isnull().sum()

class_data.isnull().sum()

classData_rows_deleted = class_data.dropna()
classData_columns_deleted = class_data.dropna(axis=1)

myData2 = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
myArr = myData2.to_numpy()
