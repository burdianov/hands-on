import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sp_x = [2, 3, 4, 6, 2, 8, 6, 9, 12, 1, 7, 1]
sp_y = [10, 3, 4, 12, 5, 12, 4, 5, 6, 8, 9, 10]
plt.scatter(sp_x, sp_y)
# plt.show()

b_x = ["0-3", "4-6", "7-9"]
b_y = [20, 50, 30]
plt.bar(b_x, b_y)

h_x = [7, 7, 7, 1, 1, 0, 0, 4, 5, 5, 6, 6, 8, 9, 9, 10]
plt.hist(h_x, bins=5)
plt.hist(h_x)

l_x = [7, 1, 4, 8, 5, 2, 3]
l_y = [98, 2, 32, 128, 15, 28, 18]
plt.plot(l_x, l_y)

zipped = zip(l_x, l_y)
sorted_zip = sorted(zipped)
l_x, l_y = zip(*sorted_zip)
plt.plot(l_x, l_y)

sp_x = [2, 3, 4, 6, 2, 8, 6, 9, 12, 1, 7, 1]
sp_y = [10, 3, 4, 12, 5, 12, 4, 5, 6, 8, 9, 10]
df = pd.DataFrame({"A": sp_x, "B": sp_y})
df.plot(kind="scatter", x="A", y="B")

b_x = ["0-3", "4-6", "7-9"]
b_y = [20, 50, 30]
df = pd.DataFrame({"A": b_x, "B": b_y})
df.plot(kind="bar", x="A", y="B")

l_x = [7, 1, 4, 8, 5, 2, 3]
l_y = [98, 2, 32, 128, 15, 28, 18]
df = pd.DataFrame({"A": l_x, "B": l_y})
df = df.sort_values(["A"])
df.plot(kind="line", x="A", y="B")

h_x = [7, 7, 7, 1, 1, 0, 0, 4, 5, 5, 6, 6, 8, 9, 9, 10]
df = pd.DataFrame({"A": h_x})
df["A"].plot(kind="hist", bins=5)

df = pd.DataFrame({"A": [2, 3, 1, 1, 4, 4], "B": [3, 4, 4, 1, 2, 2]})
df.hist()

x1 = [1, 2, 4, 6, 8, 9, 10]
y1 = [3, 4, 6, 12, 4, 5, 7]
y2 = [5, 1, 2, 6, 8, 9, 1]

plt.figure(figsize=(10, 5))
plt.scatter(x1, y1, color="black", s=50, marker="x", label="With Reward")
plt.scatter(x1, y2, color="darkgray", s=40, marker="o", label="Without Reward")
plt.legend(loc="best")
plt.grid()
plt.xticks([0, 2, 4, 6, 8, 10])
plt.yticks(range(0, 13))
plt.xlabel("Age")
plt.ylabel("Number of Tries")
plt.title("Designing our Charts")

fig, my_ax = plt.subplots()
a = [1, 2, 3, 4]
b = [7, 3, 1, 4]
my_ax.scatter(a, b)
