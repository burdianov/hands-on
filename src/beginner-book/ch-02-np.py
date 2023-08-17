import numpy as np

list1 = [1, 2, 3, 4]
list2 = [[1, 2, 3, 4]]
list3 = [[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400]]

arr1 = np.array(list1)
arr2 = np.array(list2)
arr3 = np.array(list3)

print(type(arr1), type(arr2), type(arr3))

print(arr1.shape, arr2.shape, arr3.shape)

arr4 = np.linspace(0, 10, 5)

print(arr1[2])
print(arr1[-1])

arr2 = np.array([[1, "b", "c"], ["d", "e", "f"]])
print(arr2[1][0])
print(arr2[1, 0])

arr3 = np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
print(arr3[1:6:2])

arr4 = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [6, 7, 8, 9, 10]])
print(arr4[0:2, 2:4])

arr1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(arr1.sum())
print(arr1.sum(axis=0))
print(arr1.sum(axis=1))

arr2 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print(arr2.reshape((4, 2)))

arr3 = np.array([1, 2, 3, 4, 5, 6])
reshaped_array_2 = arr3.reshape((2, -1))
print(arr3)
print(reshaped_array_2)

arr4 = np.array([1, 2, 3, 4, 5, 6])
reshaped1 = arr4.reshape((2, -1))
reshaped2 = np.reshape(arr4, (2, -1))
