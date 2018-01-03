import numpy as np
import matplotlib.pyplot as plt
myarray = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
# print(myarray.shape)
# print(myarray)
# print(myarray[:,1])
# print(myarray[0])
# print(myarray[:, 0:2])
# print(myarray[2,2])

plt.plot(myarray)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()
