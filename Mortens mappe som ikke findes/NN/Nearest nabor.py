import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

Class1 = np.loadtxt("C:/Users/stron/Desktop/P3-Gruppe-7/Mortens mappe som ikke findes/NN/trainClass1.dat")
Class2 = np.loadtxt("C:/Users/stron/Desktop/P3-Gruppe-7/Mortens mappe som ikke findes/NN/trainClass2.dat")
Class3 = np.loadtxt("C:/Users/stron/Desktop/P3-Gruppe-7/Mortens mappe som ikke findes/NN/trainClass3.dat")
Class4 = np.loadtxt("C:/Users/stron/Desktop/P3-Gruppe-7/Mortens mappe som ikke findes/NN/trainClass4.dat")

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(Class1[:,0 ], Class1[:,1], Class1[:,2], "1")
ax.scatter(Class2[:,0 ], Class2[:,1], Class2[:,2], "2")
ax.scatter(Class3[:,0 ], Class3[:,1], Class3[:,2], "3")
ax.scatter(Class4[:,0 ], Class4[:,1], Class4[:,2], "4")

#plt.matplotlib_fname(Class1)

plt.show()

