import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math as m

class1 = np.loadtxt("Pierres Mappe/kNN/trainClass1.dat")
class2 = np.loadtxt("Pierres Mappe/kNN/trainClass2.dat")
class3 = np.loadtxt("Pierres Mappe/kNN/trainClass3.dat")
class4 = np.loadtxt("Pierres Mappe/kNN/trainClass4.dat")
unknown = np.loadtxt("Pierres Mappe/kNN/unknown.dat")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(class1[:,0], class1[:,1], class1[:,2], 'wow1')
ax.scatter(class2[:,0], class2[:,1], class2[:,2], 'wow2')
ax.scatter(class3[:,0], class3[:,1], class3[:,2], 'wow3')
ax.scatter(class4[:,0], class4[:,1], class4[:,2], 'wow4')
ax.scatter(unknown[:,0], unknown[:,1], unknown[:,2], 'wow5')

#plt.show()

#print(class1.shape[0])


def near(input, k):

    x = input[0]
    y = input[1]
    z = input[2]
    t = input[3]

    distList = []

    for nr in range(class1.shape[0]):
        dist = m.sqrt((input[0]-))

near(unknown, 2)
