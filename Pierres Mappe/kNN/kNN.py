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

plt.show()

def near(input, k):

    distList = []

    for nr in range(class1.shape[0]):
        dist = m.sqrt((input[0]-class1[nr,0])**2+(input[1]-class1[nr,1])**2+(input[2]-class1[nr,2])**2+(input[3]-class1[nr,3])**2)
        distList.append([dist, 1])

    for nr in range(class2.shape[0]):
        dist = m.sqrt((input[0]-class2[nr,0])**2+(input[1]-class2[nr,1])**2+(input[2]-class2[nr,2])**2+(input[3]-class2[nr,3])**2)
        distList.append([dist, 2])

    for nr in range(class3.shape[0]):
        dist = m.sqrt((input[0]-class3[nr,0])**2+(input[1]-class3[nr,1])**2+(input[2]-class3[nr,2])**2+(input[3]-class3[nr,3])**2)
        distList.append([dist, 3])

    for nr in range(class4.shape[0]):
        dist = m.sqrt((input[0]-class4[nr,0])**2+(input[1]-class4[nr,1])**2+(input[2]-class4[nr,2])**2+(input[3]-class4[nr,3])**2)
        distList.append([dist, 4])

    distList.sort()
    #print(distList)

    kNearest = []

    for z in range(k):
        kNearest.append(distList[z])
    
    print(kNearest)
    #print(distList)

near(unknown[0], 2)