import numpy as np
import matplotlib.pyplot as plt

#defining training data
class1 = np.loadtxt("Pierres Mappe/kNN/trainClass1.dat")
class2 = np.loadtxt("Pierres Mappe/kNN/trainClass2.dat")
class3 = np.loadtxt("Pierres Mappe/kNN/trainClass3.dat")
class4 = np.loadtxt("Pierres Mappe/kNN/trainClass4.dat")

#defining test data
unknown = np.loadtxt("Pierres Mappe/kNN/unknown.dat")

#plotting training data in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.scatter(class1[:,0], class1[:,1], class1[:,2], c='r', marker='o', label = 'class1')
ax.scatter(class2[:,0], class2[:,1], class2[:,2], c='b', marker='o', label = 'class2')
ax.scatter(class3[:,0], class3[:,1], class3[:,2], c='g', marker='o', label = 'class3')
ax.scatter(class4[:,0], class4[:,1], class4[:,2], c='y', marker='o', label = 'class4')
ax.scatter(unknown[:,0], unknown[:,1], unknown[:,2], c='k', marker='o', label = 'unknown')

#euclidian distance
#def distance(x1, x2):
    #return np.sqrt(np.sum((x1-x2)**2))

#def nearN(input, k):


plt.show()



