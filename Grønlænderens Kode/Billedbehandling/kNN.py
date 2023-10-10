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

plt.plot(class1[:,0], class1[:,1], class1[:,2], 'ro')
plt.plot(class2[:,0], class2[:,1], class2[:,2], 'bo')
plt.plot(class3[:,0], class3[:,1], class3[:,2], 'go')
plt.plot(class4[:,0], class4[:,1], class4[:,2], 'yo')
plt.plot(unknown[:,0], unknown[:,1], unknown[:,2], 'ko')


plt.show()



