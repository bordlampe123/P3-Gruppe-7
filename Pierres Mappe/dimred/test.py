import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math as m

data = np.loadtxt("Pierres Mappe/dimred/iris.dat")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

#ax.scatter(data[:,0], data[:,1], data[:, 2], 'wow1')
#ax.scatter(data[:,0], data[:,2], data[:, 3], 'wow2')
ax.scatter(data[:,0], data[:,3], data[:, 2], 'wow3')


plt.show()




