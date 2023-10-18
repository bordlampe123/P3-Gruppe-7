import numpy as np
import matplotlib.pyplot as plt
import scipy
import math as m
from sklearn import preprocessing

Data0 = np.loadtxt('Benjas_Lort\\trainClass1.dat')
Data1 = np.loadtxt('Benjas_Lort\\trainClass2.dat')
Data2 = np.loadtxt('Benjas_Lort\\trainClass3.dat')
Data3 = np.loadtxt('Benjas_Lort\\trainClass4.dat')
Data4 = np.loadtxt('Benjas_Lort\\unknown.dat')

def TwoD_plotter(x, y, x1, y1, x2, y2, x3, y3, x4, y4):
    #colors = color_Matter*10

    fig, ax = plt.subplots()

    #ax.scatter(x, y, s=sizes, c='green', vmin=0, vmax=100)
    ax.scatter(x, y, c='red', vmin=0, vmax=100)
    ax.scatter(x1, y1,  c='blue', vmin=0, vmax=100)
    ax.scatter(x2, y2,  c='orange', vmin=0, vmax=100)
    ax.scatter(x3, y3,  c='green', vmin=0, vmax=100)
    ax.scatter(x4, y4, c='black', vmin=0, vmax=100)


    plt.show()

def TwoD_Bytte(x,y):
    TwoD_plotter(Data0[:,x], Data0[:,y], Data1[:,x], Data1[:,y], Data2[:,x], Data2[:,y], Data3[:,x], Data3[:,y], Data4[:,x], Data4[:,y])


def ThreeD_plotter(x, y, z, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4):
    #colors = color_Matter*10
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    #ax.scatter(x, y, s=sizes, c='green', vmin=0, vmax=100)
    ax.scatter(x, y, z, c='red', vmin=0, vmax=100)
    ax.scatter(x1, y1, z1, c='blue', vmin=0, vmax=100)
    ax.scatter(x2, y2, z2, c='orange', vmin=0, vmax=100)
    ax.scatter(x3, y3, z3, c='green', vmin=0, vmax=100)
    ax.scatter(x4, y4, z4, c='black', vmin=0, vmax=100)


    plt.show()



def ThreeD_Bytte(x, y, z):
    ThreeD_plotter(Data0[:,x], Data0[:,y], Data0[:,z],
                    Data1[:,x], Data1[:,y], Data1[:,z],
                    Data2[:,x], Data2[:,y], Data2[:,z],
                    Data3[:,x], Data3[:,y], Data3[:,z],
                    Data4[:,x], Data4[:,y], Data4[:,z])
    #ThreeD_plotter(Data0[0], Data0[1], Data0[2], Data1[0], Data1[1], Data1[2], Data2[0], Data2[1], Data2[2], Data3[0], Data3[1], Data3[2])
#TwoD_Bytte(0,1)
#ThreeD_Bytte(0,1,2)




def distanceCAL_ThreeD(Subject, Neighbourh1, Neighbourh2, Neighbourh3, Neighbourh4):
    distance1 = m.sqrt(((Neighbourh1[0]-Subject[0])**2)+((Neighbourh1[1]-Subject[1])**2)+((Neighbourh1[2]-Subject[2])**2))
    distance2 = m.sqrt(((Neighbourh2[0]-Subject[0])**2)+((Neighbourh2[1]-Subject[1])**2)+((Neighbourh2[2]-Subject[2])**2))
    distance3 = m.sqrt(((Neighbourh3[0]-Subject[0])**2)+((Neighbourh3[1]-Subject[1])**2)+((Neighbourh3[2]-Subject[2])**2))
    distance4 = m.sqrt(((Neighbourh4[0]-Subject[0])**2)+((Neighbourh4[1]-Subject[1])**2)+((Neighbourh4[2]-Subject[2])**2))
    distancearray = np.array([distance1, distance2, distance3, distance4])
    return distancearray

print(Data4.shape)
print(Data3.shape)
distances = np.zeros((40, 400, 4))

def Assert_distances(i):
    for k in range(400):
        distances[i,k] = distanceCAL_ThreeD(Data4[i], Data0[k], Data1[k], Data2[k], Data3[k])



print(distances)














