import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

data1 = np.loadtxt("Ding_Dings_kode/Billedbehandling/Gang9/trainClass1.dat")
data2 = np.loadtxt("Ding_Dings_kode/Billedbehandling/Gang9/trainClass2.dat")
data3 = np.loadtxt("Ding_Dings_kode/Billedbehandling/Gang9/trainClass3.dat")
data4 = np.loadtxt("Ding_Dings_kode/Billedbehandling/Gang9/trainClass4.dat")
datau = np.loadtxt("Ding_Dings_kode/Billedbehandling/Gang9/unknown.dat")


def splot(data,size):
    data = np.reshape(data, (4, size))
    data0 = data[0]
    data1 = data[1]
    data2 = data[2]
    data3 = data[3]


    fig, ax = plt.subplots()
    #ax.scatter(x=data[0], y=data[1], s=data[2], c=data[3], vmin=0, vmax=400)
    ax.scatter(x=data0, y=data1, s=data2, c=data3, vmin=0, vmax=400)
    plt.show()

splot(datau, 40)