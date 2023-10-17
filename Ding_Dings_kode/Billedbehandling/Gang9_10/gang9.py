import numpy as np
import matplotlib.pyplot as plt

data1 = np.loadtxt("Ding_Dings_kode/Billedbehandling/Gang9_10/trainClass1.dat")
data2 = np.loadtxt("Ding_Dings_kode/Billedbehandling/Gang90_10/trainClass2.dat")
data3 = np.loadtxt("Ding_Dings_kode/Billedbehandling/Gang9_10/trainClass3.dat")
data4 = np.loadtxt("Ding_Dings_kode/Billedbehandling/Gang9_10/trainClass4.dat")
datau = np.loadtxt("Ding_Dings_kode/Billedbehandling/Gang9_10/unknown.dat")


def splot(data,size):
    data = np.reshape(data, (5, size))
    data0 = data[0]
    data1 = data[1]
    data2 = data[2]
    data3 = data[3]
    data4 = data[4]


    fig, ax = plt.subplots()
    #ax.scatter(x=data[0], y=data[1], s=data[2], c=data[3], vmin=0, vmax=400)
    ax.scatter(x=data0, y=data1, s=data2, c=data3, vmin=0, vmax=400)
    plt.show()

splot(datau, 40)