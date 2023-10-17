import numpy as np
import matplotlib.pyplot as plt

iris = np.loadtxt("Ding_Dings_kode/Billedbehandling/Gang9_10/iris.dat")

def splot(data, type):
    data = np.reshape(data, (5, 150))
    sl = data[0]
    sw = data[1]
    pl = data[2]
    pw = data[3]
    race = data[4]

    if type == "2d1":
        fig, ax = plt.subplots()
        ax.scatter(x=sl, y=sw, c=race*10, vmin=0, vmax=400)
    if type == "2d2":
        fig, ax = plt.subplots()
        ax.scatter(x=pl, y=pw, c=race*10, vmin=0, vmax=400)
        plt.show()
    
    if type == "3d":
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xs=sl, ys=sw, zs=pl, s=pw, c=race, vmin=0, vmax=10)
        plt.show()

splot(iris,"2d2")