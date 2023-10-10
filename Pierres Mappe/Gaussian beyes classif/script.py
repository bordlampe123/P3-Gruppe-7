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

