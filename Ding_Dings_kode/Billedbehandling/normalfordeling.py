import numpy as np
import math
import matplotlib.pyplot as plt

def normal(samples, mean, sigma):
    sample = np.random.normal(samples, mean, sigma)
    
    plt.hist(sample, 100)

    plt.show()

normal(2, 2, 1000)