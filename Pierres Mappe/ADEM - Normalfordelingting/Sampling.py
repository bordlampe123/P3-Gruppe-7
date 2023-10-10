import numpy as np
import math as m
import matplotlib.pyplot as plt


mean = 2

sigma = 2

sampleCount = 10000

samples = np.random.normal(size=sampleCount, loc = mean, scale = sigma)





x  = np.linspace(mean - 3*sigma, mean + 3*sigma, sampleCount)
scaling = sampleCount
print(scaling)
plt.plot(x, scaling*(1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * sigma**2))), linewidth=2, color='r')


_ = plt.hist(samples, bins = 'auto')

plt.show()
