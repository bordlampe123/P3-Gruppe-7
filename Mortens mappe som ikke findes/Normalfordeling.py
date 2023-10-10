import numpy as np

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

bins = 1000

mu, sigma = 2, 2 # mean and standard deviation

s = np.random.normal(mu, sigma, 10000)

count, bins, ignored = plt.hist(s, bins , density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
plt.show()
