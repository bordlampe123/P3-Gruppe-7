import numpy as np
import math as m
import matplotlib.pyplot as plt


mean = 2

sigma = 2

sampleCount = 10000000

samples = np.random.normal(size=sampleCount, loc = mean, scale = sigma)


x = np.linspace(-10, 10, 100)

plt.plot(x, (1/(sigma*m.sqrt(2*m.pi())))*m.exp(-((x-mean)**2)/(2*sigma**2)), color='red')

plt.show()



_ = plt.hist(samples, bins = 'auto')

plt.show()

print(samples)