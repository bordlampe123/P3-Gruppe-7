import num as np
import math
import matplotlib.pyplot as plt

mean = 2
sigma = 2
samples = 1000

sample = np.random.normal(mean, sigma, samples)

count = plt.hist(sample, 100)

print(count)
