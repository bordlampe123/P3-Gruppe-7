import numpy as np
import matplotlib.pyplot as plt

def gay(x):
    mu, sigma = 0, 2 # mean and standard deviation
    s = np.random.normal(mu, sigma, size=100000000)

    abs(mu - np.mean(s))
    0.0  # may vary

    abs(sigma - np.std(s, ddof=1))
    0.1  # may vary

    count, bins, ignored = plt.hist(s, 30, density=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')



def V_Out(TMP35, AD589):
    return 


plt.show()




