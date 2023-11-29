import math
import numpy as np
import matplotlib.pyplot as plt

def normalfordeling(samples, mu, sigma, bins):
    #making samples
    samples = np.random.normal(mu, sigma, samples)
    
    #plotting
    plt.hist(samples, bins)
    
    #plotting the probability density function
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, len(samples)*(1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))), linewidth=2, color='r')

    
    print(x)
    


    #showing
    plt.show()


normalfordeling(10000, 2, 2, 100)