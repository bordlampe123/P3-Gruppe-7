import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


data = np.loadtxt("Pierres Mappe/dimred/iris.dat")

print(data)

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12,12)
sns.scatterplot(x = data[:,0], y = data[:,3], hue = data[:,4])

plt.show()


