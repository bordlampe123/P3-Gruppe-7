import numpy as np

def binary(x):

    sum = 0

    for z in range(len(x)):
        sum += x[z]*2**(len(x)-z-1)

    return sum    
        

print(binary([1,1,0,1]))

def SAR(z, input, vcc):

    SA = np.zeros(z)
    maxBit = binary(np.ones(z))+1

    for x in range(len(SA)):
        SA[x] = 1
        tempV = (binary(SA)/maxBit)*vcc
        if input<tempV:
            SA[x] = 0

    return [(binary(SA)/maxBit)*vcc, SA]


print(SAR(4, 3.75, 5))


