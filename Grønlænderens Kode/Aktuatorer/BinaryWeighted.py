import numpy as np

binary_1 = np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])

binary_2 = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])

binary_3 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1])


def WeightedDAC(a, vcc):
    N = len(a)
    RB = np.flip(a)
    Sum = 0
    print(RB)
    for x in range(N):
        if RB[x] == 1:
            Sum += 2**(x)
            print(Sum)
        else:
            Sum += 0
    
    print((Sum/2**N)*vcc)

WeightedDAC(binary_1, 5)

WeightedDAC(binary_2, 5)

WeightedDAC(binary_3, 5)