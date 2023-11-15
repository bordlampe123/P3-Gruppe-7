import numpy as np

Vcc = 5

bit_seq = [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]

print(bit_seq)

def binary_weighted(a, vcc):
    summed = 0

    a.reverse()
    print(a)

    for x in range(len(a)):
        if x == 0:
            summed += a[x]*vcc
        else:
            summed += (a[x]*vcc)/(2**x)
        print(summed, x, a[x])


    print(summed)

binary_weighted(bit_seq, Vcc)