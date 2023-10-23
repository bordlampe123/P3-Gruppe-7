import numpy as np
import math

OutBit = []

Bits = []

input = 0

def createBits(n):
    Bits = [2**i for i in range(n)]
    Bits.reverse()
    return Bits

def SAR(vcc, input, n):
    BitSum = 0
    Bits = createBits(n)
    print(Bits)
    MaxBits = Bits[0]*2
    print(MaxBits)
    for i in range(n):
        BitSum += Bits[i]
        print(BitSum)
        Tempv = (BitSum/MaxBits)*vcc
        print(Tempv)
        if input >= Tempv:
            OutBit.append(1)
        else:
            OutBit.append(0)
    print(OutBit)

SAR(5, 4.375, 4)