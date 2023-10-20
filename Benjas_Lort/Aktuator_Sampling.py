import numpy as np
import matplotlib as plt



Sampling_Array = np.zeros(4)
#[12,4],[14,2],[15,1]
VCC = 5.00
VTS = 2.03
bits = 4
Rep_Values = 2**bits
Første_Værdi = Rep_Values/2

if VTS >= (Første_Værdi/Rep_Values)*VCC:
    Sampling_Array[0]=1
else:
    Sampling_Array[0]=0

if Sampling_Array[0]==1:
    if VTS >= (12/Rep_Values)*VCC:
        Sampling_Array[1]=1
    else:
        Sampling_Array[1]=0
else:
    if VTS >= (4/Rep_Values)*VCC:
        Sampling_Array[1]=1
    else:
        Sampling_Array[1]=0


if Sampling_Array[1]==1:
    if VTS >= (14/Rep_Values)*VCC:
        Sampling_Array[2]=1
    else:
        Sampling_Array[2]=0
else:
    if VTS >= (2/Rep_Values)*VCC:
        Sampling_Array[2]=1
    else:
        Sampling_Array[2]=0


if Sampling_Array[2]==1:
    if VTS >= (15/Rep_Values)*VCC:
        Sampling_Array[2]=1
    else:
        Sampling_Array[2]=0
else:
    if VTS >= (1/Rep_Values)*VCC:
        Sampling_Array[2]=1
    else:
        Sampling_Array[2]=0

print(Sampling_Array)

















