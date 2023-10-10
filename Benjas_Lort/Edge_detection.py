import cv2 as cv
import numpy as np
import math

def get_gaussian_kernel(kernel_size, std):
    # Gaussian formular:
    # 1/(2*pi*o*o)*e^(-((x*x+y*y)/(2*o*o)))
    # o = std
    kernel = np.zeros(np.asarray([kernel_size, kernel_size]))
    kernel_radius = kernel_size // 2
    for index_x in range(kernel_size):
        x = index_x - kernel_radius
        for index_y in range(kernel_size):
            y = index_y - kernel_radius
            kernel[index_x][index_y] = (1/(2*math.pi*std**2)*math.exp(-((x**2+y**2)/(2*std**2))))
    return np.asarray(kernel)

def gaussian_blur(img, kernel_size):
    assert not kernel_size % 2 == 0, "Kernel size must be odd number"
    kernel = get_gaussian_kernel(kernel_size, kernel_size/6)
    output = cv.filter2D(img, -1, kernel)
    return np.asarray(output, dtype=np.uint8)

def Grasfire(Input):
    ID = 0
    for X, row in enumerate(Input):
        for Y, pixel in enumerate(row):
            Scope = Input[X,Y,0]
            if Scope == 255:
                BurnQue.append([X,Y, ID])
                Scope_O = Input[X,Y+1,0]
                Scope_V = Input[X-1,Y,0]
                Scope_N = Input[X,Y-1,0]
                Scope_H = Input[X+1,Y,0]
                while True:
                    if Scope_O > 254:
                        BurnQue.append([X,Y+1, ID])
                    elif Scope_V > 254:
                        BurnQue.append([X-1,Y, ID])
                    elif Scope_N > 254:
                        BurnQue.append([X,Y-1, ID])
                    elif Scope_H > 254:
                        BurnQue.append([X+1,Y, ID])

                
    

    
BurnQue = []

image = cv.imread("Vores_MiniProjekt\King Domino dataset\King Domino dataset\Cropped and perspective corrected boards\\1.jpg")


kernel_size = 5
Filtered = gaussian_blur(image, kernel_size)

gradiant_X = np.array([[-1,  0,   1],
                       [-2,  0,   2],
                       [-1,  0,   1]])

gradiant_Y = np.array([[1,   2,   1],
                       [0,   0,   0],
                       [-1, -2,  -1]])

Gradiant = np.zeros((500-kernel_size+1,500-kernel_size+1,2))

image_GS = cv.cvtColor(Filtered, cv.COLOR_BGR2GRAY)
Blurred_Image = image_GS
print(Blurred_Image.shape)
kernel_size = 3
Slice = np.zeros((kernel_size,kernel_size,1), dtype='uint8')

Binary = np.zeros((500,500,3), dtype='uint8')
thresh = 50

for X, row in enumerate(Gradiant):
    for Y, pixel in enumerate(row):
        Slice = Blurred_Image[X:X+kernel_size, Y:Y+kernel_size]


        Gradiant[X][Y][0] = np.sum(Slice*gradiant_X)
        Gradiant[X][Y][1] = np.sum(Slice*gradiant_Y)

        Gradiant_Var = math.sqrt((Gradiant[X][Y][0]**2)+(Gradiant[X][Y][1]**2))
        if Gradiant_Var > thresh:
            Binary[X][Y] = [255, 255, 255]
        else:
            Binary[X][Y] = [0, 0, 0]






#Grasfire(Binary)
#print(BurnQue)
cv.imshow("PIS", Binary)
cv.imshow("Filtered", Filtered)

cv.waitKey()
cv.destroyAllWindows






