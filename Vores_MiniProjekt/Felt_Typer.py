import numpy as np
import cv2 as cv

image = cv.imread('Vores_MiniProjekt\\1.jpg')

def set_SV(low, high, LS, US, LV, UV): #LH, UH, LS, US, LV, UV
    while True:
        image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        lower_HSV_Bound = np.array([low, LS, LV])  # Adjust these values
        upper_HSV_Bound = np.array([high, US, UV])  # Adjust these values
        
        mask = cv.inRange(image_hsv, lower_HSV_Bound, upper_HSV_Bound)

        result = cv.bitwise_not(image, image, mask=mask)

        cv.imshow("Video", result) #Here we create a window which shows the video/image
        cv.waitKey()
        cv.destroyAllWindows() # close all windows
        return


#LH, UH, LS, US, LV, UV = [39, 71, 127, 255, 106, 255] #Eng
#LH, UH, LS, US, LV, UV = [126, 153, 227, 255, 63, 255] #Vand
#LH, UH, LS, US, LV, UV = [34, 106, 55, 187, 39, 84] #Skov
#LH, UH, LS, US, LV, UV = [27, 47, 97, 155, 88, 255] #Mose
#LH, UH, LS, US, LV, UV = [34, 40, 222, 255, 133, 255] #Ørken


Eng = [39, 71, 127, 255, 106, 255] #Eng
Vand = [126, 153, 227, 255, 63, 255] #Vand
Skov = [34, 106, 55, 187, 39, 84] #Skov
Mose = [27, 47, 97, 155, 88, 255] #Mose
Ørken = [34, 40, 222, 255, 133, 255] #Ørken


def main():
    print('hello world')
    print('getting bounds')
    LH, UH, LS, US, LV, UV = Eng
    set_SV(LH, UH, LS, US, LV, UV)
    LH, UH, LS, US, LV, UV = Vand
    set_SV(LH, UH, LS, US, LV, UV)
    LH, UH, LS, US, LV, UV = Skov
    set_SV(LH, UH, LS, US, LV, UV)
    LH, UH, LS, US, LV, UV = Mose
    set_SV(LH, UH, LS, US, LV, UV)
    LH, UH, LS, US, LV, UV = Ørken
    set_SV(LH, UH, LS, US, LV, UV)

main()