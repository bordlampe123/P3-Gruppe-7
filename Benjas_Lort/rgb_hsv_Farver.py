import cv2 as cv
import numpy as np




def lav_en_pixels_billede_HSV(input):
    hsv_color = (input)  # Hue, Saturation, Value

    # Create a 3-channel blank HSV image filled with the desired color
    for i in range(len(hsv_color)):
        hsv_image = np.zeros((1, 1, 3), dtype=np.uint8)
        hsv_image[0, 0, 0] = hsv_color[i][0]  # Set Hue
        hsv_image[0, 0, 1] = hsv_color[i][1]  # Set Saturation
        hsv_image[0, 0, 2] = hsv_color[i][2]  # Set Value

        bgr_color = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)[0, 0]
        print(bgr_color)


def lav_en_pixels_billede_BGR(input):
    hsv_color = (input)  # Hue, Saturation, Value

    # Create a 3-channel blank HSV image filled with the desired color
    for i in range(len(hsv_color)):
        hsv_image = np.zeros((1, 1, 3), dtype=np.uint8)
        hsv_image[0, 0, 0] = hsv_color[i][0]  # Set Hue
        hsv_image[0, 0, 1] = hsv_color[i][1]  # Set Saturation
        hsv_image[0, 0, 2] = hsv_color[i][2]  # Set Value

        bgr_color = cv.cvtColor(hsv_image, cv.COLOR_RGB2HSV_FULL)[0, 0]
        print(bgr_color)


#Gul - Mark (7,155,171)
#grøn - eng (21,162,111)
#mørkegrøn - skov  (37,52,38)
#blå - vand  (192,99,14)
#brun - vissen  (43,95,108)
#sort - mine (0, 0, 0) 
#Anden brun Bordplade (19, 92, 124)
hsv = [[42, 195, 149],[60, 195, 149],[120, 195, 149],[187, 195, 149]]
BGR = [[7,155,171],[21,162,111],[37,52,38],[192,99,14], [43,95,108], [43,95,108], [0, 0, 0], [19, 92, 124]]

#lav_en_pixels_billede_HSV(hsv)
lav_en_pixels_billede_BGR(BGR)