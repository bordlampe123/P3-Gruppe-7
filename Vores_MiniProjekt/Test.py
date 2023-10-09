
import cv2 as cv
import numpy as np

Image = cv.imread('Vores_MiniProjekt\\1.jpg')
template = np.zeros(Image.shape, dtype=np.uint8)

Blå = [255, 0, 0]  # BGR format for blue
Cyan = [0, 255, 255]  # BGR format for cyan
Grøn = [35, 149,     103]  # BGR format for green
Rød = [0, 0, 255]  # BGR format for red
Gul = [0, 255, 255]  # BGR format for yellow
Farver = [Blå, Cyan, Grøn, Rød, Gul]

y_1 = -100
y_2 = 0
for i in range(5):
    x_1 = 0
    x_2 = 100
    y_1 = y_1 + 100
    y_2 = y_2 + 100

    for j in range(5):
        template = cv.rectangle(template, (x_1, y_1), (x_2, y_2), tuple(Farver[j]), -1)
        x_1 = x_1 + 100
        x_2 = x_2 + 100
        print(x_1, y_1, x_2, y_2)

print(template.shape)

cv.imshow("Image", template)

cv.waitKey()
cv.destroyAllWindows()