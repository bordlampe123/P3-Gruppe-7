import cv2 as cv
import numpy as np




path = 'Vores_MiniProjekt\King Domino dataset\King Domino dataset\Cropped and perspective corrected boards\\'

image_count = 74
image_dict = {}
color_level = 3
Resolution = 25
Library = np.zeros((image_count, Resolution, color_level), dtype='uint8')

for i in range (1, image_count + 1):
    File_Name = path + str(i) + '.jpg'
    image = cv.imread(File_Name)
    image_dict[f'image{i}'] = image


SBS = 100
iterations = int(500/SBS)
for N in range(1, image_count + 1):
    image = image_dict[f'image{N}']
    print("iterations", iterations)
    def gembillede(input, x, y):
        outputimg = input[x*SBS:(x+1)*SBS, y*SBS:(y+1)*SBS]
        return outputimg

    list = []
    for i in range(iterations):
        for j in range(iterations):
            sub_image = gembillede(image, i, j)
            list.append(sub_image)

    #cv.imshow("test", list[2])

    #Gul - Mark (7,155,171)
    #grøn - eng (21,162,111)
    #mørkegrøn - skov  (37,52,38)
    #blå - vand  (192,99,14)
    #brun - vissen  (43,95,108)
    #sort - mine (0, 0, 0) 
    #Anden brun Bordplade (19, 92, 124)

    # Wait for a key press and close all OpenCV windows
    #cv.imshow("org",image)

    def prominent(img):
        data = np.reshape(img, (-1,3))
        #print(data.shape)
        data = np.float32(data)

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv.KMEANS_PP_CENTERS
        compactness,labels,centers = cv.kmeans(data,1,None,criteria,10,flags)

        #print('Dominant color is: bgr({})'.format(centers[0].astype(np.int32)))
        return centers[0].astype('uint8')

    BGR = []
    for i in range(iterations*iterations):
        BGR.append([int(prominent(list[i])[0]), int(prominent(list[i])[1]), int(prominent(list[i])[2])])

    template = np.zeros_like(image)
    template_Improved = np.zeros_like(image)
    c = 0
    y_1 = -SBS
    y_2 = 0
    for i in range(iterations):
        x_1 = 0
        x_2 = SBS
        y_1 = y_1 + SBS
        y_2 = y_2 + SBS
        for j in range(iterations):
            template = cv.rectangle(template, (x_1, y_1), (x_2, y_2), tuple(BGR[c]), -1)
            #print("Farve i ", c, " = ", BGR[c])

            x_1 = x_1 + SBS
            x_2 = x_2 + SBS
            c = c+1
    Ting = 100
    Ting2 = 0
    HSV_Edition = cv.cvtColor(template, cv.COLOR_BGR2HSV)
    for R in range(5):
        for S in range(5):
            Specific_AVG_Color = template[50+(Ting*R)][50+(Ting*S)][:]
            #Specific_AVG_Color = HSV_Edition[50+(Ting*R)][50+(Ting*S)][:]
            Library[N-1][Ting2] = Specific_AVG_Color
            print(Library[N-1][Ting2][:])
            Ting2 = Ting2 +1
            print(R,S, " = ", Specific_AVG_Color)
    print(Ting2)

    temp2 = np.zeros_like(template)
    temp2 = sub_image[4][0]
    #print("shape", template.shape)
    #print("shape", sub_image.shape)
    Window_name = f'vindue{N}'
    #cv.imshow(Window_name, template)
    #cv.imshow("testing", image_dict[f'image{N}'])
    #print(N)
    #cv.imshow("nyt lille", temp2)






    cv.destroyAllWindows()
HolderArray = np.zeros((image_count, Resolution, color_level+1))
#Library.resize(image_count, Resolution, color_level+1)
print("her")
for k in range(image_count):
    for i in range(Resolution):
        for j in range (color_level):
            HolderArray[k][i][j] = Library[k][i][j]
        print(HolderArray[k][i][:])
        #if 75 < HolderArray[k][i][0] < 150:
            #print(i, "= grøn")


cv.imshow(Window_name, template)
cv.waitKey()
cv.destroyAllWindows()







#[ 42 195 149]
#[105 242 172]
#[ 42 173  65]
#[ 42 163  61]
#[ 41 173  62]
