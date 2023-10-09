import cv2 as cv
import numpy as np


def lav_en_pixels_billede_BGR(input):
    bgr_color = (input)  # Hue, Saturation, Value
    template2 = np.zeros_like(Data_Image)
    template2[:] = bgr_color
    cv.imshow('Pixel_BGR', template2)


def lav_en_pixels_billede_HSV(input):
    hsv_color = (input)  # Hue, Saturation, Value

    # Create a 3-channel blank HSV image filled with the desired color
    hsv_image = np.zeros((1, 1, 3), dtype=np.uint8)
    hsv_image[0, 0, 0] = hsv_color[0]  # Set Hue
    hsv_image[0, 0, 1] = hsv_color[1]  # Set Saturation
    hsv_image[0, 0, 2] = hsv_color[2]  # Set Value

    # Convert the HSV image to BGR
    bgr_color = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)[0, 0]
    print(bgr_color)


    template2 = np.zeros_like(Data_Image)
    template2[:] = bgr_color
    cv.imshow('Pixel_HSV', template2)






#Picture path
path = 'Vores_MiniProjekt\King Domino dataset\King Domino dataset\Cropped and perspective corrected boards\\'
#Kedelige vars
Data_Image =cv.imread('Vores_MiniProjekt\King Domino dataset\King Domino dataset\Cropped and perspective corrected boards\\1.jpg')
image_Size = 500
image_count = 1 #4 #images
image_dict = {} #Dictionary for images
color_level = 3 #We are looking at BGR pictures, so 3 levels
LOPF = [] #"Liste Over Prominente Farver: Holder en lang liste med alle
template = np.zeros_like(Data_Image) #billede vi tegner på


#Spændene vars
Størelse_På_Firkanter = 100 #i pixels
Resolution = int((500**2)/(Størelse_På_Firkanter**2)) #hov many times
HolderArray = np.zeros((image_count, Resolution, color_level+1)) #Farverne for alle felter i et array. Med størrelserne 74 billeder resolution på 25 og color depth på 3 bliver det en 74x25x4 matrix
HolderArray2 = np.zeros((image_count, Resolution, color_level+1))
Library = np.zeros((image_count, Resolution, color_level), dtype='uint8')
Library2 = np.zeros((image_count, Resolution, color_level), dtype='uint8')
iterations = int(image_Size/Størelse_På_Firkanter) #størelsen af billedet divideret med størelsen vi vil lave firkanter giver iterationstallet


#Funktioner
#Denne funktion gemmer et outputbillede som er et udklip af det originale billede, baseret på coordinaterne som input og størrelsen af firkanterne
def gembillede(input, x, y):
    outputimg = input[x*Størelse_På_Firkanter:(x+1)*Størelse_På_Firkanter, y*Størelse_På_Firkanter:(y+1)*Størelse_På_Firkanter]
    return outputimg


#Funktionen her finder den "the mean color" af billeder den får som input. dette billede skal være subbilledet fra gembillede funktionen
def prominent(image_input):
        data = np.reshape(image_input, (-1,3))
        data = np.float32(data)

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv.KMEANS_PP_CENTERS
        compactness,labels,centers = cv.kmeans(data,1,None,criteria,10,flags)
        return centers[0].astype('uint8')

#Denne funktion gemmer underbilleder i en lang liste
def Gem_Alle_Billeder(input, ite):
    for i in range(ite):
        for j in range(ite):
            sub_image = gembillede(input, i, j)
            list.append(sub_image)

#Denne funktion gemmer mean colors fra et enkelt billedes underbilleder
def Liste_Med_Underbilleder(UB, ite):
    for i in range(ite*ite):
        LOPF.append([int(prominent(UB[i])[0]), int(prominent(UB[i])[1]), int(prominent(UB[i])[2])])

#Lav en library med alle billeder
def Dictionary_generator(Amount, path, dictionary):
    for i in range (1, Amount + 1):
        File_Name = path + str(i) + '.jpg'
        image_Temp = cv.imread(File_Name)
        dictionary[f'image{i}'] = image_Temp


#Tegn firkanter
def Tegn_Firkanter(L_B, ite, temp):
    c = 0
    y_1 = -Størelse_På_Firkanter
    y_2 = 0
    for i in range(iterations):
        x_1 = 0
        x_2 = Størelse_På_Firkanter
        y_1 = y_1 + Størelse_På_Firkanter
        y_2 = y_2 + Størelse_På_Firkanter
        for j in range(iterations):
            temp = cv.rectangle(temp, (x_1, y_1), (x_2, y_2), tuple(LOPF[c]), -1)
            x_1 = x_1 + Størelse_På_Firkanter
            x_2 = x_2 + Størelse_På_Firkanter
            c = c+1
    


def Find_farverne_På_Firkanterne_igen(temp, Lib):
    Ting = 100
    Ting2 = 0
    for R in range(5):
        for S in range(5):
            Specific_AVG_Color = temp[50+(Ting*R)][50+(Ting*S)][:]
            Lib[N-1][Ting2] = Specific_AVG_Color
            #print(Library[N-1][Ting2][:])
            Ting2 = Ting2 +1
            #print(R,S, " = ", Specific_AVG_Color)






Dictionary_generator(image_count, path, image_dict)

for N in range(1, image_count + 1):
    image = image_dict[f'image{N}']
    list = []
    Gem_Alle_Billeder(image, iterations)
    Liste_Med_Underbilleder(list, iterations)
    Tegn_Firkanter(Størelse_På_Firkanter, iterations, template)

    HSV = cv.cvtColor(template, cv.COLOR_BGR2HSV_FULL)
    Find_farverne_På_Firkanterne_igen(HSV, Library)
    Find_farverne_På_Firkanterne_igen(template, Library2)

    Window_name = f'vindue{N}'
    cv.imshow(Window_name, template)
    cv.imshow("testing", image_dict[f'image{N}'])
    #lav_en_pixels_billede_HSV([42, 195, 149])
    #lav_en_pixels_billede_BGR(HolderArray2[0][0][:])
    cv.waitKey()
    cv.destroyAllWindows()






#Library.resize(image_count, Resolution, color_level+1)
def Array_Til_HSV_Farver(Amount, Res, CL, lib):
    print("BGR")
    for k in range(Amount):
        for i in range(Res):
            for j in range (CL):
                HolderArray[k][i][j] = lib[k][i][j]
                #if 75 < HolderArray[k][i][0] < 150:
                #print(i, "= grøn")

def Array_Til_BGR_Farver(Amount, Res, CL, lib):
    print("RGB")
    for k in range(Amount):
        for i in range(Res):
            for j in range (CL):
                HolderArray2[k][i][j] = lib[k][i][j]
                #if 75 < HolderArray[k][i][0] < 150:
                #print(i, "= grøn")

Array_Til_HSV_Farver(image_count, Resolution, color_level, Library)
Array_Til_BGR_Farver(image_count, Resolution, color_level, Library2)

print(HolderArray.shape)
print('hsv', HolderArray[0][0][:])
print('bgr', HolderArray2[0][0][:])
for i in range(image_count):
    for j in range(Resolution):
        #print("j = ", j, "=", HolderArray[1][j][0])
        if 59 < HolderArray[i][j][0] < 90:
            HolderArray[i][j][3] = 1

for i in range(25):
    if HolderArray[0][i][3] == True:
        print(i, " = q", HolderArray[0][i][3], " and ", HolderArray[0][i][0])
    









#cv.imshow(Window_name, template)
cv.waitKey()
cv.destroyAllWindows()


#Gul - Mark (7,155,171)   - [ 93 245 171]
#grøn - eng (21,162,111)   - [ 79 222 162]
#mørkegrøn - skov  (37,52,38)   - [62 74 52]
#blå - vand  (192,99,14)   - [ 14 236 192]
#brun - vissen  (43,95,108)   - [ 96 153 108]
#sort - mine (0, 0, 0)    - [ 96 153 108]
#Anden brun Bordplade (19, 92, 124)   - [ 99 216 124]







