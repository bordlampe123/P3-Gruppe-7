import cv2 as cv
import numpy as np
import math

#Funktionerne kan evt. bruges til at print et billede med en bestem farve enten opgivet i RGB/BGR eller HSV
def lav_en_pixels_billede_BGR(input, Shape):
    bgr_color = (input)  # Hue, Saturation, Value
    template2 = np.zeros_like(Shape)
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

    temp = np.zeros((500,500,3), dtype='uint8')
    temp[:] = bgr_color
    cv.imshow('Pixel_HSV', temp)
path = 'Vores_MiniProjekt\King Domino dataset\King Domino dataset\Cropped and perspective corrected boards\\'




#Denne funktion laver et library med alle billederne. Deres navn bliver 'image1' til whatever billede man siger den skal læse.
#Den skal køres først og minimums værdien for læste billeder er 1. billederne bliver gemt i deres normale matrixform, side om side med andre billeder i listen. 
def Dictionary_generator(Amount, path, dictionary):
    for i in range (1, Amount + 1):
        File_Name = path + str(i) + '.jpg'
        image_Temp = cv.imread(File_Name)
        dictionary[f'image{i}'] = image_Temp



#Denne funktion køres først og returnere et udklip at den originale billede
def gembillede(input, x, y, Resolution):
    outputimg = input[x*Resolution:(x+1)*Resolution, y*Resolution:(y+1)*Resolution]
    return outputimg

#Denne funktion gemmer underbilleder i en matrix
def Gem_Alle_Billeder(input, iterations, Output_Matrix, Resolution):
    for i in range(iterations):
        for j in range(iterations):
            Output_Matrix[i][j] = gembillede(input, i, j, Resolution)




#Denne funktion ved har jeg kopiret fra nettet, men den finder mean farven af et billede ligegyldigt størrelse
#Det er meningen at den skal bruges på hvert af underbillederne, og derved få deres gennemsnitsfarve.
#Den returnere de farveværdien i RGB ex. [55, 117, 202]
def prominent(image_input):
        data = np.reshape(image_input, (-1,3))
        data = np.float32(data)

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv.KMEANS_PP_CENTERS
        compactness,labels,centers = cv.kmeans(data,1,None,criteria,10,flags)
        return centers[0].astype('uint8')


def Liste_Med_Underbilleders_Farver(input_sub_image_Matrix, ite, output_matrix1, output_matrix2):
    for i in range(ite):
        for j in range(ite):
            output_matrix1[i][j] = prominent(input_sub_image_Matrix[i][j])
            output_matrix2[i][j] = cv.cvtColor(np.uint8([[output_matrix1[i][j]]]), cv.COLOR_BGR2HSV)[0][0]
            #output_matrix2[i][j] = pixel_hsv = cv.cvtColor(np.uint8([[pixel_bgr_Array]]), cv.COLOR_BGR2HSV)[0][0]
                


def Liste_Med_Underbilleders_Farver_Special(input_sub_image_Matrix, ite, output_matrix1):
    for i in range(ite):
        for j in range(ite):
            if i%4 == False or (i+1)%4 == False:
                output_matrix1[i][j] = prominent(input_sub_image_Matrix[i][j])
            else:
                if j%4 == False or (j+1)%4 == False:#1,3,4,7,8
                    output_matrix1[i][j] = prominent(input_sub_image_Matrix[i][j]) 
                #elif (j-1)%4 == False:#1,5,9
                    #output_matrix1[i][j] = prominent(input_sub_image_Matrix[i][j-1])
                #elif (j+2)%4 == False:#2,6,10
                    #output_matrix1[i][j] = prominent(input_sub_image_Matrix[i][j+1])
                else: output_matrix1[i][j] = [0,0,0]



def ColorA_To_ColorA2(input):

    A = np.sum([input[0][0],input[0][1],input[0][2],input[0][3],input[1][0],input[1][1],input[1][2],input[1][3],input[2][0],input[2][1],input[2][2],input[2][3],input[3][0],input[3][1],input[3][2],input[3][3]], axis=0)
    A = A/16
    A = A.astype(int)  

    return input




def Tegn_Firkanter(ite, temp, resolution, color_Matrix):
    y_1 = int(-resolution)
    y_2 = 0
    for i in range(ite):
        x_1 = 0
        x_2 = int(resolution)
        y_1 = int(y_1 + resolution)
        y_2 = int(y_2 + resolution)
        for j in range(ite):
            b = int(color_Matrix[i][j][0])
            g = int(color_Matrix[i][j][1])
            r = int(color_Matrix[i][j][2])
            temp = cv.rectangle(temp, (x_1, y_1), (x_2, y_2), (b, g, r), -1)
            x_1 = int(x_1 + resolution)
            x_2 = int(x_2 + resolution)


def Tegn_Firkanter_Special(ite, temp, resolution, color_Matrix):
    y_1 = int(-resolution)
    y_2 = 0
    for i in range(ite):
        x_1 = 0
        x_2 = int(resolution)
        y_1 = int(y_1 + resolution)
        y_2 = int(y_2 + resolution)
        for j in range(ite):
            b = int(color_Matrix[i][j][0])
            g = int(color_Matrix[i][j][1])
            r = int(color_Matrix[i][j][2])
            temp = cv.rectangle(temp, (x_1, y_1), (x_2, y_2), (b, g, r), -1)
            x_1 = int(x_1 + resolution)
            x_2 = int(x_2 + resolution)


def Tegn_Firkanter_Special2(ite, temp, resolution, color_Matrix, farver):
    y_1 = int(-resolution)
    y_2 = 0
    for i in range(5):
        x_1 = 0
        x_2 = int(resolution)
        y_1 = int(y_1 + resolution)
        y_2 = int(y_2 + resolution)
        for j in range(5):
                if color_Matrix[i][j][3] == 1:
                    temp = cv.rectangle(temp, (x_1, y_1), (x_2, y_2), (21,162,111), -1)
                    x_1 = int(x_1 + resolution)
                    x_2 = int(x_2 + resolution)
                if color_Matrix[i][j][3] == 2:
                    temp = cv.rectangle(temp, (x_1, y_1), (x_2, y_2), (7,155,171), -1)
                    x_1 = int(x_1 + resolution)
                    x_2 = int(x_2 + resolution)
                if color_Matrix[i][j][3] == 3:
                    temp = cv.rectangle(temp, (x_1, y_1), (x_2, y_2), (37,52,38), -1)
                    x_1 = int(x_1 + resolution)
                    x_2 = int(x_2 + resolution)
                if color_Matrix[i][j][3] == 4:
                    temp = cv.rectangle(temp, (x_1, y_1), (x_2, y_2), (192,99,14), -1)
                    x_1 = int(x_1 + resolution)
                    x_2 = int(x_2 + resolution)
                if color_Matrix[i][j][3] == 5:
                    temp = cv.rectangle(temp, (x_1, y_1), (x_2, y_2), (43,95,108), -1)
                    x_1 = int(x_1 + resolution)
                    x_2 = int(x_2 + resolution)
                if color_Matrix[i][j][3] == 6:
                    temp = cv.rectangle(temp, (x_1, y_1), (x_2, y_2), (0, 0, 0), -1)
                    x_1 = int(x_1 + resolution)
                    x_2 = int(x_2 + resolution)
                if color_Matrix[i][j][3] == 0:
                    temp = cv.rectangle(temp, (x_1, y_1), (x_2, y_2), (255,255,255), -1)
                    x_1 = int(x_1 + resolution)
                    x_2 = int(x_2 + resolution)





#Gul - Mark (7,155,171)   - [ 93 245 171]
#grøn - eng (21,162,111)   - [ 79 222 162]
#mørkegrøn - skov  (37,52,38)   - [62 74 52]
#blå - vand  (192,99,14)   - [ 14 236 192]
#brun - vissen  (43,95,108)   - [ 96 153 108]
#sort - mine (0, 0, 0)    - [ 96 153 108]
#Anden brun Bordplade (19, 92, 124)   - [ 99 216 124]
Eng    = np.array([[42, 193, 148],[44, 205, 152],[41, 210, 141],[40, 197, 149],[40, 220, 138],[42, 214, 149],[37, 183, 128],[40, 219, 134],[40, 189, 119]])
Mark   = np.array([[26, 233, 189],[26, 233, 193],[26, 243, 193],[26, 233, 189],[26, 241, 186],[26, 236, 174],[27, 216, 168],[27, 243, 185],[26, 240, 186]])
Skov   = np.array([[42, 176, 68],[40, 166, 63],[40, 175, 64],[42, 166, 63],[40, 174, 63],[38, 162, 63],[40, 111, 71],[36, 181, 62],[34, 153, 70]])
Vand   = np.array([[104, 206, 129],[105, 233, 138],[106, 219, 129],[107, 236, 137],[106, 240, 153],[106, 236, 144],[105, 242, 160],[104, 233, 161],[105, 191, 140]])
Vissen = np.array([[23, 124, 111],[23, 130, 114],[23, 160, 113],[26, 250, 165],[23, 130,  94],[21, 156,  93],[22, 131, 121],[23, 158, 121],[25, 118, 115]])
Mine   = np.array([[22, 130,  51],[23, 125,  51],[23, 118,  56],[24, 134,  63],[20, 115,  40],[24, 118,  52],[23, 190,  55],[23, 157,  39],[22, 147,  40]])

np.reshape(Eng,    (9,3))
np.reshape(Mark,   (9,3))
np.reshape(Skov,   (9,3))
np.reshape(Vand,   (9,3))
np.reshape(Vissen, (9,3))
np.reshape(Mine,   (9,3))

Eng_Tresh    = np.array([[np.min(Eng, axis=0)],    [np.max(Eng, axis=0)]])
Mark_Tresh   = np.array([[np.min(Mark, axis=0)],   [np.max(Mark, axis=0)]])
Skov_Tresh   = np.array([[np.min(Skov, axis=0)],   [np.max(Skov, axis=0)]])
Vand_Tresh   = np.array([[np.min(Vand, axis=0)],   [np.max(Vand, axis=0)]])
Vissen_Tresh = np.array([[np.min(Vissen, axis=0)], [np.max(Vissen, axis=0)]])
Mine_Tresh   = np.array([[np.min(Mine, axis=0)],   [np.max(Mine, axis=0)]])



EH, EH1, ES, ES1, EV, EV1 = 2, 5, 5, 5, 5, 10
MH, MH1, MS, MS1, MV, MV1 = 10, 10, 15, 10, 20, 10
SH, SH1, SS, SS1, SV, SV1 = 5, 10, 20, 25, 30, 40
VH, VH1, VS, VS1, VV, VV1 = 5, 5, 5, 15, 30, 20
VIH, VIH1, VIS, VIS1, VIV, VIV1 = 1, 0, 25, 0, 18, 0
MIH, MIH1, MIS, MIS1, MIV, MIV1 = 3, 2, 5, 5, 3, 5

#[[ 20 115  39]]
#[[ 24 190  63]]
#25 155  63
#20 112  40
#23 190  55
#

def Type_Finder(input, output):
    print("Eng", Vand_Tresh)
    for i in range(5):
        for j in range(5):
            output[i][j][0] = input[i][j][0]
            output[i][j][1] = input[i][j][1]
            output[i][j][2] = input[i][j][2]
            if (Eng_Tresh[0][0][0]-EH < input[i][j][0] < Eng_Tresh[1][0][0]+EH1 and
            Eng_Tresh[0][0][1]-ES < input[i][j][1] < Eng_Tresh[1][0][1]+ES1 and
            Eng_Tresh[0][0][2]-EV < input[i][j][2] < Eng_Tresh[1][0][2]+EV1):
                output[i][j][3] = 1

            elif (Mark_Tresh[0][0][0]-MH <input[i][j][0]<Mark_Tresh[1][0][0]+MH1 and
            Mark_Tresh[0][0][1]-MS<input[i][j][1]<Mark_Tresh[1][0][1]+MS1 and 
            Mark_Tresh[0][0][2]-MV<input[i][j][2]<Mark_Tresh[1][0][2]+MV1):
                output[i][j][3] = 2

            elif (Skov_Tresh[0][0][0]-SH<input[i][j][0]<Skov_Tresh[1][0][0]+SH1 and 
            Skov_Tresh[0][0][1]-SS<input[i][j][1]<Skov_Tresh[1][0][1]+SS1 and 
            Skov_Tresh[0][0][2]-SV<input[i][j][2]<Skov_Tresh[1][0][2]+SV1):
                output[i][j][3] = 3

            elif (Vand_Tresh[0][0][0]-VH<input[i][j][0]<Vand_Tresh[1][0][0]+VH1 and
            Vand_Tresh[0][0][1]-VS<input[i][j][1]<Vand_Tresh[1][0][1]+VS1 and
            Vand_Tresh[0][0][2]-VV<input[i][j][2]<Vand_Tresh[1][0][2]+VV1):
                output[i][j][3] = 4

            elif (Vissen_Tresh[0][0][0]-VIH<input[i][j][0]<Vissen_Tresh[1][0][0]+VIH1 and
            Vissen_Tresh[0][0][1]-VIS<input[i][j][1]<Vissen_Tresh[1][0][1]+VIS1 and
            Vissen_Tresh[0][0][2]-VIV<input[i][j][2]<Vissen_Tresh[1][0][2]+VIV1):
                output[i][j][3] = 5

            elif (Mine_Tresh[0][0][0]-MIH<input[i][j][0]<Mine_Tresh[1][0][0]+MIH1 and
            Mine_Tresh[0][0][1]-MIS<input[i][j][1]<Mine_Tresh[1][0][1]+MIS1 and
            Mine_Tresh[0][0][2]-MIV<input[i][j][2]<Mine_Tresh[1][0][2]+MIV1):
                output[i][j][3] = 6

            else:
                output[i][j][3] = 0


def Viewer(input, input2, input3, input4):
    cv.imshow("Test1", input)
    cv.imshow("Test2", input2)
    cv.imshow("Test3", input3)
    cv.imshow("Test4", input4)
    cv.waitKey()


#Vars som kan ændres fra run til run
image_Count = 74
image_Size = 500
sub_Image_Size = 25
resolution_In_Drawn_Squares = int((500**2)/(sub_Image_Size**2))
size = int(math.sqrt(resolution_In_Drawn_Squares))
library_Of_Images = {}
iterations = int(image_Size/sub_Image_Size) #Skal forhåbentligt slettes
color_Level = 3
sub_Image_Matrix = np.zeros((size, size, sub_Image_Size, sub_Image_Size, color_Level), dtype='uint8')
color_Array = np.zeros((size, size, color_Level), dtype='uint8')
color_Array_HSV = np.zeros((size, size, color_Level), dtype='uint8')
template = np.zeros((image_Size, image_Size, color_Level), dtype='uint8')
template2 = np.zeros((image_Size, image_Size, color_Level), dtype='uint8')
template3 = np.zeros((image_Size, image_Size, color_Level), dtype='uint8')
sub_Image_Size2 = 100
resolution_In_Drawn_Squares2 = int((500**2)/(sub_Image_Size2**2))
size2 = int(math.sqrt(resolution_In_Drawn_Squares2))
print(size2)
color_Array2 = np.zeros((size2, size2, color_Level), dtype='uint8')
sub_Image_Matrix2 = np.zeros((size2, size2, sub_Image_Size2, sub_Image_Size2, color_Level), dtype='uint8')
TileArray = np.zeros((size2, size2, color_Level+1), dtype='uint8')
color_List = [[42, 193, 148],[26, 233, 189],[42, 176, 68],[104, 206, 129],[23, 124, 111],[22, 130,  51]]


Dictionary_generator(image_Count, path, library_Of_Images)
for i in range(1, image_Count+1):
    Gem_Alle_Billeder(library_Of_Images[f'image{i}'], size, sub_Image_Matrix, sub_Image_Size)
    Liste_Med_Underbilleders_Farver_Special(sub_Image_Matrix, size, color_Array)
    #color_Array2 = ColorA_To_ColorA2(color_Array)    
    print(sub_Image_Matrix.shape)   

    Tegn_Firkanter(size, template, sub_Image_Size, color_Array)
    Tegn_Firkanter_Special(size, template2, sub_Image_Size2, color_Array)

    Gem_Alle_Billeder(template, size2, sub_Image_Matrix2, sub_Image_Size2)
    Liste_Med_Underbilleders_Farver(sub_Image_Matrix2, size2, color_Array, color_Array2)
    Tegn_Firkanter_Special(size, template2, sub_Image_Size2, color_Array)
    print(color_Array[0][0], color_Array2[0][0])
    Type_Finder(color_Array2, TileArray)
    print(TileArray)
    print(Mine_Tresh[0])
    print(Mine_Tresh[1])
    Tegn_Firkanter_Special2(size, template3, sub_Image_Size2, TileArray, color_List)

    #lav_en_pixels_billede_HSV((42, 193, 148))
    Viewer(template, template2, template3, library_Of_Images[f'image{i}'])

#Viewer(template2)
















