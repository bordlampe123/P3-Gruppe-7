import cv2 as cv
import numpy as np
import math

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
#Input er det originale billeder der bliver procceceret
# Iterations er udregnet på baggrund af resolution of er det tal man får ved 500/resolution for at se hvor mange sub image man får på hvert led
#Resolution er størrelsen på underbilledet 
#Output matrixen er så en matrix med størrelsen (iteration, iteration, Resolution, Resolution, 3)
#nærmere beskrevet iteration gange iteration billeder med resolution gange resolution pixels som alle har 3 farveværdier i BGR
def Gem_Alle_Billeder(input, iterations, Resolution):
    Output_Matrix = np.zeros((iterations, iterations, Resolution, Resolution, 3))
    for i in range(iterations):
        for j in range(iterations):
            Output_Matrix[i][j] = gembillede(input, i, j, Resolution)
    return Output_Matrix
    




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

#Denne funktion tager matricen med underbillederne og finder den gennemsnitsfarven af billedet og giver det så som output i HSV og BGR
def Liste_Med_Underbilleders_Farver(input_sub_image_Matrix, ite):
    output_matrix_BGR = np.zeros((20, 20, 3))
    output_matrix_HSV = np.zeros((5, 5, 3))
    for i in range(ite):
        for j in range(ite):
            output_matrix_BGR[i][j] = prominent(input_sub_image_Matrix[i][j])
            output_matrix_HSV[i][j] = cv.cvtColor(np.uint8([[output_matrix_BGR[i][j]]]), cv.COLOR_BGR2HSV)[0][0]
    return output_matrix_BGR, output_matrix_HSV


#Her bestemmer
def Liste_Med_Underbilleders_Farver_Special(input_sub_image_Matrix, ite):
    output_matrix = np.zeros((ite,ite,3))
    for i in range(ite):
        for j in range(ite):
            if i%4 == False or (i+1)%4 == False:
                output_matrix[i][j] = prominent(input_sub_image_Matrix[i][j])
            else:
                if j%4 == False or (j+1)%4 == False:#1,3,4,7,8
                    output_matrix[i][j] = prominent(input_sub_image_Matrix[i][j]) 
                elif (j-1)%4 == False:#1,5,9
                    output_matrix[i][j] = prominent(input_sub_image_Matrix[i][j-1])
                elif (j+2)%4 == False:#2,6,10
                    output_matrix[i][j] = prominent(input_sub_image_Matrix[i][j+1])
                #debugging
                #else: output_matrix[i][j] = [0,0,0]
    return output_matrix



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



Eng_Tresh    = np.zeros((1,2,3))
Mark_Tresh   = np.zeros((1,2,3))
Skov_Tresh   = np.zeros((1,2,3))
Vand_Tresh   = np.zeros((1,2,3))
Vissen_Tresh = np.zeros((1,2,3))
Mine_Tresh   = np.zeros((1,2,3))

Eng_Tresh    = np.array([[[32, 160, 85]],    [[50, 255, 163]]])
Mark_Tresh   = np.array([[[23, 210, 130]],    [[33, 254, 204]]])
Skov_Tresh   = np.array([[[28, 105, 39]],    [[52, 221, 85]]])
Vand_Tresh   = np.array([[[31, 105, 56]],    [[108, 255, 194]]])
Vissen_Tresh = np.array([[[18, 56, 67]],    [[32, 255, 176]]])
Mine_Tresh   = np.array([[[17, 48, 33]],    [[31, 196, 74]]])

EH, EH1, ES, ES1, EV, EV1 = 0,0,0,0,0,0
MH, MH1, MS, MS1, MV, MV1 = 0,0,0,0,0,0
SH, SH1, SS, SS1, SV, SV1 = 0,0,0,0,0,0
VH, VH1, VS, VS1, VV, VV1 = 0,0,0,0,0,0
VIH, VIH1, VIS, VIS1, VIV, VIV1 = 0,0,0,0,0,0
MIH, MIH1, MIS, MIS1, MIV, MIV1 = 0,0,0,0,0,0

def Type_Finder(input):
    output = np.zeros((5, 5, 4))
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
    return output


def Tile_Assert(input):
    output = np.zeros((5,5,1))
    for i in range(5):
        for j in range(5):
            output[i][j] = input[i][j][3]
    
    return output





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
color_Array2 = np.zeros((size2, size2, color_Level), dtype='uint8')
sub_Image_Matrix2 = np.zeros((size2, size2, sub_Image_Size2, sub_Image_Size2, color_Level), dtype='uint8')
TileArray = np.zeros((size2, size2, color_Level+1), dtype='uint8')
color_List = [[42, 193, 148],[26, 233, 189],[42, 176, 68],[104, 206, 129],[23, 124, 111],[22, 130,  51]]


Dictionary_generator(image_Count, path, library_Of_Images)

for i in range(1, image_Count+1):
    sub_Image_Matrix = Gem_Alle_Billeder(library_Of_Images[f'image{i}'], size, sub_Image_Size)
    color_Array = Liste_Med_Underbilleders_Farver_Special(sub_Image_Matrix, size)
    Tegn_Firkanter(size, template, sub_Image_Size, color_Array)
    Tegn_Firkanter(size, template2, sub_Image_Size2, color_Array)
    sub_Image_Matrix2 = Gem_Alle_Billeder(template, size2, sub_Image_Size2)
    color_Array, color_Array2 = Liste_Med_Underbilleders_Farver(sub_Image_Matrix2, size2)
    
    Tegn_Firkanter(size, template2, sub_Image_Size2, color_Array)
    TileArray = Type_Finder(color_Array2)
    Tegn_Firkanter_Special2(size, template3, sub_Image_Size2, TileArray, color_List)
    
    TileArray_Kun_Type = Tile_Assert(TileArray)
    print(TileArray_Kun_Type)

    Viewer(template, template2, template3, library_Of_Images[f'image{i}'])

    #Miniks kode

    img_gray = cv.cvtColor(library_Of_Images[f'image{i}'], cv.COLOR_BGR2GRAY)

    #defining the template
    template = cv.imread("Vores_MiniProjekt\krone.png", cv.IMREAD_COLOR)
    template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    img_HSV = cv.cvtColor(template, cv.COLOR_BGR2HSV)

    w, h = template_gray.shape[::-1]

    #perform match operations
    res = cv.matchTemplate(library_Of_Images[f'image{i}'], template, cv.TM_CCOEFF)

    #specify a threshold
    threshold = 0.1

    #loc = np.where(res >= threshold)
    #for pt in zip(*loc[::-1]):
    # cv.rectangle(input, pt, (pt[0]+w, pt[1]+h), (0,0,255), 2)

    #cv.imwrite('res.png',input)

    output = res.copy()

    for y in range(res.shape[0]):
        for x in range(res.shape[1]):
            if res[y,x] < threshold:
                output[y,x] = 0
            else:
                output[y,x] = 255

    #show the final image with the matched area
    cv.imshow('input',library_Of_Images[f'image{i}'])
    cv.imshow('img_HSV',img_HSV)
    cv.imshow('img_gray',img_gray)
    cv.imshow('res',res)
    cv.imshow('template',template)
    cv.imshow('output',output)
    cv.waitKey(0)   
    cv.destroyAllWindows()
























