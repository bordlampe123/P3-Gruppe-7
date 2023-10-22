import cv2
import numpy as np
import statistics as st
import os

picture_folder = "Pierres Mappe/Kingdomino/Billeder"

picture_files = [f for f in os.listdir(picture_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

picture_files.sort()

def increment():
    global current_id
    current_id += 1



def Split(A, B, G, R):
    for y in range(A.shape[0]):
        for x in range(A.shape[1]):
            B[y, x] = A[y, x, 0]
            G[y, x] = A[y, x, 1]
            R[y, x] = A[y, x, 2]


def MeanFilterColor(A, B, z):

    kernel = np.ones((z*2+1, z*2+1), np.uint8)

    for y in range(z, img.shape[0]-z):
        starty = y-z    
        for x in range(z, img.shape[1]-z):
            startx = x-z

            sum = np.zeros(3)

            for yy in range(kernel.shape[0]):
                for xx in range(kernel.shape[1]):
                    sum = np.add(sum, A[starty+yy, startx+xx])

            sumnorm = np.divide(sum, kernel.shape[0]**2)        

            B[y, x] = sumnorm

def Meanifier(A, B, C):

    meadfindB = []
    meadfindG = []
    meadfindR = []

    for y in range(20):
        starty = y*25
        #print("y" + str(starty))
        for x in range(20):
            startx = x*25
            #print("x" + str(startx))
            for yy in range(25):
                for xx in range(25):
                    #print(starty+yy, startx+xx)
                    meadfindB.append(A[starty+yy, startx+xx])
                    #meadfindG.append(A[starty+yy, startx+xx, 1])
                    #meadfindR.append(A[starty+yy, startx+xx, 2])
                    #print(meadfind)
            medianB = st.mean(meadfindB)
            #medianG = st.mean(meadfindG)
            #medianR = st.mean(meadfindR)
            C[y, x] = medianB
            #C[y, x, 1] = medianG
            #C[y, x, 2] = medianR
            #print(median)
            for yy in range(25):
                for xx in range(25):
                    B[starty+yy, startx+xx] = medianB
                    #B[starty+yy, startx+xx, 1] = medianG
                    #B[starty+yy, startx+xx, 2] = medianR
            meadfindB = []
            #meadfindG = []
            #meadfindR = []

def OldMeanifier(A, B):

    meadfindB = [0]
    meadfindG = [0]
    meadfindR = [0]

    for y in range(5):
        starty = y*100
        #print("y" + str(starty))
        for x in range(5):
            startx = x*100
            #print("x" + str(startx))
            for yy in range(100):
                for xx in range(100):
                    #print(starty+yy, startx+xx)
                    meadfindB.append(A[starty+yy, startx+xx, 0])
                    meadfindG.append(A[starty+yy, startx+xx, 1])
                    meadfindR.append(A[starty+yy, startx+xx, 2])
                    #print(meadfind)
            medianB = st.mean(meadfindB)
            medianG = st.mean(meadfindG)
            medianR = st.mean(meadfindR)
            #print(median)
            for yy in range(100):
                for xx in range(100):
                    B[starty+yy, startx+xx, 0] = medianB
                    B[starty+yy, startx+xx, 1] = medianG
                    B[starty+yy, startx+xx, 2] = medianR
            meadfindB = [0]
            meadfindG = [0]
            meadfindR = [0]

def MeanifierSmall(A, B):

    meadfindB = []
    meadfindG = []
    meadfindR = []

    for y in range(5):
        starty = y*4
        #print("y" + str(starty))
        for x in range(5):
            startx = x*4

            #print("x" + str(startx))
            for yy in range(4):
                for xx in range(4):
                    if (starty+yy+1)%4 == 0 or (starty+yy)%4 == 0 or (startx+xx+1)%4 == 0 or (startx+xx)%4 == 0:
                        #print(starty+yy, startx+xx)
                        meadfindB.append(A[starty+yy, startx+xx, 0])
                        meadfindG.append(A[starty+yy, startx+xx, 1])
                        meadfindR.append(A[starty+yy, startx+xx, 2])
                        #print(meadfind)
            medianB = st.mean(meadfindB)
            medianG = st.mean(meadfindG)
            medianR = st.mean(meadfindR)

            #print(median)
            for yy in range(100):
                for xx in range(100):
                    B[starty*25+yy, startx*25+xx, 0] = medianB
                    B[starty*25+yy, startx*25+xx, 1] = medianG
                    B[starty*25+yy, startx*25+xx, 2] = medianR
            meadfindB = []
            meadfindG = []
            meadfindR = []


def MeanifierGray(A, B):

    meadfind = [0]

    for y in range(20):
        starty = y*25
        #print("y" + str(starty))
        for x in range(20):
            startx = x*25
            #print("x" + str(startx))
            for yy in range(25):
                for xx in range(25):
                    #print(starty+yy, startx+xx)
                    meadfind.append(A[starty+yy, startx+xx])
                    #print(meadfind)
            median = st.mean(meadfind)
            #print(median)
            for yy in range(25):
                for xx in range(25):
                    B[starty+yy, startx+xx] = median

            meadfind = [0]

def Thresh(A, B, z, t, c):
    for y in range(A.shape[0]):
        for x in range(A.shape[1]):
            if z <= A[y, x, 0] <= t:
                B[y, x, 0] = c
            else:
                B[y, x, 0] = 0

def PutHSV(A):

    hsv = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    
    fontScale              = 0.35
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2



    for y in range(5):
        for x in range(5):
            bottomLeftCornerOfText = (x*100,y*100+50)
            cv2.putText(A,str(A[y*100,x*100]), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

def ignite(y, x, A):

    hsv = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)
    
    queue = []

    if hsv[y, x, 0]-10 <= hsv[y, x, 0] <= hsv[y, x, 0]+10 and hsv[y, x, 2]-40 <= hsv[y, x, 2] <= hsv[y, x, 2]+40 and id[y, x, 0] == 0:
        queue.append([y, x])
    
        while len(queue) > 0:

            temp = queue.pop()

            posy = temp[0]
            posx = temp[1]

            id[posy, posx, 0] = current_id

            print(posy, posx)

            if hsv[posy-1, posx, 0]-10 <= hsv[posy-1, posx, 0] <= hsv[posy-1, posx, 0]+10 and hsv[posy-1, posx, 2]-40 <= hsv[posy-1, posx, 2] <= hsv[posy-1, posx, 2]+40 and id[posy-1, posx, 0] == 0:
                queue.append([posy-1, posx])
            
            if hsv[posy, posx-1, 0]-10 <= hsv[posy, posx-1, 0] <= hsv[posy, posx-1, 0]+10 and hsv[posy, posx-1, 2]-40 <= hsv[posy, posx-1, 2] <= hsv[posy, posx-1, 2]+40 and id[posy, posx-1, 0] == 0:
                queue.append([posy, posx-1])

            if hsv[posy+1, posx, 0]-10 <= hsv[posy+1, posx, 0] <= hsv[posy+1, posx, 0]+10 and hsv[posy+1, posx, 2]-40 <= hsv[posy+1, posx, 2] <= hsv[posy+1, posx, 2]+40 == 255 and id[posy+1, posx, 0] == 0:
                queue.append([posy+1, posx])

            if hsv[posy, posx+1, 0]-10 <= hsv[posy, posx+1, 0] <= hsv[posy, posx+1, 0]+10 and hsv[posy, posx+1, 2]-40 <= hsv[posy, posx+1, 2] <= hsv[posy, posx+1, 2]+40 == 255 and id[posy, posx+1, 0] == 0:
                queue.append([posy, posx+1])
        
        increment()

def process_picture(picture_path):

    img = cv2.imread(picture_path)

    img_blurred = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_mean = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_mean_RG = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_mean_RGB = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_mean_BR = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_mean_G = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
 
    img_old_mean = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    img_spice = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_small = np.zeros((20, 20, 3), np.uint8)
    img_small_RG = np.zeros((20, 20, 3), np.uint8)
    img_small_RGB = np.zeros((20, 20, 3), np.uint8)
    img_small_BR = np.zeros((20, 20, 3), np.uint8)
    img_small_G = np.zeros((20, 20, 3), np.uint8)

    img_small_big = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_small_big_RG = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_small_big_RGB = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_small_big_BR = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_small_big_G = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    id = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

    map_simplified = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

    img_outB = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    img_outG = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    img_outR = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

    img_Meadows = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    img_Forest = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    img_Waste = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    img_Field = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    img_Ocean = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    img_Mines = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

    current_id = 1

    # Your code to process the picture goes here
    img = cv2.imread(picture_path)
    
    Split(img, img_outB, img_outG, img_outR)

    img_GsubB = cv2.subtract(img_outG, img_outB)
    img_GsubBsubR = cv2.subtract(img_GsubB, img_outR)
    img_GsubR = cv2.subtract(img_outG, img_outR)
    img_RsubG = cv2.subtract(img_outR, img_outG)
    img_RsubGsubB = cv2.subtract(img_RsubG, img_outB)
    img_BsubR = cv2.subtract(img_outB, img_outR)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray_inverted = cv2.bitwise_not(img_gray)
    img_I_B = cv2.bitwise_not(img_outB)
    img_I_G = cv2.bitwise_not(img_outG)
    img_I_R = cv2.bitwise_not(img_outR)

    img_graysubR = cv2.subtract(img_gray_inverted, img_I_R)
    img_graysubRsubB = cv2.subtract(img_graysubR, img_I_B)
    img_graysubRsubBsubG = cv2.subtract(img_graysubRsubB, img_I_G)

    img_outGmod = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    for y in range(img_outGmod.shape[0]):
        for x in range(img_outGmod.shape[1]):
            img_outGmod[y, x] = img_outG[y, x, 0]

    alpha = 5 # Contrast control
    beta = 5

    adjusted = cv2.convertScaleAbs(img_outGmod, alpha=alpha, beta=beta)

    Meanifier(img_GsubR, img_mean, img_small)
    Meanifier(img_RsubG, img_mean_RG, img_small_RG)
    Meanifier(img_RsubGsubB, img_mean_RGB, img_small_RGB)
    Meanifier(img_BsubR, img_mean_BR, img_small_BR)
    Meanifier(adjusted, img_mean_G, img_small_G)

    MeanifierSmall(img_small, img_small_big)
    MeanifierSmall(img_small_RG, img_small_big_RG)
    MeanifierSmall(img_small_RGB, img_small_big_RGB)
    MeanifierSmall(img_small_BR, img_small_big_BR)
    MeanifierSmall(img_small_G, img_small_big_G)

    Thresh(img_small_big, img_Meadows, 25, 60, 255)
    Thresh(img_small_big, img_Forest, 5, 21, 255)
    Thresh(img_small_big_RG, img_Waste, 5, 19, 255)
    Thresh(img_small_big_RGB, img_Field, 5, 22, 255)
    Thresh(img_small_big_BR, img_Ocean, 80, 255, 255)
    Thresh(img_small_big_G, img_Mines, 100, 180, 255)

    for y in range(img_Waste.shape[0]):
        for x in range(img_Waste.shape[1]):
            if img_Waste[y, x] == 255:
                map_simplified[y, x] = 129

    for y in range(img_Meadows.shape[0]):
        for x in range(img_Meadows.shape[1]):
            if img_Meadows[y, x] == 255:
                map_simplified[y, x] = 43

    for y in range(img_Forest.shape[0]):
        for x in range(img_Forest.shape[1]):
            if img_Forest[y, x] == 255:
                map_simplified[y, x] = 86

    for y in range(img_Field.shape[0]):
        for x in range(img_Field.shape[1]):
            if img_Field[y, x] == 255:
                map_simplified[y, x] = 172     

    for y in range(img_Ocean.shape[0]):
        for x in range(img_Ocean.shape[1]):
            if img_Ocean[y, x] == 255:
                map_simplified[y, x] = 215 

    for y in range(img_Mines.shape[0]):
        for x in range(img_Mines.shape[1]):
            if img_Mines[y, x] == 255:
                map_simplified[y, x] = 255        

    """ 
    cv2.imshow("Blue", img_outB)
    cv2.imshow("Green", img_outG)
    cv2.imshow("Gray Inverted", img_gray_inverted)
    cv2.imshow("Gray I % Red ", img_graysubR)
    cv2.imshow("Gray I % Red % Blue ", img_graysubRsubB)
    cv2.imshow("Gray I % Red % Blue % Green ", img_graysubRsubBsubG) """

    """ cv2.imshow("Red", img_outR)
    cv2.imshow("Green % Blue", img_GsubB)
    cv2.imshow("Green % Red", img_GsubR)
    #cv2.imshow("Green % Blue % Red", img_GsubBsubR)
    cv2.imshow("Red % Green % Blue", img_RsubGsubB) """

    cv2.imshow("Forest?", img_Forest)
    cv2.imshow("Meadow?", img_Meadows)
    cv2.imshow("Waste?", img_Waste)
    cv2.imshow("Field?", img_Field)
    cv2.imshow("Ocean", img_Ocean)
    cv2.imshow("Mines", img_Mines)

    cv2.imshow("Map Simplified", map_simplified)

    """ cv2.imshow("Contrasted", adjusted)

    cv2.imshow("Red Minus Green", img_RsubG)
    cv2.imshow("Blue % Red", img_BsubR) """

    PutHSV(img_small_big)
    cv2.imshow("Forest and Meadow val", img_small_big)
    PutHSV(img_small_big_RG)
    cv2.imshow("Waste val", img_small_big_RG)
    PutHSV(img_small_big_G)
    cv2.imshow("Mines val", img_small_big_G)
    PutHSV(img_small_big_RGB)
    cv2.imshow("Field val", img_small_big_RGB)
    PutHSV(img_small_big_BR)
    cv2.imshow("Ocean val", img_small_big_BR)
    #PutHSV(img_old_mean)
    #cv2.imshow("Old Mean", img_old_mean)

    #cv2.imshow("Contrasted", adjusted)
    #cv2.imshow("Gray Mean", img_MeanGray)
    #cv2.imshow("Threshed M", img_Meadows)
    #cv2.imshow("Threshed F", img_Forest)


    #cv2.imshow("Gray Blurred", blurred)
    #cv2.imshow("Gray Mean Blurred", img_MeanGrayBlurred)

    cv2.imshow("mean", img_mean)

    cv2.imshow(picture_path, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

for picture_file in picture_files:
    picture_path = os.path.join(picture_folder, picture_file)

    # Process the current picture
    process_picture(picture_path)

    # Wait for a key press to move to the next picture
    key = cv2.waitKey(0)

    # If the key pressed is 'q', break the loop and close the window
    if key == ord('q'):
        break

# Close any open windows
cv2.destroyAllWindows()

