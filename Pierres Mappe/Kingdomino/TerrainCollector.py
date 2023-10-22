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

def Meanifier(A, B, C):

    meadfindB = []
    #meadfindG = []
    #meadfindR = []

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

def Thresh(A, B, z, t, c):
    for y in range(A.shape[0]):
        for x in range(A.shape[1]):
            if z <= A[y, x, 0] <= t:
                B[y, x, 0] = c
            else:
                B[y, x, 0] = 0

def PutHSV(A):

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

def PutCount(A):

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

def ignite(y, x, A, c, id):
    
    queue = []

    if A[y, x, 0] == c and id[y, x, 0] == 0:
        queue.append([y, x])
    
        while len(queue) > 0:

            temp = queue.pop()

            posy = temp[0]
            posx = temp[1]

            id[posy, posx, 0] = current_id

            try:
                if x+1 < A.shape[1] and A[posy, posx+1, 0] == c and id[posy, posx+1, 0] == 0:
                    queue.append([posy, posx+1])
            except:
                pass
            try:
                if y+1 < A.shape[0] and A[posy+1, posx, 0] == c and id[posy+1, posx, 0] == 0:
                    queue.append([posy+1, posx])
            except:
                pass

            try:
                if x > 0 and A[posy, posx-1, 0] == c and id[posy, posx-1, 0] == 0:
                    queue.append([posy, posx-1])
            except:
                pass

            try:
                if y > 0 and A[posy-1, posx, 0] == c and id[posy-1, posx, 0] == 0:
                    queue.append([posy-1, posx])
            except:
                pass
        
        increment()

def crowndetect(picture_path):
    #test ideer:
    #forsøg at sharpen images inden de bliver opdelt,
    #se på right template for at få image 46 til at virke
    #leg evt med tresholds

    # Read the main image
    img = cv2.imread(picture_path)

    #make new images to place the new color chanels into

    img_outB = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    img_outG = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    img_outR = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    img_out2 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    img_out3 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    img_out4 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    img_out5 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

    #a function that splits the original image into its 3 color chanels
    def Split(A, B, G, R):
        for y in range(A.shape[0]):
            for x in range(A.shape[1]):
                B[y, x] = A[y, x, 0]
                G[y, x] = A[y, x, 1]
                R[y, x] = A[y, x, 2]


    #the original image it split
    Split(img, img_outB, img_outG, img_outR)

    #the diferent color chanels are subtracted from one another in order to end up ith only the white colors form the original image

    img_out3 = cv2.subtract(img_outR, img_outB)

    img_out4 = cv2.subtract(img_outB, img_outR)

    img_out5 = cv2.subtract(img_outB, img_out3)

    img_rgb = cv2.subtract(img_out5, img_out4)

    #img_out6 = cv2.subtract(img_outR, img_outB)

    #img_rgb = cv2.subtract(img_out3, img_out4)


    #bugfix showing the created images only for debugging
    #cv2.imshow("Only Blue", img_outB)
    #cv2.imshow("Only Green", img_outG)
    #cv2.imshow("Only Red", img_outR)
    cv2.imshow("Original", img)
    #cv2.waitKey(0)



    Crown1 = "Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Crown templates/Crown_bw.png"
    #placing marks where template 1 was found
    # Read the template
    template1 = cv2.imread(Crown1, 0)

    # Store width and height of template in w and h
    w1, h1 = template1.shape[::-1]

    # Perform match operations.
    res = cv2.matchTemplate(img_rgb, template1, cv2.TM_CCOEFF_NORMED)

    # Specify a threshold
    threshold = 0.6

    # Store the coordinates of matched area in a numpy array
    loc1 = np.where(res >= threshold)


    Crown2 = "Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Crown templates/Crown facing left.png"
    # placing marks where template 2 was found
    # Read the template
    template2 = cv2.imread(Crown2, 0)

    # Store width and height of template in w and h
    w2, h2 = template2.shape[::-1]

    # Perform match operations.
    res = cv2.matchTemplate(img_rgb, template2, cv2.TM_CCOEFF_NORMED)

    # Store the coordinates of matched area in a numpy array
    loc2 = np.where(res >= threshold)



    Crown3 = "Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Crown templates/Crown_facing_right.png"
    # placing marks where template 3 was found
    # Read the template
    template3 = cv2.imread(Crown3, 0)

    # Store width and height of template in w and h
    w3, h3 = template3.shape[::-1]

    # Perform match operations.
    res = cv2.matchTemplate(img_rgb, template3, cv2.TM_CCOEFF_NORMED)

    # Store the coordinates of matched area in a numpy array
    loc3 = np.where(res >= threshold)




    Crown4 = "Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Crown templates/Crown_facing_down.png"
    # placing marks where template 4 was found
    # Read the template
    template4 = cv2.imread(Crown4, 0)

    # Store width and height of template in w and h
    w4, h4 = template4.shape[::-1]

    # Perform match operations.
    res = cv2.matchTemplate(img_rgb, template4, cv2.TM_CCOEFF_NORMED)


    # Store the coordinates of matched area in a numpy array
    loc4 = np.where(res >= threshold)


    # a nice function taht filters multiple detections of the same crown out so we only detect each one once
    def filter_close_points(points, threshold):
        result = []
        for i in range(len(points)):
            x1, y1 = points[i]
            keep = True
            for j in range(i + 1, len(points)):
                x2, y2 = points[j]
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if distance < threshold:
                    keep = False
                    break
            if keep:
                result.append((x1, y1))
        return result


    #here the crowns are filterd using the filter_close_points fuction that looks at the next point in the array and determines if it is the same crow
    result1 = filter_close_points(np.column_stack(loc1), 5)

    #print(result1)

    result2 = filter_close_points(np.column_stack(loc2), 5)

    #print(result2)

    result3 = filter_close_points(np.column_stack(loc3), 5)

    #print(result3)

    result4 = filter_close_points(np.column_stack(loc4), 5)
    print(result4)
    print(len(result3))
    print(result2)
    print(result1)
    print(result4[0][0], result4[0][1])

    if len(result4) != 0:
        array4 = np.array(result4) 
    if len(result4) != 0:
        array3 = np.array(result4) 
    if len(result4) != 0:
        array2 = np.array(result4) 
    if len(result4) != 0:
        array1 = np.array(result4) 


    # Convert to NumPy array
    coordinates_array = np.concatenate([array4, array3, array2, array1])

    print(coordinates_array)

    unique_array = np.unique(coordinates_array, axis=0)

    print(unique_array)


    #now boxes are drawn around the crowns
    #for template 1
    #Draw a rectangle around the matched region.
    for y, x in result1:
        cv2.rectangle(img, (x, y), (x + w1, y + h1), (255, 50, 255), 2)


    for y, x in result2:
        cv2.rectangle(img, (x, y), (x + w2, y + h2), (255, 50, 255), 2)

    for y, x in result3:
        cv2.rectangle(img, (x, y), (x + w3, y + h3), (255, 50, 255), 2)

    for y, x in result4:
        cv2.rectangle(img, (x, y), (x + w4, y + h4), (255, 50, 255), 2)


    # Show the final image with the matched area.
    cv2.imshow('Detected', img)

    return result1, result2, result3, result4

def process_picture(picture_path):

    global current_id
    current_id = 1

    img = cv2.imread(picture_path)

    crown_loc = crowndetect(picture_path)

    print(crown_loc)

    #print(crown_loc.shape)

    img_mean = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_mean_RG = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_mean_RGB = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_mean_BR = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_mean_G = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_mean_GBR = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    img_small = np.zeros((20, 20, 3), np.uint8)
    img_small_RG = np.zeros((20, 20, 3), np.uint8)
    img_small_RGB = np.zeros((20, 20, 3), np.uint8)
    img_small_BR = np.zeros((20, 20, 3), np.uint8)
    img_small_G = np.zeros((20, 20, 3), np.uint8)
    img_small_GBR = np.zeros((20, 20, 3), np.uint8)


    img_small_big = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_small_big_RG = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_small_big_RGB = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_small_big_BR = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_small_big_G = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img_small_big_GBR = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    id_matrix = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

    print(id_matrix)

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

    img = cv2.imread(picture_path)
    
    Split(img, img_outB, img_outG, img_outR)

    img_GsubB = cv2.subtract(img_outG, img_outB)
    img_GsubBsubR = cv2.subtract(img_GsubB, img_outR)
    img_GsubR = cv2.subtract(img_outG, img_outR)
    img_RsubG = cv2.subtract(img_outR, img_outG)
    img_RsubGsubB = cv2.subtract(img_RsubG, img_outB)
    img_BsubR = cv2.subtract(img_outB, img_outR)

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
    Meanifier(img_GsubBsubR, img_mean_GBR, img_small_GBR)


    MeanifierSmall(img_small, img_small_big)
    MeanifierSmall(img_small_RG, img_small_big_RG)
    MeanifierSmall(img_small_RGB, img_small_big_RGB)
    MeanifierSmall(img_small_BR, img_small_big_BR)
    MeanifierSmall(img_small_G, img_small_big_G)
    MeanifierSmall(img_small_GBR, img_small_big_GBR)
 

    Thresh(img_small_big_GBR, img_Meadows, 8, 60, 255)
    Thresh(img_small_big, img_Forest, 4, 22, 255)
    Thresh(img_small_big_RG, img_Waste, 5, 22, 255)
    Thresh(img_small_big_RGB, img_Field, 5, 36, 255)
    Thresh(img_small_big_BR, img_Ocean, 80, 255, 255)
    Thresh(img_small_big_G, img_Mines, 100, 189, 255)

    for y in range(img_Waste.shape[0]):
        for x in range(img_Waste.shape[1]):
            if img_Waste[y, x] == 255:
                map_simplified[y, x] = 129

    for y in range(img_Field.shape[0]):
        for x in range(img_Field.shape[1]):
            if img_Field[y, x] == 255:
                map_simplified[y, x] = 172 

    for y in range(img_Mines.shape[0]):
        for x in range(img_Mines.shape[1]):
            if img_Mines[y, x] == 255:
                map_simplified[y, x] = 255 

    for y in range(img_Meadows.shape[0]):
        for x in range(img_Meadows.shape[1]):
            if img_Meadows[y, x] == 255:
                map_simplified[y, x] = 43

    for y in range(img_Forest.shape[0]):
        for x in range(img_Forest.shape[1]):
            if img_Forest[y, x] == 255:
                map_simplified[y, x] = 86    

    for y in range(img_Ocean.shape[0]):
        for x in range(img_Ocean.shape[1]):
            if img_Ocean[y, x] == 255:
                map_simplified[y, x] = 215 

       

    """ 
    cv2.imshow("Blue", img_outB)
    cv2.imshow("Green", img_outG)
    cv2.imshow("Gray Inverted", img_gray_inverted)
    cv2.imshow("Gray I % Red ", img_graysubR)
    cv2.imshow("Gray I % Red % Blue ", img_graysubRsubB)
    cv2.imshow("Gray I % Red % Blue % Green ", img_graysubRsubBsubG) """

    """ cv2.imshow("Red", img_outR)
    cv2.imshow("Green % Blue", img_GsubB)
    cv2.imshow("Green % Red", img_GsubR) """
    cv2.imshow("Green % Blue % Red", img_GsubBsubR)
    """ cv2.imshow("Red % Green % Blue", img_RsubGsubB) """

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

    for y in range(map_simplified.shape[0]):
        for x in range(map_simplified.shape[1]):
            ignite(y, x, map_simplified, map_simplified[y, x, 0], id_matrix)
            #print(current_id)

    id1_count = 0
    id2_count = 0
    id3_count = 0
    id4_count = 0
    id5_count = 0
    id6_count = 0
    id7_count = 0
    id8_count = 0
    id9_count = 0
    id10_count = 0
    id11_count = 0
    id12_count = 0
    id13_count = 0

    for y in range(id_matrix.shape[0]):
        for x in range(id_matrix.shape[1]):
            if id_matrix[y, x, 0] == 1:
                id1_count += 1
            if id_matrix[y, x, 0] == 2:
                id2_count += 1
            if id_matrix[y, x, 0] == 3:
                id3_count += 1
            if id_matrix[y, x, 0] == 4:
                id4_count += 1
            if id_matrix[y, x, 0] == 5:
                id5_count += 1
            if id_matrix[y, x, 0] == 6:
                id6_count += 1
            if id_matrix[y, x, 0] == 7:
                id7_count += 1
            if id_matrix[y, x, 0] == 8:
                id8_count += 1
            if id_matrix[y, x, 0] == 9:
                id9_count += 1
            if id_matrix[y, x, 0] == 10:
                id10_count += 1
            if id_matrix[y, x, 0] == 11:
                id11_count += 1
            if id_matrix[y, x, 0] == 12:
                id12_count += 1
            if id_matrix[y, x, 0] == 13:
                id13_count += 1

    id1_count_tiles = id1_count/10000
    id2_count_tiles = id2_count/10000
    id3_count_tiles = id3_count/10000
    id4_count_tiles = id4_count/10000
    id5_count_tiles = id5_count/10000
    id6_count_tiles = id6_count/10000
    id7_count_tiles = id7_count/10000
    id8_count_tiles = id8_count/10000
    id9_count_tiles = id9_count/10000
    id10_count_tiles = id10_count/10000
    id11_count_tiles = id11_count/10000
    id12_count_tiles = id12_count/10000
    id13_count_tiles = id13_count/10000

    print(id1_count_tiles)
    print(id2_count_tiles)
    print(id3_count_tiles)
    print(id4_count_tiles)
    print(id5_count_tiles)
    print(id6_count_tiles)
    print(id7_count_tiles)
    print(id8_count_tiles)
    print(id9_count_tiles)
    print(id10_count_tiles)
    print(id11_count_tiles)
    print(id12_count_tiles)
    print(id13_count_tiles)

    PutHSV(img_small_big)
    cv2.imshow("Forest", img_small_big)
    PutHSV(img_small_big_GBR)
    cv2.imshow("Meadows", img_small_big_GBR)
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

    """ PutCount(id_matrix)
    cv2.imshow("ID", id_matrix) """

    PutCount(id_matrix)
    cv2.imshow("ID", id_matrix)
    

    cv2.waitKey(0)
    cv2.destroyAllWindows()

""" for picture_file in picture_files:
    picture_path = os.path.join(picture_folder, picture_file)

    # Process the current picture
    process_picture(picture_path)

    # Wait for a key press to move to the next picture
    key = cv2.waitKey(0)

    # If the key pressed is 'q', break the loop and close the window
    if key == ord('q'):
        break

# Close any open windows
cv2.destroyAllWindows() """

process_picture("Pierres Mappe/Kingdomino/Billeder/7.jpg")

