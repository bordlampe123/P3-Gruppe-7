import cv2
import numpy as np
import os

pic_count = 0

prev_img = []

imgprev = 0

fps = 30

picture_folder = "Pierres Mappe/Video Analysis/Data/Train001"

picture_files = [f for f in os.listdir(picture_folder) if f.endswith(('.tif'))]

picture_files.sort()

for idx, picture_file in enumerate(picture_files):
    
    print(picture_file)
    picture_path = os.path.join(picture_folder, picture_file)

    img = cv2.imread(picture_path, cv2.IMREAD_GRAYSCALE)

    if idx > 0:
        img_diff1 = cv2.subtract(img, prev_img[idx-1])
    
    if idx > 2:
        img_diff3 = cv2.subtract(img, prev_img[idx-3])

    if idx > 6:
        img_diff7 = cv2.subtract(img, prev_img[idx-7])

    """ if idx > 99:
        img_diff30 = cv2.subtract(img, prev_img[idx-100]) """


    prev_img.append(img)
    #print(idx)
    #print(len(prev_img))

    control = cv2.subtract(img, imgprev)

    imgprev = img

    if idx > 0:
        cv2.imshow("Diff-1", img_diff1)
    if idx > 0:
        cv2.imshow("Prev-1", prev_img[idx-1])
    
    if idx > 2:
        cv2.imshow("Diff-3", img_diff3)
    if idx > 2:
        cv2.imshow("Prev-3", prev_img[idx-3])
    
    if idx > 6:
        cv2.imshow("Diff-7", img_diff7)
    if idx > 6:
        cv2.imshow("Prev-7", prev_img[idx-7])

    """ if idx > 99:
        cv2.imshow("Diff-30", img_diff30)
    if idx > 99:
        cv2.imshow("Prev-30", prev_img[idx-100]) """


    cv2.imshow("Current", img)
    cv2.waitKey(80)
