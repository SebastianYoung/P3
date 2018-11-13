#Module 3
import cv2
import numpy as np
from matplotlib import pyplot as plt

handCords = []

def unitVector(vector):
    return vector / np.linalg.norm(vector)

def angle(uv1, uv2):
    return np.arccos(np.clip(np.dot(uv1, uv2), -1.0, 1.0))

def Module3(cords):
    # Required array of finger-coords
    # [ x_thumb, y_thumb,
    #   x_index, y_index,
    #   x_middle, y_middle,
    #   x_ring, y_ring
    #   x_pinky, y_pinky
    #   x_center, y_center ]
    if (cords.all() == None):
        print("The Leap Motion did not return anything, it may be offline")
        return
    global handCords
    handCords = cords

    thumb = np.array([cords[0], cords[1]])
    index = np.array([cords[2], cords[3]])
    middle = np.array([cords[4], cords[5]])
    ring = np.array([cords[6], cords[7]])

    thumbUnit = unitVector(thumb)
    indexUnit = unitVector(index)
    middleUnit = unitVector(middle)
    ringUnit = unitVector(ring)

    tiAngle = np.degrees(angle(thumbUnit, indexUnit))
    imAngle = np.degrees(angle(indexUnit, middleUnit))
    mrAngle = np.degrees(angle(middleUnit, ringUnit))

    print("TI:{:.2f}, IM:{:.2f}, MR:{:.2f}".format(tiAngle, imAngle, mrAngle))
    if (tiAngle >= 20 && mrAngle >= 14 && imAngle <= 20): # SCISSOR
        return 2 # RPS.SCISSORS = 2
    '''
    indexMag = np.linalg.norm(index)
    middleMag = np.linalg.norm(middle)
    ringMag = np.linalg.norm(ring)

    imAngle = (index[0]*middle[0])+(index[1]*middle[1])/(indexMag*middleMag)
    mrAngle = (ring[0]*middle[0])+(ring[1]*middle[1])/(middleMag*ringMag
    #imAngle = np.arccos(imAngle
    #mrAngle = np.arccos(mrAngle)

    print(index, indexMag)
    print(middle, middleMag)
    print(ring, ringMag)
    print(imAngle, mrAngle)
    print(np.dot(index, middle))
    print((indexMag*middleMag))
    print(np.dot(middle, ring))
    print((middleMag*ringMag))
    '''
