#Module 3
import cv2
import numpy as np
from matplotlib import pyplot as plt

handCords = []

def unitVector(vector):
	return vector / np.linalg.norm(vector)

def angle(uv1, uv2):
	return np.arccos(np.clip(np.dot(uv1, uv2), -1.0, 1.0))

#Based on fingertip to fingertip
def module3Tips(cords):
    print(cords)
    if cords.any() == None:
        return -1
    global handCords
    handCords = cords

    thumbCord	= np.array([cords[0], cords[1]])
    indexCord 	= np.array([cords[2], cords[3]])
    middleCord 	= np.array([cords[4], cords[5]])
    ringCord 	= np.array([cords[6], cords[7]])

    index 	= indexCord  - thumbCord
    middle 	= middleCord - indexCord
    ring 	= ringCord   - middleCord
	
    thumbUnit	= unitVector(thumbCord)
    indexUnit 	= unitVector(index)
    middleUnit 	= unitVector(middle)
    ringUnit 	= unitVector(ring)

    tiAngle = np.degrees(angle(thumbUnit, indexUnit))
    imAngle = np.degrees(angle(indexUnit, middleUnit))
    mrAngle = np.degrees(angle(middleUnit, ringUnit))

    print("TI:{:.2f}, IM:{:.2f}, MR:{:.2f}".format(tiAngle, imAngle, mrAngle))
    if (tiAngle <= 5 and tiAngle >= 0.5 and imAngle <= 3 and imAngle >= 0.3 and mrAngle <= 5 and mrAngle >= 0.9):
        return 0 # Rock
    if (tiAngle <= 11 and tiAngle >= 0 and
        imAngle <= 4 and imAngle >= 1 and
        mrAngle <= 6 and mrAngle >= 2):
        return 1 # Paper
    if (tiAngle <= 21 and tiAngle >= 4 and
        imAngle <= 15 and imAngle >= 7 and
        mrAngle <= 19 and mrAngle >= 4):
        return 2 # Scissor
	'''
	indexMag = np.linalg.norm(index)
	middleMag = np.linalg.norm(middle)
	ringMag = np.linalg.norm(ring)

	imAngle = (index[0]*middle[0])+(index[1]*middle[1])/(indexMag*middleMag)
	mrAngle = (ring[0]*middle[0])+(ring[1]*middle[1])/(middleMag*ringMag)
	#imAngle = np.arccos(imAngle)
	#mrAngle = np.arccos(mrAngle)
	'''

#Based on centre to fingertips
def module3Centre(cords):
    if cords.any() == None:
        return -1
    global handCords
    handCords = cords

    thumb   = np.array([cords[0], cords[1]])
    index 	= np.array([cords[2], cords[3]])
    middle 	= np.array([cords[4], cords[5]])
    ring 	= np.array([cords[6], cords[7]])
    centre 	= np.array([cords[10], cords[11]])

    thumb   = centre - thumb
    index 	= centre - index
    middle 	= centre - middle
    ring 	= centre - ring

    thumbUnit   = unitVector(thumb)
    indexUnit 	= unitVector(index)
    middleUnit 	= unitVector(middle)
    ringUnit 	= unitVector(ring)

    tiAngle = np.degrees(angle(thumbUnit, indexUnit))
    imAngle = np.degrees(angle(indexUnit, middleUnit))
    mrAngle = np.degrees(angle(middleUnit, ringUnit))

    print(imAngle, mrAngle, tiAngle)
    return 1