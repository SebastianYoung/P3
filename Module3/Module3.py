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
	if (cords.all() == None):
        print("The Leap Motion did not return anything, it may be offline")
        return
	global handCords
	handCords = cords

	thumb	= np.array([cords[0], cords[1]])
	index 	= np.array([cords[2], cords[3]])
	middle 	= np.array([cords[4], cords[5]])
	ring 	= np.array([cords[6], cords[7]])

	thumb 	-= thumb
	index 	-= thumb
	middle 	-= index
	ring 	-= middle
	
	thumbUnit	= unitVectir(thumb)
	indexUnit 	= unitVector(index)
	middleUnit 	= unitVector(middle)
	ringUnit 	= unitVector(ring)

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
	mrAngle = (ring[0]*middle[0])+(ring[1]*middle[1])/(middleMag*ringMag)
	#imAngle = np.arccos(imAngle)
	#mrAngle = np.arccos(mrAngle)
	'''

#Based on centre to fingertips
def module3Centre(cords):
	global handCords
	handCords = cords

	index 	= np.array([cords[2], cords[3]])
	middle 	= np.array([cords[4], cords[5]])
	ring 	= np.array([cords[6], cords[7]])
	centre 	= np.array([cords[10], cords[11]])

	index 	-= centre
	middle 	-= centre
	ring 	-= centre

	indexUnit 	= unitVector(index)
	middleUnit 	= unitVector(middle)
	ringUnit 	= unitVector(ring)

	imAngle = np.degrees(angle(indexUnit, middleUnit))
	mrAngle = np.degrees(angle(middleUnit, ringUnit))

	print(imAngle, mrAngle)