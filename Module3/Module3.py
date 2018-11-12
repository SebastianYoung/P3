#Module 3
import cv2
import numpy as np
from matplotlib import pyplot as plt

handCords = []

def unitVector(vector):
	return vector / np.linalg.norm(vector)

def angle(uv1, uv2):
    return np.arccos(np.clip(np.dot(uv1, uv2), -1.0, 1.0))

def module3(cords):
	global handCords
	handCords = cords

	index = np.array([cords[2], cords[3]])
	middle = np.array([cords[4], cords[5]])
	ring = np.array([cords[6], cords[7]])

	indexUnit = unitVector(index)
	middleUnit = unitVector(middle)
	ringUnit = unitVector(ring)

	imAngle = np.degrees(angle(indexUnit, middleUnit))
	mrAngle = np.degrees(angle(middleUnit, ringUnit))

	print(imAngle, mrAngle)
	'''
	indexMag = np.linalg.norm(index)
	middleMag = np.linalg.norm(middle)
	ringMag = np.linalg.norm(ring)

	imAngle = (index[0]*middle[0])+(index[1]*middle[1])/(indexMag*middleMag)
	mrAngle = (ring[0]*middle[0])+(ring[1]*middle[1])/(middleMag*ringMag)
	#imAngle = np.arccos(imAngle)
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
test = np.array([2, 8, 5, 5, 5, -5, -5, -9, 2, 3, 4, 5])
module3(test)