import Leap
import time
import cv2
import numpy

imgSize = [500, 500]

controller = Leap.Controller()

time.sleep(1)
while (1):
	image = numpy.zeros((imgSize[0], imgSize[1]), numpy.uint8)
	image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	frame = controller.frame()
	hands = frame.hands
	for hand in hands:
		#print(str(hand) + "\nPalm position: " + str(hand.palm_position))
		#image[hand.palm_position[0] + 250][hand.palm_position[1] + 250] = 255
		cv2.circle(image, (int(hand.palm_position[0]) + imgSize[0]/2, - int(hand.palm_position[1]) + imgSize[1]), 3, [255, 255, 255], -1)
		fingers = hand.fingers
		for finger in fingers:
			if (finger.type == finger.TYPE_INDEX):
				fingerType = "Index"
			elif (finger.type == finger.TYPE_MIDDLE):
				fingerType = "Middle"
			elif (finger.type == finger.TYPE_RING):
				fingerType = "Ring"
			elif (finger.type == finger.TYPE_PINKY):
				fingerType = "Pinky"
			elif (finger.type == finger.TYPE_THUMB):
				fingerType = "Thumb"
			tip = finger.bone(3)
			tipCenter = tip.center
			tipDirection = tip.direction
			cv2.circle(image, (int(tipCenter[0]) + imgSize[0]/2, - int(tipCenter[1]) + imgSize[1]), 3, [0, 0, 255], -1)
			#print(fingerType + " finger:" + "\nCenter: " + str(tipCenter) + "\nDirection: " + str(tipDirection))
	cv2.imshow("Data", image)
	cv2.waitKey(1)

