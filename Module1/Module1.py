import Leap
import time
import cv2
import numpy


imgSize = [500, 500]

controller = Leap.Controller()

def leapMotion():
	frame = controller.frame()
	hands = frame.hands
	handsAmount = numpy.size(hands)
	leftHand, rightHand = ["None"], ["None"]
	for hand in hands:
		fingers = hand.fingers
		fingerInfo = []
		for i in range(len(fingers)):
			fingerInfo.extend([fingers[i].type, fingers[i].bone(1).center.to_tuple(), fingers[i].bone(2).center.to_tuple(), fingers[i].bone(3).center.to_tuple()])
		if (hand.is_left):
			leftHand.extend(fingerInfo)
			leftHand[0] = "Left"
		else:
			rightHand.extend(fingerInfo)
			rightHand[0] = "Right"
	hands = [leftHand, rightHand]
	return handsAmount, hands


time.sleep(1)
while (1):
	image = numpy.zeros((imgSize[0], imgSize[1]), numpy.uint8)
	image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	_, hands = leapMotion()
	for hand in hands:
		if (hand[0] != "None"):
			if (hand[0] == "Right"):
				colour = [0, 0, 255]
			else:
				colour = [255, 0, 0]
			for i in range(2, 18, 4):
				for j in range(3):
					cv2.circle(image, (int(hand[i+j][0]) + imgSize[0]/2, - int(hand[i+j][1]) + imgSize[1]), 2 , colour, -1)
	cv2.imshow("Data", image)
	cv2.waitKey(1)

'''
  Depricated implementation
======= 
import sys

if sys.version_info[0] == 3:
	import os, inspect, _thread, time
else:
	import os, inspect, thread, time

src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))

arch_dir = '../lib/x64' if sys.maxsize > 2**32 else '../lib/x86'

sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))

import Leap

class ControllerListener(Leap.Listener):

	def on_connect(self, controller):
		print("Controller is connected")
		controller.remove_listener(listener)

def main():
	listener = ControllerListener()
	controller = Leap.Controller(listener)

	while True:
		if controller.is_connected:
			print("Doing things")
			frame = controller.frame()
			prevFrame = controller.frame(1)
>>>>>>> f4413e99c8638d1c3a84b3b4c251a5b7bf1cf0a7
'''