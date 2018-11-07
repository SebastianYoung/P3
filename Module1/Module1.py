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