import Leap
import cv2
import numpy


imgSize = [500, 500]

controller = Leap.Controller()

def leapMotion():
	frame = controller.frame()
	hands = frame.hands
	if hands[0].is_valid:
		fingers = hands[0].fingers
		fingerTips = numpy.zeros(12)
		for finger in fingers:
			fingerTips[finger.type*2] = finger.bone(3).center[0]
			fingerTips[finger.type*2 + 1] = finger.bone(3).center[1]
		fingerTips[10] = hands[0].palm_position[0]
		fingerTips[11] = hands[0].palm_position[1]
		return fingerTips
	return numpy.array(None)







'''
while (1):
	image = numpy.zeros((imgSize[0], imgSize[1]), numpy.uint8)
	image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	fingerTips = leapMotion()
	
	if fingerTips.any() != None:
		for i in range(0,numpy.size(fingerTips), 2):
			colour = [(255*i)/numpy.size(fingerTips),255,(255*i)/numpy.size(fingerTips)]
			cv2.circle(image, (int(fingerTips[i]) + imgSize[0]/2, - int(fingerTips[i+1]) + imgSize[1]), 3, colour, -1)
	cv2.imshow("Data", image)
	
	k = cv2.waitKey(1)
	if k == 27:
		break
'''





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