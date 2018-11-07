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