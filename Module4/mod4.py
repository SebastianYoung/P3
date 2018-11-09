import cv2, time, numpy as np

init = False
run_start = 0
run_end = 0
det_time = 0
total_time = 0

def Module4(img, posture, guess):
    global run_start, run_end, init, det_time, total_time

    key = cv2.waitKey(1)
    if key == 107 and not init:
        init = True
        run_start = time.time()
        print("\n########################\n### Module 4 - DEBUG ###\n########################\n|")
        print("| Started: {}".format(run_start))
    if key == 108 and init:
        init = False
        run_end = time.time()
        print("| Ended: {}\n| Runtime: {}\n|".format(run_end, run_end - run_start))

    if (init):
        total_time += 1
        if (posture==guess):
            det_time += 1
        # Efficiency comparison REVISE
        if (posture==guess and total_time + det_time <= 4):
            print("| Efficiency: {}".format(total_time+det_time))

    if not init and det_time > 0:
        _fps = np.divide(total_time, run_end - run_start)
        print("| REAL FPS:{}".format(_fps))
        print("| Detected difference: {} (0 = 100%) || Accuracy: {}%\n|".format(total_time - det_time, np.divide(det_time, total_time)*100))
        print("########################\n")
        det_time = 0
