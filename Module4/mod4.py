import cv2, time, numpy as np

init = False
run_start = 0
run_end = 0
det_time = 0
total_time = 0
eff = "0"

def Module4(img, posture, guess):
    global run_start, run_end, init, det_time, total_time, eff

    key = cv2.waitKey(1)
    if key == 107 and not init:
        init = True
        run_start = time.time()
        print("\n########################\n### Module 4 - DEBUG ###\n########################\n|")
        print("| Started: {}".format(run_start))
    if key == 108 and init:
        init = False
        run_end = time.time()
        print("| Ended: {0}\n| Runtime: {1:.2f}s\n|".format(run_end, run_end - run_start))

    if (init):
        total_time += 1
        if (posture==guess):
            det_time += 1
        # Efficiency comparison REVISE
        if (posture==guess and len(eff) <= 1):
            if (time.time() - run_start) == 0:
                eff = "| Efficiency: 0s"
            else:
                t_fps = np.divide(total_time, time.time() - run_start)
                eff = np.divide(total_time, t_fps)
                eff = "| Efficiency: {0:.2f}s".format(eff)

    if not init and total_time > 0:
        _fps = np.divide(total_time, run_end - run_start)
        print("| REAL FPS: {0:.2f}".format(_fps))
        if (total_time - det_time) == 0:
            print("| Accuracy: {}%".format(100))
        else:
            print("| Accuracy: {}%".format(total_time - det_time, np.divide(det_time, total_time)*100))

        print(eff)
        print("| |{}| |{}|".format(total_time, det_time))
        print("########################\n")
        det_time = 0
        total_time = 0
