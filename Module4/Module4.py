import cv2, time, numpy as np

init_4 = False
run_start = 0
run_end = 0
det_time = 0
detl_time = 0
total_time = 0
eff = "0"
effl = "0"
hand_cons = 0
leap_cons = 0

last_handguess = None
last_leapguess = None
def Module4(img, posture, guess, leap_posture, leap_guess, cam_fb):
    global run_start, run_end, init_4, det_time, total_time, eff, effl, detl_time, hand_cons, leap_cons, last_leapguess, last_handguess
    #if not (leap_guess == None):
    #	print(leap_guess)
    key = cv2.waitKey(1)
    if key == 107 and not init_4:
        init_4 = True
        run_start = time.time()
        print("\n########################\n### Module 4 - DEBUG ###\n########################\n|")
        print("| Started: {}".format(run_start))
    if key == 108 and init_4:
        init_4 = False
        run_end = time.time()
        print("| Ended: {0}\n| Runtime: {1:.2f}s\n|".format(run_end, run_end - run_start))

    if (init_4):
        if (guess == last_handguess):
            hand_cons += 1
        last_handguess = guess
        if (leap_guess == last_leapguess):
            leap_cons += 1
        last_leapguess = leap_guess
        print("det frame : {}, det_leap_frame {}".format(det_time, detl_time))
        total_time += 1
        print("hand: {} {}".format(posture, guess))
        print("leap: {} {}".format(leap_posture, leap_guess))
        print(posture == guess)
        if (posture==guess):
            det_time += 1
        if (leap_posture==leap_guess):
            detl_time += 1
            #print("SCISSOR")
            if (len(effl) <= 1):
                if (time.time() - run_start) == 0:
                    effl = "| L_Efficiency: 0s"
                else:
                    t_fps = np.divide(total_time, time.time() - run_start)
                    effl = np.divide(total_time, t_fps)
                    effl = "| L_Efficiency: {:.2f}s".format(effl)

        if (posture==guess and len(eff) <= 1):
            if (time.time() - run_start) == 0:
                eff = "| Efficiency: 0s"
            else:
                t_fps = np.divide(total_time, time.time() - run_start)
                eff = np.divide(total_time, t_fps)
                eff = "| Efficiency: {0:.2f}s".format(eff)

    if not init_4 and total_time > 0:
        _fps = np.divide(total_time, run_end - run_start)
        print("| REAL FPS: {0:.2f}".format(_fps))
        print("| Accuracy: {}%".format(np.divide(float(det_time), float(total_time))*100))
        print("| L_Accuracy: {}%".format(np.divide(float(detl_time), float(total_time))*100))
        l_frames = cam_fb - _fps
        if (l_frames <= 0):
            l_frames = 0
        print("{}, lost frames: {:.2f}".format(eff, l_frames))
        print("| |{}| |{}| |{}|".format(total_time, det_time, detl_time))
        print("########################\n")
        print("leap_consistency: {}\nhand_consistency: {}\n".format(leap_cons, hand_cons))
        print("########################\n")
        det_time = 0
        detl_time = 0
        total_time = 0
        leap_cons = 0
        hand_cons = 0
