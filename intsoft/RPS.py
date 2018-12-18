import cv2, random, time, numpy as np
from enum import Enum

SCALE = 15;
GUESS_TIME = 5

# -- -- #
time_correct = 0

# Get random new posture from system to compare with software
def GetRndPosture():
    rnd = random.randint(0, 2);
    global time_correct
    time_correct = 0
    return rnd

# Set initial global values
rps_rnd = GetRndPosture()
rps_t = time.time()

# Interactive software
def IS(im):
    global rps_t, rps_rnd
    if (time.time() - rps_t > GUESS_TIME):
        rps_t = time.time()
        rps_rnd = GetRndPosture()
    the_print = ""
    if (rps_rnd == 0):
        the_print = "Rock"
    if (rps_rnd == 1):
        the_print = "Paper"
    if (rps_rnd == 2):
        the_print = "Scissor"
    cv2.putText(im, "{}".format(the_print), (SCALE+1, im.shape[0]-SCALE+1), 1, 1, (0,0,0)) #Text shadow
    cv2.putText(im, "{}".format(the_print), (SCALE, im.shape[0]-SCALE), 1, 1, (255,255,255))

def retRandom():
    global rps_rnd
    return rps_rnd
# Debugs & Draw the guess result
def DrawGuess(im, cap, guess, debug):
    global time_correct, rps_rnd
    FPS = cap.get(cv2.CAP_PROP_FPS)
    if (debug):
        cv2.putText(im, "FPS: {}".format(FPS), (SCALE, SCALE), 1, 1, color=(0, 255, 0))
    if (guess==rps_rnd):
        time_correct += 1
        if (time_correct > FPS):
            time_correct = FPS
    elif(time_correct > 0):
        time_correct -= 1

    correct = np.divide(time_correct, float(FPS))
    cv2.putText(im, "GUESS: {} -> {} | Acc: {}%".format(guess, guess==rps_rnd, float(correct)*100), (int(im.shape[1]/2-SCALE), int(im.shape[0]-SCALE)), 1, 1, color=(0, 255, 0))
