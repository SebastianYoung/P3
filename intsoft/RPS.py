import cv2, random, time
from enum import Enum

SCALE = 15;

RPS = Enum(
    "RPS",
    "ROCK PAPER SCISSORS"
)

# -- -- #

# Get random new posture from system to compare with software
def GetRndPosture():
    rnd = random.randint(0, 2);
    choices = {
        0: RPS.ROCK,
        1: RPS.PAPER,
        2: RPS.SCISSORS
    }
    return choices.get(rnd, 'default')

# Set initial global values
rps_rnd = GetRndPosture()
rps_t = time.time()

# Interactive software
def IS(im):
    if (time.time() - rps_t > 2):
        global rps_t, rnd
        rps_t = time.time()
        rps_rnd = GetRndPosture()
    global rps_rnd
    cv2.putText(im, "{}".format(rps_rnd), (SCALE+1, im.shape[0]-SCALE+1), 1, 1, (0,0,0)) #Text shadow
    cv2.putText(im, "{}".format(rps_rnd), (SCALE, im.shape[0]-SCALE), 1, 1, (255,255,255))

# Debugs & Draw the guess result
def DrawGuess(im, cap, guess, debug):
    if (debug):
        cv2.putText(im, "FPS: {}".format(cap.get(cv2.CAP_PROP_FPS)), (SCALE, SCALE), 1, 1, color=(0, 255, 0))
    global rps_rnd
    cv2.putText(im, "GUESS: {} -> {}".format(guess, guess==rps_rnd), (im.shape[1]/2, im.shape[0]-SCALE), 1, 1, color=(0, 255, 0))
