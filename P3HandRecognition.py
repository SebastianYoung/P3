import cv2
import numpy as np
import sys
# Import RPS module
import math
sys.path.insert(0, 'interactive-software/')
import intsoft.RPS as RPS

cap = cv2.VideoCapture(0)

go = False
start = True
actualColour = [0, 0, 0]
actualShadowColour = [0, 0, 0]

minlen = 32
maxlen = 62


while True:

    _, frame = cap.read()

    # Flips the Image for Gab
    frame = cv2.flip(frame, 1)
    copyFrame = frame.copy()

    # Converts the video capture to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if start:
        hullMask = np.zeros_like(frame)
        start = False
########################################################################################################################
#                                                                                                                      #
#                                                  Calibration                                                         #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    ### HSV ###

    # Draws the rectangle we use for calibration in the Original Frame
    cv2.rectangle(frame, (270, 190), (370, 290), [0, 0, 255], 1)

    # Creates an array with the axis of the HSV capture
    calibrateMask = np.zeros(hsv.shape[:2], np.uint8)

    # Creates 255 bins for the Array within a 100 x 100 area starting at x =  190, y = 270
    calibrateMask[190: 290, 270: 370] = 255

    # Creates an src with the mask over the HSV capture
    caliMasked_hsv = cv2.bitwise_and(hsv, hsv, mask = calibrateMask)

    # Waits for input
    key = cv2.waitKey(3) & 0xFF

    # Specifies that the key input should be 'c'
    if key == ord('c'):

        # Creates a Struct (Structure) for color which are 'b', 'g', 'r'
        colour = ('b', 'g', 'r')

        # Creates an empty array name actualColour which will later be used to store 3 values.
        actualColour = []

        # For loop which enumerates (Allows for loops over something with an automatic counter)
        for each, col in enumerate(colour):

            # Creates a variable which holds the histogram of the area in calibrateMask between the values of 0 and 256
            histr = cv2.calcHist([caliMasked_hsv], [each], calibrateMask, [256], [0, 256])

            # Checks each value in the histogram, if the value it finds is the same as the maximum value of the
            # histogram, then append that value to the array actualColour.
            for i in range(256):
                if histr[i] == histr.max():
                    actualColour.append(i)

            # For testing and adjusting values in the calibration
            print(actualColour)
            print(hsv[240, 320])

    key = cv2.waitKey(5) & 0xFF

    # Specifies that the key input should be 'c'
    if key == ord('s'):

        # Creates a Struct (Structure) for color which are 'b', 'g', 'r'
        shadowColour = ('b', 'g', 'r')

        # Creates an empty array name actualColour which will later be used to store 3 values.
        actualShadowColour = []

        # For loop which enumerates (Allows for loops over something with an automatic counter)
        for each, col in enumerate(shadowColour):

            # Creates a variable which holds the histogram of the area in calibrateMask between the values of 0 and 256
            histr = cv2.calcHist([caliMasked_hsv], [each], calibrateMask, [256], [0, 256])

            # Checks each value in the histogram, if the value it finds is the same as the maximum value of the
            # histogram, then append that value to the array actualColour.
            for i in range(256):
                if histr[i] == histr.max():
                    actualShadowColour.append(i)


########################################################################################################################
#                                                                                                                      #
#                                                    Isolating                                                         #
#                                                   ### HSV ###                                                        #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    # The colour array used for the 'c' calibration
    lowerCalibratedColour = np.array([actualColour[0] - 10, actualColour[1] - 40, actualColour[2] - 40])
    upperCalibratedColour = np.array([actualColour[0] + 10, actualColour[1] + 40, actualColour[2] + 40])

    # The colour array used for the 's' calibration
    lowerCalibratedShadowColour = np.array([actualShadowColour[0] - 10, actualShadowColour[1] - 40, actualShadowColour[2] - 40])
    upperCalibratedShadowColour = np.array([actualShadowColour[0] + 10, actualShadowColour[1] + 40, actualShadowColour[2] + 40])

    # Creates a mask (an Array) within the range of the lower colour ranges (lowerCalibratedColour) and the upper colour
    # ranges (upperCalibratedColour) for HSV
    mask = cv2.inRange(hsv, lowerCalibratedColour, upperCalibratedColour)
    mask2 = cv2.inRange(hsv, lowerCalibratedShadowColour, upperCalibratedShadowColour)

    # Merges the two masks we have, each mask being a single calibration array of the colour.
    finalRes = cv2.bitwise_or(mask, mask2, mask=None)

########################################################################################################################
#                                                                                                                      #
#                                                   Blurring                                                           #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    # This adaptThreshold creates an outline similar to how Edge Detection would
    # This is used to find the contours in the users hand
    res_adaptThresh = cv2.adaptiveThreshold(finalRes, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

########################################################################################################################
#                                                                                                                      #
#                            For changing the the size of the boundary box to be detected                              #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    testKey = cv2.waitKey(3) & 0xFF

    # Specifies that the key input should be 'c'
    if testKey == ord('+'):
        minlen = minlen + 1
        # print("minlen: " + str(minlen))

    if testKey == ord('-'):
        minlen = minlen - 1
        # print("minlen: " + str(minlen))

    if testKey == ord('/'):
        maxlen = maxlen + 1
        # print("maxlen: " + str(maxlen))
    if testKey == ord('*'):
        maxlen = maxlen - 1
        # print("maxlen: " + str(maxlen))

########################################################################################################################
#                                                                                                                      #
#                                                   Boundary                                                           #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    # The _ in the beginning indicate that we do not want to store the first output of this function.
    # The function uses an algorithm developed by by Satoshi Suzuki and Keiichi Abe in 1985.
    # We do not need to explain exactly how this algorithm works.
    _, cnt, _ = cv2.findContours(res_adaptThresh, 2,3)

    if (len(cnt) > 0):
        maxArea = -1
        handFound = False
        for i in range(len(cnt)):
            temp = cnt[i]
            area = cv2.contourArea(temp)
            tmh = cv2.convexHull(temp, returnPoints=False)

            if area > maxArea and not handFound:
                try:
                    maxArea = area
                    ci = i
                except:
                    print("err")
            if len(tmh) >= minlen and len(tmh) <= maxlen:
                handFound = True
                ci = i
        hull = cv2.convexHull(cnt[ci], returnPoints=False)
        defects = cv2.convexityDefects(cnt[ci], hull)

        cnt = cnt[ci]

    ####################################################################################################################
    #                                                      Boundary                                                    #
    #                                                      Defects                                                     #
    #                                                                                                                  #
    ####################################################################################################################

    try:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            if go:

                diff_s1 = np.linalg.norm((tmp_s[0] - start[0]))
                diff_s2 = np.linalg.norm((tmp_s[1] - start[1]))
                diff_e1 = np.linalg.norm((tmp_e[0] - end[0]))
                diff_e2 = np.linalg.norm((tmp_e[1] - end[1]))
                glob = diff_s1 + diff_s2 + diff_e1 + diff_e2
                dist = 3
                if glob <= dist:
                    continue
            tmp_s = start
            tmp_e = end

            ############################################################################################################
            #                                              Boundary                                                    #
            #                                              Drawing                                                     #
            #                                                                                                          #
            ############################################################################################################
            cv2.line(frame, start, end, [255, 0, 0], 1)

            cv2.line(frame, start, far, [0, 0, 255], 1)
            cv2.line(frame, end, far, [0, 0, 255], 1)
            cv2.circle(frame, far, 3, [0, 200, 0], -1)

            cv2.putText(frame, str(i), end, 1, 1, (0, 0, 255), 2)
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(frame, "Center", (cX - 25, cY - 10), 1, 1, (0, 0, 255), 2)

            centdis = math.sqrt((cX - far[0]) * (cX - far[0]) + ((cY - far[1]) * (cY - far[1])))
            # print(cX)
            # print("The Euclidian distance of: " + str(i) + " is: " + str(centdis))
            # print("The Ratio of the Euclidian distance is: {}".format(centdis/area))
            # averageX = (cX + start[0])/2
            # averageY = (cY + start[1])/2
            # feretX = hull[0] / math.pi
            # feretY = hull[1] / math.pi
            # feretXY = feretX + feretY

            # print(round(feretXY, 2))

            # cv2.circle(res_adaptThresh, (cX, cY),feretXY, [255, 255, 255], -1)

            if 0.001 <= (centdis/area) <= 0.0022:
                cv2.line(frame, (cX, cY), start, [255,255,255], 1)
                cv2.line(copyFrame, (cX, cY), start, [255, 255, 255], 1)

                cv2.putText(copyFrame, str(abs(centdis/area)), start, 1, 1, (255, 255, 255), 2)
                cv2.circle(copyFrame, start, 10, [0, 0, int(centdis) * 2], -1)


            go = True
    except Exception as e:
        pass
        #print(sys.exc_info(), sys.exc_info()[2].tb_lineno)

########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                                             Young Trying Weird Shit                                                  #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################


########################################################################################################################
#                                                                                                                      #
#                                                   # RPS module                                                       #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    RPS.DrawGuess(frame, cap, RPS.RPS.ROCK, True) # Change RPS.RPS.ROCK later with the detected hand posture
    RPS.IS(frame)

########################################################################################################################
#                                                                                                                      #
#                                                 Showing Windows                                                      #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################
    cv2.imshow("Original Frame", frame)
    cv2.imshow("ResAdaptThresh", res_adaptThresh)
    cv2.imshow("Frame Copy", copyFrame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break



