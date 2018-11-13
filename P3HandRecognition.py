import cv2
import numpy as np
import sys
import math
import intsoft.RPS as RPS
import Module1.Module1 as M1
import Module4.Module4 as M4
import Module3.Module3 as M3

cap = cv2.VideoCapture(1)

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
    copyFrame = frame

    # Converts the video capture to HSVg
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
    key = cv2.waitKey(1) & 0xFF

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

    key = cv2.waitKey(1) & 0xFF

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

    openKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    closeKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    
 #   finalRes = cv2.erode(finalRes, openKernel)

#    finalRes = cv2.dilate(finalRes, closeKernel)

    # finalRes = cv2.morphologyEx(finalRes, cv2.MORPH_CLOSE, openKernel)

    fingerRes = cv2.bitwise_or(mask, mask, mask = None)


########################################################################################################################
#                                                                                                                      #
#                                                   Blurring                                                           #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    # This adaptThreshold creates an outline similar to how Edge Detection would
    # This is used to find the contours in the users hand
    res_adaptThresh = cv2.adaptiveThreshold(finalRes, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 29, 2)

########################################################################################################################
#                                                                                                                      #
#                            For changing the the size of the boundary box to be detected                              #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################


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
                glob_S = diff_s1 + diff_s2
                print("Glob_S is equal to " + str(glob_S))
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




            # cv2.putText(frame, str(i), end, 1, 1, (0, 0, 255), 2)
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(frame, "Center", (cX - 25, cY - 10), 1, 1, (0, 0, 255), 2)

            centdis = math.sqrt((cX - far[0]) * (cX - far[0]) + ((cY - far[1]) * (cY - far[1])))

            for c in cnt:
                peri = (cv2.arcLength(c, True))
                approx = cv2.approxPolyDP(c, 0.01 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(copyFrame, (x, y), (w+x, h+y), (0, 255, 0), 1)

            if start[1] <= cY + 10 and glob_S > 70:
                for k in range(5):
                    cv2.line(frame, (cX, cY), start, [255, 255, 255], 1)
                    cv2.line(copyFrame, (cX, cY), start, [255, 255, 255], 1)
                    cv2.putText(copyFrame, str(i) + str(tmp_e), start, 1, 1, (255, 255, 255), 2)




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

    # Module 4 hook
    M4.Module4(frame, RPS.RPS.ROCK, RPS.RPS.ROCK, cap.get(cv2.CAP_PROP_FPS))
    M3.Module3(M1.leapMotion()) 
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

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break



