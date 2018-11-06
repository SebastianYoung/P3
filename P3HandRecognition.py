import cv2
import numpy as np
import sys
# Import RPS module
sys.path.insert(0, 'interactive-software/')
import RPS

cap = cv2.VideoCapture(0)

go = False
start = True
actualColour = [0, 0, 0]
actualShadowColour = [0, 0, 0]
# actualFingerColour = [0, 0, 0]


minlen = 18
maxlen = 28


while True:

    _, frame = cap.read()

    # Flips the Image for Gab
    frame = cv2.flip(frame, 1)

    # Converts the video capture to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

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

    ### YCBCR ###
    # Draws the rectangle we use for calibration in the Original Frame
#    cv2.rectangle(frame, (270, 190), (370, 290), [0, 0, 255], 1)

#    cv2.rectangle(frame, (270,100), (290, 130), [0, 0, 255], 1)

    # Creates an array with the axis of the HSV capture
#    calibrateMask = np.zeros(hls.shape[:2], np.uint8)

#    fingerMask = np.zeros(hls.shape[:2], np.uint8)

    # Creates 255 bins for the Array within a 100 x 100 area starting at x =  190, y = 270
#    calibrateMask[190: 290, 270: 370] = 255

#    fingerMask[100: 130, 270: 290] = 255



    # Creates an src with the mask over the HSV capture
#    caliMasked_hls = cv2.bitwise_and(hls, hls, mask=calibrateMask)

#    fingerMasked_hls = cv2.bitwise_and(hls, hls, mask= fingerMask)

    # Waits for input
#    key = cv2.waitKey(3) & 0xFF

    # Specifies that the key input should be 'c'
#    if key == ord('c'):

        # Creates a Struct (Structure) for color which are 'b', 'g', 'r'
#        colour = ('b', 'g', 'r')
#        fingerColour = ('b', 'g', 'r')

        # Creates an empty array name actualColour which will later be used to store 3 values.
#        actualColour = []
#        actualFingerColour = []

        # For loop which enumerates (Allows for loops over something with an automatic counter)
#        for each, col in enumerate(colour):

            # Creates a variable which holds the histogram of the area in calibrateMask between the values of 0 and 256
#            histr = cv2.calcHist([caliMasked_hls], [each], calibrateMask, [256], [0, 256])

            # For testing and adjusting values
#            print("The maximum value for " + str(col) + " is: ")
#            print(histr.max())
#            print("The minimum value for " + str(col) + " is: ")
#            print(histr.min())

            # Checks each value in the histogram, if the value it finds is the same as the maximum value of the
            # histogram, then append that value to the array actualColour.
#            for i in range(256):
#                if histr[i] == histr.max():
                    # print("This is the highest value for " + str(col) + str(i))
#                    actualColour.append(i)

            # For testing and adjusting values in the calibration
#            print(actualColour)
#            print(hls[240, 320])

#        for each, col in enumerate(fingerColour):

            # Creates a variable which holds the histogram of the area in calibrateMask between the values of 0 and 256
#            fingerhistr = cv2.calcHist([fingerMasked_hls], [each], fingerMask, [256], [0, 256])

            # For testing and adjusting values
#            print("The maximum value for " + str(col) + " is: ")
#            print(fingerhistr.max())
#            print("The minimum value for " + str(col) + " is: ")
#            print(fingerhistr.min())

            # Checks each value in the histogram, if the value it finds is the same as the maximum value of the
            # histogram, then append that value to the array actualColour.
#            for i in range(256):
#                if fingerhistr[i] == fingerhistr.max():
                    # print("This is the highest value for " + str(col) + str(i))
#                    actualFingerColour.append(i)

    ### HSV ###

#    # Draws the rectangle we use for calibration in the Original Frame
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

            # For testing and adjusting values
            print("The maximum value for " + str(col) + " is: ")
            print(histr.max())


            # Checks each value in the histogram, if the value it finds is the same as the maximum value of the
            # histogram, then append that value to the array actualColour.
            for i in range(256):
                if histr[i] == histr.max():
                    # print("This is the highest value for " + str(col) + str(i))
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

            # For testing and adjusting values
            print("The maximum value for " + str(col) + " is: ")
            print(histr.max())
            print("The minimum value for " + str(col) + " is: ")
            print(histr.min())

            # Checks each value in the histogram, if the value it finds is the same as the maximum value of the
            # histogram, then append that value to the array actualColour.
            for i in range(256):
                if histr[i] == histr.max():
                    # print("This is the highest value for " + str(col) + str(i))
                    actualShadowColour.append(i)

            # For testing and adjusting values in the calibration
            print(actualShadowColour)
            print("This is the ycbcr thingy: " + str(ycbcr[240, 320]))

########################################################################################################################
#                                                                                                                      #
#                                                    Isolating                                                         #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    # Old Colour ranges values for Young's webcam on Yellow. Don't Delete
    # lowerTargetedColour = np.array([4, 200, 96])
    # upperTargetColour = np.array([25, 255, 220])
    # mask = cv2.inRange(hsv, lowerTargetedColour, upperTargetColour)

    ### HLS ###
    # The current colour ranges used for calibration
    lowerCalibratedColour = np.array([actualColour[0] - 10, actualColour[1] - 20, actualColour[2] - 30])
    upperCalibratedColour = np.array([actualColour[0] + 10, actualColour[1] + 20, actualColour[2] + 30])

    lowerCalibratedShadowColour = np.array([actualShadowColour[0] - 10, actualShadowColour[1] - 20, actualShadowColour[2] - 30])
    upperCalibratedShadowColour = np.array([actualShadowColour[0] + 10, actualShadowColour[1] + 20, actualShadowColour[2] + 30])

    ### HSV ###
    # The current colour ranges used for calibration
#    lowerCalibratedColour = np.array([actualColour[0]-10, actualColour[1]-100, actualColour[2]-40])
#    upperCalibratedColour = np.array([actualColour[0]+10, actualColour[1]+100, actualColour[2]+40])

    # Creates a mask (an Array) within the range of the lower colour ranges (lowerCalibratedColour) and the upper colour
    # ranges (upperCalibratedColour) for HSV
    mask = cv2.inRange(hsv, lowerCalibratedColour, upperCalibratedColour)
    mask2 = cv2.inRange(hsv, lowerCalibratedShadowColour, upperCalibratedShadowColour)

    # Bitwise conjunction with the mask
    res = cv2.bitwise_and(frame, frame, mask = mask)
    res2 = cv2.bitwise_and(frame, frame, mask = mask2)

    # Does nothing at the moment but the 2nd argument here is where we can put the callibrated shadows
    finalRes = cv2.bitwise_or(mask, mask2, mask=None)

    # Converts our res to Gray from BGR
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # Thresholds the image an inverses it so the only thing left are our hand
    thresh, res_thresh = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY_INV)
    thresh2, res_thresh2 = cv2.threshold(res2, 0, 255, cv2.THRESH_BINARY_INV)

    finalres = cv2.bitwise_or(res, res2, mask=None)

    thresh3, finalThresh = cv2.threshold(finalres, 0, 255, cv2.THRESH_BINARY_INV)


    edges = cv2.Canny(finalres, 50, 200)




########################################################################################################################
#                                                                                                                      #
#                                                   Blurring                                                           #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    # This adaptThreshold creates an outline similar to how Edge Detection would
    # This is used to find the contours in the users hand
    res_adaptThresh = cv2.adaptiveThreshold(finalRes, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 2)

    # Creates 2 variables h & w which each corresponds to the shape of the above adaptThresh's x and y axis
    h, w, = res_adaptThresh.shape[:2]

    # Creates an array to create an area which should be blurred out
    blurMask = np.zeros((h + 2, w + 2), np.uint8)

    # Various blurring methods
    res_thresh = cv2.blur(res_thresh, (13, 13))
    res_thresh = cv2.medianBlur(res_thresh, 19)
    res_thresh = cv2.GaussianBlur(res_thresh, (5, 5), 0)
    res_thresh = cv2.bitwise_or(res_thresh, res)

    # Fills out the holes in the blurred image
    cv2.floodFill(res_thresh, blurMask, (0, 0), 255)

    rat = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 2)


    testKey = cv2.waitKey(3) & 0xFF

    # Specifies that the key input should be 'c'
    if testKey == ord('+'):
        minlen = minlen + 1
        print("minlen: " + str(minlen))

    if testKey == ord('-'):
        minlen = minlen - 1
        print("minlen: " + str(minlen))

    if testKey == ord('/'):
        maxlen = maxlen + 1
        print("maxlen: " + str(maxlen))
    if testKey == ord('*'):
        maxlen = maxlen - 1
        print("maxlen: " + str(maxlen))
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
    _, cnt, _ = cv2.findContours(rat, 2, 1)

    #
    if (len(cnt) > 0):
        maxArea = -1
        handFound = False
        for i in range(len(cnt)):
            temp = cnt[i]
            area = cv2.contourArea(temp)
            tmh = cv2.convexHull(temp, returnPoints=False)
            tmv = cv2.convexityDefects(temp, tmh)

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
            cv2.circle(frame, (cX, cY), 3, [255, 255, 0], -1)
            bounding_rect = cv2.boundingRect(cnt[i])

            rect = frame(bounding_rect).clone()
            cv2.imshow("Bounding Box", rect)

            #for x in range(len(hullMask[0])):
            #    for y in range(len(hullMask[1])):
            #        print("cnt is equal to: " + str(cnt[0][0][0]))
                    # hullMask[x][0] = cnt[0][0][0]
                    # hullMask[0][y] = cnt[0][0][1]
                    # print("HullMask is equal to: " + str(hullMask))
                    # hullMask[x,y] = [255, 255, 255]

            if (i) > 15:
                cv2.line(frame, (cX, cY), start, [255,255,255], 1)


            go = True
    except Exception as e:
        print(sys.exc_info(), sys.exc_info()[2].tb_lineno)

########################################################################################################################
#                                                                                                                      #
#                                                 Showing Windows                                                      #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

    # RPS module
    RPS.DrawGuess(frame, cap, RPS.RPS.ROCK, True) # Change RPS.RPS.ROCK later with the detected hand posture
    RPS.IS(frame)
    # ----------

    cv2.imshow("Original Frame", frame)
    cv2.imshow("HSV", hsv)
    cv2.imshow("res", res)
    cv2.imshow("Res2", res2)
    cv2.imshow("ResAdaptThresh", res_adaptThresh)
    cv2.imshow("Edges", edges)
    cv2.imshow("FinalRes", finalRes)

    #cv2.imshow("HLS", hls)
    #cv2.imshow("Both?", hls - hsv)
    # cv2.imshow("FingerMask", mask2)
    # cv2.imshow("Darker Mask", darkerMask)
    cv2.imshow("Grey", gray)
    cv2.imshow("finalThresh", finalThresh)
    # cv2.imshow("HullMask", hullMask)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
