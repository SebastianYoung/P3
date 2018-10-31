import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt



cap = cv2.VideoCapture(0)

go = False
running = True
actualColour = [0,0,0]


while (True):

    _, frame = cap.read()

    # Converts the video capture to HSV

    frame = cv2.flip(frame, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    ysv = hsv + ycr

    # cv2.circle(frame, (cX, cY), 3, [255, 255, 0], -1)
    cv2.rectangle(frame, (270, 190), (370, 290), [0, 0, 255], 1)



    calibrateMask = np.zeros(hsv.shape[:2], np.uint8)
    calibrateMask[190: 290, 270: 370] = 255
    caliMasked_hsv = cv2.bitwise_and(hsv, hsv, mask = calibrateMask)

    # hist_mask = cv2.calcHist([frame], [0], calibrateMask,[256], [0, 256])
    key = cv2.waitKey(3) & 0xFF

    if key == ord('c'):
        color = ('b', 'g', 'r')
        actualColour = []
        for each, col in enumerate(color):
            histr = cv2.calcHist([caliMasked_hsv], [each], calibrateMask, [256], [0, 256])
            # plt.plot(histr, color=col)
            print("The maximum value for " + str(col) + " is: ")
            print(histr.max())

            print("The minimum value for " + str(col) + " is: ")
            print(histr.min())

            for i in range(256):
                # print(str(i) + ":" + str(histr[i]))
                if histr[i] == histr.max():
                    # print("This is the highest value for " + str(col) + str(i))
                    actualColour.append(i)

            print(actualColour)
            print(hsv[240, 320])


    # plt.xlim([0, 256])

    # plt.show()

   #  key = cv2.waitKey(5) & 0xFF


    #if key == 255 and running:
    #    for x in range(270, 370):
    #        for y in range(190, 290):
    #            colour = hsv[x, y]
    #            print("I HAVE REACHED THIS CODE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #print("Key is equal to: " + str(key))
    #print("Colour is equal to: " + str(colour))

    # Target colour ranges
    lowerTargetedColour = np.array([4, 200, 96])
    upperTargetColour = np.array([25, 255, 220])
    lowerCallibratedColour = np.array([actualColour[0]-10, actualColour[1]-100, actualColour[2]-40])
    upperCallibratedColour = np.array([actualColour[0]+10, actualColour[1]+100, actualColour[2]+40])

    lowerTargetedDarkerColour = np.array([160, 15, 0])
    upperTargetedDarkerColour = np.array([200, 110, 43])

    # mask = cv2.inRange(hsv, lowerTargetedColour, upperTargetColour)
    darkerMask = cv2.inRange(hsv + ycr, lowerTargetedDarkerColour, upperTargetedDarkerColour)
    mask = cv2.inRange(hsv, lowerCallibratedColour, upperCallibratedColour)

    res = cv2.bitwise_and(frame, frame, mask = mask)
    finalRes = cv2.bitwise_or(mask, darkerMask, mask=None)

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    thresh, res_thresh = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY_INV)
    res_gray = cv2.bitwise_and(gray, gray, mask = None)
#######################################################################################################################

    # Blurring
#######################################################################################################################
    # Need to test exactly how useful this is:

    res_adaptThresh = cv2.adaptiveThreshold(finalRes, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 3)

    h, w, = res_adaptThresh.copy().shape[:2]
    blurMask = np.zeros((h + 2, w + 2), np.uint8)

    res_thresh = cv2.blur(res_thresh, (13, 13))
    res_thresh = cv2.medianBlur(res_thresh, 19)
    res_thresh = cv2.GaussianBlur(res_thresh, (5, 5), 0)
    res_thresh = cv2.bitwise_or(res_thresh, res)

    cv2.floodFill(res_thresh, blurMask, (0, 0), 255)

    edges = cv2.Canny(res_gray, 180, 200)

    #######################################################################################################################

    # Contours
######################################################################################################################
    im2, cont, hier = cv2.findContours(res_adaptThresh, 2, 1)
    _, contours, hierarchy = cv2.findContours(res_adaptThresh, 2, 1)
    cnt = contours

    hull = None
    defects = None
    ci = 0
    if (len(cnt) > 0):
        maxArea = -1
        handFound = 0
        for i in range(len(cnt)):
            temp = cnt[i]
            area = cv2.contourArea(temp)
            tmh = cv2.convexHull(temp, returnPoints=False)
            tmv = cv2.convexityDefects(temp, tmh)
            #print("temp is equal to: ")
            #print(temp[0][0])
            if area > maxArea and not handFound:
                try:
                    # print("Biggest choice {}, v {}".format(len(tmh), len(tmv)))
                    maxArea = area
                    ci = i
                except:
                    print("err")
            if len(tmh) >= 18 and len(tmh) <= 28:
                # print("Hand contours {}, v {}".format(len(tmh), len(tmv)))
                handFound = 1
                ci = i
        hull = cv2.convexHull(cnt[ci], returnPoints=False)
        hull2 = cv2.convexHull(cnt[ci], returnPoints=True)
        defects = cv2.convexityDefects(cnt[ci], hull)

        cnt = cnt[ci]


    try:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            if go:
                #diff_s1 = abs(tmp_s[0] - start[0])
                #diff_s2 = abs(tmp_s[1] - start[1])
                #diff_e1 = abs(tmp_e[0] - end[0])
                #diff_e2 = abs(tmp_e[1] - end[1])
                diff_s1 = np.linalg.norm((tmp_s[0] - start[0]))
                diff_s2 = np.linalg.norm((tmp_s[1] - start[1]))
                diff_e1 = np.linalg.norm((tmp_e[0] - end[0]))
                diff_e2 = np.linalg.norm((tmp_e[1] - end[1]))
                glob = diff_s1 + diff_s2 + diff_e1 + diff_e2
                # print("{} {} {} {}, {}".format(diff_s1, diff_s2, diff_e1, diff_e2, glob))
                dist = 3
                # if (diff_s1 <= dist) or (diff_s2 <= dist) or (diff_e1 <= dist) or (diff_e2 <= dist):
                if (glob <= dist):
                    print("Skipping")
                    continue
            # print("drawing")
            tmp_s = start
            tmp_e = end



            # Draw
            cv2.line(frame, start, end, [255, 0, 0], 1)
            cv2.line(ycr, start, end, [0, 0, 255], 1)

            handBound = cv2.inRange(ycr, np.array([0,0,255]), np.array([0,0,255]))
            cv2.line(frame, start, far, [0, 0, 255], 1)
            cv2.line(frame, end, far, [0, 0, 255], 1)
            cv2.circle(frame, far, 3, [0, 200, 0], -1)

            cv2.putText(frame, str(i), (end), 1, 1, (0, 0, 255), 2)
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(frame, "Center", (cX - 25, cY - 10), 1, 1, (0, 0, 255), 2)
            cv2.circle(frame, (cX, cY), 3, [255, 255, 0], -1)

            # print("Hand contours {}, v {}".format(len(tmh), len(tmv)))

            distRatioX = cX / start[0]
            distRatioY = cY / start[1]

            # distRatioX = np.linalg.norm((cX - start[0]))
            # distRatioY = np.linalg.norm((cY - start[1]))


            if (i) > 15:
                cv2.line(frame, (cX, cY), start, [255,255,255], 1)

            # print("cX / start[0] is equal to :" + str(distRatioX) + "for " + str(i))
            # print("cY / start[1] is equal to :"  + str(distRatioY) + "for " + str(i))


            go = True
    except Exception as e:
        print(sys.exc_info(), sys.exc_info()[2].tb_lineno)
########################################################################################################################


    cv2.imshow("Original Frame", frame)
    cv2.imshow("HSV", hsv)
    # cv2.imshow("HSVFULL", hsvfull)
    # cv2.imshow("Difference", difference)
    cv2.imshow("Mask", mask)
    # cv2.imshow("Res_Thresh", res_thresh)
    # cv2.imshow("Res", res)
    # cv2.imshow("Blurmask", blurMask)
    # cv2.imshow("GraySCale", res_gray)
    # cv2.imshow("ResAdaptThresh", res_adaptThresh)
    # cv2.imshow("YCRCB", ycr)
    # cv2.imshow("What?" , ysv)
    # cv2.imshow("Darker Mask", darkerMask)
    # cv2.imshow("Combined Res", finalRes)
    # cv2.imshow("HandBound", handBound)
    # cv2.imshow("Calimasked", calibrateMask)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break