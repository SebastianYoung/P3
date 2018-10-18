import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)


running = True

go = False
while(True):

    # Take each frame
    _, frame = cap.read()

# Convert BGR to HSV

#    ret, faceFrame = cap.read()
#
 #   if ret:
  #      gray = cv2.cvtColor(faceFrame, cv2.COLOR_BGR2GRAY)
#
 #       faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#
 #   for (x, y, w, h) in faces:
  #      cv2.rectangle(faceFrame, (x, y), (x + w, y + h), (0, 0, 0), -1)
   #     roi_gray = gray[y:y + h, x:x + w]
    #    roi_color = faceFrame[y:y + h, x:x + w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex, ey, ew, eh) in eyes:
        #   cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
    # Capture frame-by-frame

    # Our operations on the frame come here
#    gray = cv2.cvtColor(faceFrame, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gra = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    framo = frame.copy();

    # frame2 = cv2.equalizeHist(gra)
    # define range of blue color in HSV

    rgb = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)

    r = cv2.equalizeHist(cv2.split(rgb)[0])
    g = cv2.equalizeHist(cv2.split(rgb)[1])
    b = cv2.equalizeHist(cv2.split(rgb)[2])
    hsv_fix = cv2.merge((r,g,b))
    hsv2 = cv2.cvtColor(hsv_fix, cv2.COLOR_BGR2HSV)
    ycr = cv2.cvtColor(hsv, cv2.COLOR_BGR2YCR_CB)

    if running:
        colour = hsv[330,240]
        running = False

    # print(colour)
    # print("Red is equal to : " + str(colour[0]))
    # print("Green is equal to : " + str(colour[1]))
    # print("Blue is equal to : " + str(colour[2]))
    skinColorLower = np.array([colour[0]-100, colour[1]-100, colour[2]-100])  # This green value here was originally 56
    skinColorUpper = np.array([colour[0]+50, colour[1]+50, colour[2]+50])

    skinColorLowerB = np.array([colour[0]-60, colour[1]-80, colour[2]-70])
    skinColorUpperB = np.array([colour[0]+35, colour[1]+35, colour[2]+75])

    skinColorLowerC = np.array([144, 9, 59])
    skinColorUpperC = np.array([179, 117, 212])


    skinColorLowerE = np.array([4, 200, 96])
    skinColorUpperE = np.array([25, 255, 220])

    skinColorLowerF = np.array([15, 175, 150])
    skinColorUpperF = np.array([25, 220, 255])


    skinColorLowerG = np.array([10, 220, 50])
    skinColorUpperG = np.array([20, 255, 100])


    skinColorLowerPurple = np.array([135, 17, 95])
    skinColorUpperPurple = np.array([220, 110, 255])

    skinColorLowerBlue = np.array([120, 20, 0])
    skinColorUpperBlue = np.array([205, 105, 85])


    skinColorLowerD = np.array([0 ,85, 150])
    skinColorUpperD = np.array([15 ,120, 220])

    # skinColorLowerC = np.array([54, 131, 110])
    # skinColorUpperC = np.array([163, 157, 135])

    # Threshold the HSV image to get only blue colors

    # mask = cv2.inRange(hsv, skinColorLowerYCRCB, skinColorUpperYCRCB)
    mask = cv2.inRange(hsv, skinColorLowerE, skinColorUpperE)
    maskOrange = cv2.inRange(hsv, skinColorLowerF, skinColorUpperF)
    maskGreen = cv2.inRange(hsv, skinColorLowerBlue, skinColorUpperBlue)
    mask2 = cv2.inRange(hsv, skinColorLowerD, skinColorUpperD)
    maskgray = cv2.inRange(hsv2, skinColorLowerC, skinColorUpperC)
    #  mask2 = cv2.inRange()

    # Bitwise-AND mask and original image
    res3 = cv2.bitwise_and(frame,frame, mask= mask)
    resOrange = cv2.bitwise_and(frame, frame, mask = maskOrange)
    resBlue = cv2.bitwise_and(frame, frame, mask = maskGreen)
    res2 = cv2.bitwise_and(frame,frame, mask = mask2)
    # res = cv2.bitwise_or(res3, res2, mask= None)

    #  res2 = cv2.bitwise_and(res,res,mask = mask)

    # cv2.imshow('Equalized', frame2)
    # cv2.imshow('Gray Scale Nigga', gra)
    # cv2.imshow('frame',frame)




    th, res_th = cv2.threshold(res3 , 0, 255, cv2.THRESH_BINARY_INV)
    th2, res_th2 = cv2.threshold(resOrange, 0, 255, cv2.THRESH_BINARY_INV)
    th4, res_th4 = cv2.threshold(resBlue, 0, 255, cv2.THRESH_BINARY_INV)

    finalRes = cv2.bitwise_or(resOrange, res3, mask = None)
    finalres = cv2.bitwise_or(finalRes, resBlue)
    th3, res_th3 = cv2.threshold(finalRes, 0, 255, cv2.THRESH_BINARY_INV)

    # th4, res_th4 = cv2.bitwise_or(res_th, res_th2, cv2.THRESH_BINARY_INV)


    h, w = res_th.copy().shape[:2]
    mask3 = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(res_th3, mask3, (0, 0), 255)
    res_th3 = cv2.blur(res_th3, (9,9))
    res_th3 = cv2.medianBlur(res_th3, 5)
    res_th3 = cv2.GaussianBlur(res_th3, (5, 5), 0)
    res_th3 = cv2.bitwise_and(res_th3, res_th, 1)
    # cv2.imshow('gr', res_th)

    # cv2.floodFill(res_th, mask3, (0, 0), 255)

    # res_floodfill_inv = cv2.bitwise_not(res_th, 1)

    # res_out = res_th | res_floodfill_inv

    # h, w = res_th.shape[:2]
    #h1, w1 = res_th2.shape[:2]
    # print res_th[x,y]
    # mask3 = np.zeros((h + 2, w + 2), np.uint8)
    #mask4 = np.zeros((h1 + 2, w1 +2), np.uint8)


    # res_th = cv2.medianBlur(res_th, 5)
    #res_th2 = cv2.medianBlur(res_th2, 59)

    # res_th = cv2.GaussianBlur(res_th, (3,3), 3)
    #res_th2 = cv2.GaussianBlur(res_th2, (3, 3), 3)

    #es_th = cv2.bilateralFilter(res_th, 9, 30,30)
    #res_th2 = cv2.bilateralFilter(res_th2, 9, 30, 30)


    # res_th = cv2.blur(res_th, (1,1), 5)
    #res_th2 = cv2.blur(res_th2, (1, 1), 5)

    # res_th = cv2.bitwise_and(res_th, res_th, 1)
    #res_th2 = cv2.bitwise_and(res_th2, res_th2, 1)

    # cv2.floodFill(res_th, mask3, (0, 0), 255)
    #cv2.floodFill(res_th2, mask4, (0, 0), 255)

    # res_floodfill_inv = cv2.bitwise_not(res_th, -1)
    #res_floodfill_inv2 = cv2.bitwise_not(res_th2, -1)

    #resCombined = cv2.bitwise_or(res_th, res_th2, -1)

    # res_out = res_th | res_floodfill_inv
    #res_out2 = res_th2 | res_floodfill_inv2

    # res = cv2.inRange(res, np.array([0,0,0]), np.array([255,255,255]))



    # th3, res_th3 = cv2.adaptiveThreshold(res_th3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)


    lol = ycr + hsv

    im2, cont, hier = cv2.findContours(mask, 2,1 )
    _, contours, hierarchy = cv2.findContours(mask, 2, 1)
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
            tmh = cv2.convexHull(temp, returnPoints = False)
            tmv = cv2.convexityDefects(temp, tmh)
            if area > maxArea and not handFound:
                try:
                    print("Biggest choice {}, v {}".format(len(tmh), len(tmv)))
                    maxArea = area
                    ci = i
                except:
                    print("err")
            if (len(tmh) >= 18 and len(tmh) <= 28):
                print("Hand contours {}, v {}".format(len(tmh), len(tmv)))
                handFound = 1
                ci = i
        hull = cv2.convexHull(cnt[ci],returnPoints = False)
        defects = cv2.convexityDefects(cnt[ci],hull)
        cnt = cnt[ci]

    try:
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            if go:
                diff_s1 = abs(tmp_s[0] - start[0])
                diff_s2 = abs(tmp_s[1] - start[1])
                diff_e1 = abs(tmp_e[0] - end[0])
                diff_e2 = abs(tmp_e[1] - end[1])
                glob = diff_s1 + diff_s2 + diff_e1 + diff_e2
                #print("{} {} {} {}, {}".format(diff_s1, diff_s2, diff_e1, diff_e2, glob))
                dist = 5
                #if (diff_s1 <= dist) or (diff_s2 <= dist) or (diff_e1 <= dist) or (diff_e2 <= dist):
                if (glob <= dist):
                    print("Skipping")
                    continue
            print("drawing")
            tmp_s = start
            tmp_e = end

        #Draw
            cv2.line(frame,start,end,[255,0,0],1)
            cv2.line(frame,start,far,[0,0,255],1)
            cv2.line(frame,end,far,[0,0,255],1)
            cv2.circle(frame,far,3,[0,200,0],-1)
            go = True
    except Exception as e:
        print(sys.exc_info(), sys.exc_info()[2].tb_lineno)


    # cv2.imshow('hsv',hsv)
    cv2.imshow('hsv2', hsv2)
    cv2.imshow('res1',framo)
    cv2.imshow('Frame', frame)
    cv2.imshow("Thresholded Image", res_th3)
    cv2.imshow("mask1", mask)
    cv2.imshow("HSV", hsv)
    cv2.imshow("FinalRes", finalRes)
    cv2.imshow("Res2" , res2)
    cv2.imshow("maskOrange", maskOrange)
    cv2.imshow("LOL", lol)

    # cv2.imshow("resBlurred", res_blurred)



    # plt.imshow(hsv2)
    # plt.show()
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break



cv2.destroyAllWindows()