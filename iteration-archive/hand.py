import cv2
import numpy as np
import sys

gray_lower = 55
gray_upper = 65

hsv_l = np.array([20, 21, 98])
hsv_u = np.array([104, 112, 255])

def Noth():
    return

#img = cv2.imread('hands.jpg')
cap = cv2.VideoCapture(0)

go = False
check = False
bg = None
while (1):
    ret, img = cap.read()
    
    #h = cv2.equalizeHist(cv2.split(img)[0])
    #s = cv2.equalizeHist(cv2.split(img)[1])
    #v = cv2.equalizeHist(cv2.split(img)[2])
    #img = cv2.merge((h,s,v))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(img[int(img.shape[0]/2),int(img.shape[1]/2)])
    #thresh = cv2.inRange(img, hsv_l, hsv_u)
    #cv2.imshow('bgr', mask)

    ret, thresh = cv2.threshold(img_gray, 80, 100, 0)
    #ret, thresh = cv2.threshold(img, 100, 255,1)
    #cv2.imshow('gray', img_gray)

    _, contours,hierarchy = cv2.findContours(thresh,2,1)
    cnt = contours
    h, w = thresh.copy().shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(thresh, mask, (0,0), 0)
    thresh = cv2.medianBlur(thresh, 5)
    thresh = cv2.GaussianBlur(thresh, (5,5), 0)
    thresh = cv2.bitwise_and(thresh, thresh, 1)
    cv2.imshow('gr', thresh)
    cv2.moveWindow('gr', 0, 0)
    
    #yes
    hull = None
    defects = None
    if (len(cnt) > 0):
        maxArea = -1
        for i in range(len(cnt)):
            temp = cnt[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i
                #print(maxArea)
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
                dist = 20
                #if (diff_s1 <= dist) or (diff_s2 <= dist) or (diff_e1 <= dist) or (diff_e2 <= dist):
                if (glob <= dist):
                    #print("Skipping")
                    continue
            #print("drawing")
            tmp_s = start
            tmp_e = end

        #Draw
            cv2.line(img,start,end,[255,0,0],2)
            cv2.line(img,start,far,[0,0,255],4)
            cv2.line(img,end,far,[0,0,255],4)
            cv2.circle(img,far,10,[0,200,0],-1)
            go = True
    except Exception as e:
        print(sys.exc_info(), sys.exc_info()[2].tb_lineno)
    cv2.imshow('img', img)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
