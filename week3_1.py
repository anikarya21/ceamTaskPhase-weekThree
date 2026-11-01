import numpy as np
import cv2 as cv

cap = cv.VideoCapture('cars.mp4')

# PARAMETERS FOR SHI TOMASI CORNER DETECTOR
feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# PARAMETERS FOR LUCAS KANADE OPTICAL FLOW
lk_params = dict( winSize  = (50, 50),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
black = np.zeros_like(old_gray)
rectangle = cv.rectangle(black, (0,100), (old_gray.shape[1]//2-50, old_gray.shape[0]), (255,255,255), -1)

mask = np.zeros_like(old_frame)

fps = cap.get(cv.CAP_PROP_FPS)
if fps <= 0:
    fps = 30
h, w = old_frame.shape[:2]
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('output.mp4', fourcc, fps, (w, h))

check = 0

gif_frame = []

while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if (check%50 == 0):
        p0 = cv.goodFeaturesToTrack(old_gray, mask = black, **feature_params)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), (0,0,255), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, (255,255,255), -1)
        # frame2 = cv.rectangle(frame, (0,0), (frame.shape[1]//2-50, frame.shape[0]), (0,0,0), -1)

    if ((cv.waitKey(30) & 0xff) == ord('c')):
        mask = np.zeros_like(frame)
        check = -1
        
    
    img = cv.add(frame, mask)
    # img = cv.add(img, rectangle)

    gif_frame.append(img)

    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    check = check + 1

    out.write(img)

cap.release()
out.release()
cv.destroyAllWindows()

