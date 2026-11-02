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

# MASK FOR DRAWING PURPOSE
mask = np.zeros_like(old_frame)

# FOR GIF
fps = cap.get(cv.CAP_PROP_FPS)
if fps <= 0:
    fps = 30
h, w = old_frame.shape[:2]
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('output.mp4', fourcc, fps, (w, h))

check = 0

gif_frame = []

while(1):

    # READ THE FRAMES
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # DETECTING NEW FEATURES EVERY 50 FRAMES
    if (check%50 == 0):
        p0 = cv.goodFeaturesToTrack(old_gray, mask = black, **feature_params)

    # OPTICAL FLOW
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # GOOD POINTS
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    # TRACKING THE POINTS
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), (0,0,255), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, (255,255,255), -1)

    # PRESS C IF YOU WANT TO CLEAR THE TRACKING POINTS    
    if ((cv.waitKey(30) & 0xff) == ord('c')):
        mask = np.zeros_like(frame)
        check = -1

    # ADD THE NEW LINES TO THE FRAME FROM THE MASK
    img = cv.add(frame, mask)

    # FOR GIF
    gif_frame.append(img)


    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    
    # UPDATING THE FRAMES ( PREVIOUS AND CURRENT )
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    check = check + 1

    # FOR GIF
    out.write(img)

cap.release()
out.release()
cv.destroyAllWindows()

