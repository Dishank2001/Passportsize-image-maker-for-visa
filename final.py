import cv2
import numpy as np
from matplotlib import pyplot as plt
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img=cv2.imread('images/dishank1.jpg')
# img=cv2.imread('dishank.jpg')
# img=cv2.imread('dishank1.jpg')
# img=cv2.resize(img,(850,1300))
faces=haar_cascade.detectMultiScale(img,1.1,9)




for(x,y,w,h) in faces:
    
    print(x,y,w,h)
    # crop = img[y+5:y+h+5, x+5:x+w+5] 
    xcenter=x+w+x
    xcenter1=int(xcenter/2)
    ycenter=y+h+y
    ycenter2=int(ycenter/2)

    x1=int(xcenter1-(1*w))
    x2=int(xcenter1+(1*w))

    y1=int(ycenter2-(h))
    y2=int(ycenter2+(2*h))

    print(xcenter1,ycenter2)
    # print(xcenter1-110,ycenter2-110,xcenter1+110,ycenter2+150)

    # cv2.rectangle(img,(xcenter1-110,ycenter2-110),(xcenter1+110,ycenter2+150),(0,255,0),5)
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),5)

    img=img[y1:y2,x1:x2]




lower_white = np.array([220, 220, 220], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)
mask = cv2.inRange(img, lower_white, upper_white)  # could also use threshold
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))  # "erase" the small white points in the resulting mask
mask = cv2.bitwise_not(mask)  # invert mask

# load background (could be an image too)
bk = np.full(img.shape, (0,255,255), dtype=np.uint8)  # white bk

# get masked foreground
fg_masked = cv2.bitwise_and(img, img, mask=mask)

# get masked background, mask must be inverted 
mask = cv2.bitwise_not(mask)
bk_masked = cv2.bitwise_and(bk, bk, mask=mask)

# combine masked foreground and masked background 
final = cv2.bitwise_or(fg_masked, bk_masked)
mask = cv2.bitwise_not(mask)  # revert mask to original

cv2.imshow('res',final)
# Window waiting for commands ,0 It means infinite waiting 
cv2.waitKey(0)

