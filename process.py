#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline
#%%
import calibration
ret, mtx, dist, rvecs, tvecs = calibration.calibrate('camera_cal/calibration*.jpg')
#%%
def undistort(bgr, dist, mtx):
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)
    return undist_img

def sobel_abs(gray, thresh=(0,255),k=3,xy=(1,0)):
    sobel = cv2.Sobel(gray, cv2.CV_64F, xy[0], xy[1], ksize=k)
    abs_sobel = np.absolute(sobel)
    sobel = np.uint8(255. * abs_sobel / np.max(abs_sobel))
    binary = np.zeros_like(sobel)
    binary[(thresh[0] <= sobel) & (sobel <= thresh[1])] = 1
    return binary

def sobel_mag(gray,thresh=(0,255),k=3):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scale = np.max(gradmag) / 255
    gradmag = (gradmag / scale).astype(np.uint8)
    binary = np.zeros_like(gradmag)
    binary[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    return binary

def sobel_dir(gray, thresh=(0, np.pi/2),k=3):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
    graddir = np.arctan2(np.absolute(sobely) ,np.absolute(sobelx) )
    binary = np.zeros_like(graddir)
    binary[(graddir >= thresh[0]) & (graddir <= thresh[1])] = 1
    return binary

def thresh(gray, thresh=(0,255)):
    binary = np.zeros_like(gray)
    binary[(gray >= thresh[0]) & (gray <= thresh[1])] = 1
    return binary

