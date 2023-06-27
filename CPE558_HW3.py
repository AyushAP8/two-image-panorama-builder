import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import math as mt

def plot_inlier_matches(img1, img2, inliers):
    fig, ax = plt.subplots(figsize=(20,10))
    res = np.hstack([img1, img2])
    ax.set_aspect('equal')
    ax.imshow(res, cmap='gray')

    ax.plot(inliers[:,0], inliers[:,1], '+r')
    ax.plot(inliers[:,2] + img1.shape[1], inliers[:,3], '+r')
    ax.plot([inliers[:,0], inliers[:,2] + img1.shape[1]],
            [inliers[:,1], inliers[:,3]], 'r', linewidth=0.4)
    ax.axis('off')


left = cv2.imread("uttower_left.jpg")
#left = cv2.resize(left, (500, 504))
leftg = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
right = cv2.imread("uttower_right.jpg")
#right = cv2.resize(right, (500, 504))
rightg = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
sift = cv2.ORB_create()
left_kp, left_des = sift.detectAndCompute(leftg, None)
right_kp, right_des = sift.detectAndCompute(rightg, None)
print("Desc size - ",len(right_des))
bfmatch = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
match = bfmatch.match(left_des, right_des)
print("no of matches - ",len(match))
match = sorted(match, key = lambda x:x.distance)
matchpercent = 0.5
goodmatchnum = int(len(match) * matchpercent)
match = match[:goodmatchnum]
points1 = np.zeros((len(match), 2), dtype=np.float32)
points2 = np.zeros((len(match), 2), dtype=np.float32)
for i, m in enumerate(match):
    points1[i, :] = left_kp[m.queryIdx].pt
    points2[i, :] = right_kp[m.trainIdx].pt
points1 = np.float32(points1)
points2 = np.float32(points2)

plot_inlier_matches(leftg, rightg, np.concatenate((points1, points2), axis = 1))

trForm, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
img = cv2.warpPerspective(right, trForm, (left.shape[1] + right.shape[1], left.shape[0]))
img[0:left.shape[0], 0:left.shape[1]] = left
cv2.imwrite('output.jpg', img)
img = cv2.resize(img, (int((left.shape[1] + right.shape[1])/4), int(left.shape[0]/4)))

cv2.imshow("Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



