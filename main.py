import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import MyLib
from mpl_toolkits.mplot3d import Axes3D


calib_images = glob.glob('ChessBoard/*.jpg')
ret, mtx, dist, rvecs, tvecs = MyLib.calibrateCamera(calib_images)
print('calibrate complete.')

img1 = cv2.imread('Gate/gate_back_center.jpg')
img2 = cv2.imread('Gate/gate_back_left.jpg')
img3 = cv2.imread('Gate/gate_back_right.jpg')
img1 = img1.transpose((1, 0, 2))
img2 = img2.transpose((1, 0, 2))
img3 = img3.transpose((1, 0, 2))

h, w = img1.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

rec_img1 = MyLib.undistort(img1, newcameramtx, dist)
rec_img2 = MyLib.undistort(img2, newcameramtx, dist)
rec_img3 = MyLib.undistort(img3, newcameramtx, dist)
cv2.imwrite('undistorted_img.jpg', rec_img1)
print('undistort complete.')

x, y, w, h = roi
rec_img1 = rec_img1[y:y+h, x:x+w]
rec_img2 = rec_img2[y:y+h, x:x+w]
rec_img3 = rec_img3[y:y+h, x:x+w]

gray1 = cv2.cvtColor(rec_img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(rec_img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(rec_img3, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(rec_img1, None)
kp2, des2 = sift.detectAndCompute(rec_img2, None)
kp3, des3 = sift.detectAndCompute(rec_img3, None)
print('key points found.')

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = []
for m,n in matches:
    if m.distance < 0.5 * n.distance:
        good.append([m])

imgmat = cv2.drawMatchesKnn(rec_img1, kp1, rec_img2, kp2, good, None, flags=2)
cv2.imwrite('match.jpg', imgmat)
print('{} key points matched.'.format(len(good)))

# F, num = MyLib.eightPointRANSAC(kp1, kp2, good, 500)

points1, points2 = MyLib.gatherPoints(kp1, kp2, good)
# F_ref, _ = cv2.findFundamentalMat(points1, points2, method=cv2.RANSAC)

# u, s, v = np.linalg.svd(F)
# E = u.dot(np.diag([1, 1, 0])).dot(v)
# E = mtx.T.dot(F).dot(mtx)


points1, points2 = MyLib.gatherPoints(kp1, kp2, good)
F_ref, _ = cv2.findFundamentalMat(points1, points2, method=cv2.RANSAC)

E_ref, _ = cv2.findEssentialMat(points1, points2, mtx, method=cv2.RANSAC)
print('essential matrix get.')
print('E = ', E_ref)
R1, R2, t = cv2.decomposeEssentialMat(E_ref)
r1, r2, t1, t2 = MyLib.deriveRotate(E_ref)
print('solutions of R, t get.')
points3d = MyLib.cvTrangulate(mtx, kp1, kp2, good, r1, r2, t1, t2)
# point3d = MyLib.triangulation(mtx, kp1, kp2, good, r1, r2, t1, t2)
print('triangulation complete.')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points3d[1], points3d[2], points3d[0])
ax.view_init(elev=20, azim=-100)
plt.savefig('3dreconstruction.png')
plt.show()
print('3D reconstruction complete and saved.')


matches2 = bf.knnMatch(des1, des3, k=2)
good2 = []
for m,n in matches2:
    if m.distance < 0.5 * n.distance:
        good2.append([m])
match, kp = MyLib.findMatch(good, good2, kp3)

points3d = points3d.T
_, R_exp, t, inliers = cv2.solvePnPRansac(points3d[match].reshape(-1, 1, 3), kp.reshape(-1, 1, 2), mtx, dist)
R = cv2.Rodrigues(R_exp)[0]

print('The position and attitude of camera 3 is:\n', 'R =\n', R, '\nt =\n', t)
