import numpy as np
import cv2


def calibrateCamera(images):
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    objp = np.zeros((5 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2)
    obj_points = []  # 3D points
    img_points = []  # 2D points
    i = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (8, 5), None)
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
            # print(corners2)
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)
            cv2.drawChessboardCorners(img, (8, 5), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
            i += 1
            cv2.imwrite('Conimg/conimg{}.jpg'.format(i), img)
            cv2.waitKey(1500)
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    return ret, mtx, dist, rvecs, tvecs


def undistort(src, mtx, coef, eps=5.1e-10, max_iter=10):
    u_size, v_size = src.shape[0:2]
    dis_points = np.mgrid[0:u_size, 0:v_size].T.reshape(-1, 2).astype(np.float32)
    dis_points[:, 0] = (dis_points[:, 0] - u_size / 2) / mtx[0, 0]
    dis_points[:, 1] = (dis_points[:, 1] - v_size / 2) / mtx[1, 1]
    new_points = dis_points.copy()
    for iter in range(max_iter):
        r_2 = (new_points ** 2).sum(axis=1, keepdims=True)
        tao = 1 + coef[0, 0] * r_2 + coef[0, 1] * r_2 ** 2 + coef[0, 4] * r_2 ** 3
        r_2 = np.squeeze(r_2)
        dx = np.zeros_like(new_points)
        dx[:, 0] = 2 * coef[0, 2] * new_points[:, 0] * new_points[:, 1] + coef[0, 3] * (r_2 + 2 * new_points[:, 0] ** 2)
        dx[:, 1] = 2 * coef[0, 3] * new_points[:, 0] * new_points[:, 1] + coef[0, 2] * (r_2 + 2 * new_points[:, 1] ** 2)
        new_points = (dis_points - dx) / tao
        new_dist = tao * new_points + dx
        error = ((new_dist - dis_points) ** 2).sum()
        # print(error)
        if error < eps:
            break
    new_points[:, 0] = new_points[:, 0] * mtx[0, 0] + u_size / 2
    new_points[:, 1] = new_points[:, 1] * mtx[1, 1] + v_size / 2
    dis_points[:, 0] = dis_points[:, 0] * mtx[0, 0] + u_size / 2
    dis_points[:, 1] = dis_points[:, 1] * mtx[1, 1] + v_size / 2
    tgt = np.zeros_like(src)
    tgt[new_points[:, 0].astype(np.int), new_points[:, 1].astype(np.int)] = \
        src[dis_points[:, 0].astype(np.int), dis_points[:, 1].astype(np.int)]
    return tgt


def eightPointRANSAC(kp1, kp2, good, epoch):
    dist = 5000
    max_num = 0
    for i in range(epoch):
        index = []
        while len(index) < 8:
            x = np.random.randint(0, len(good))
            if x not in index:
                index.append(x)
        a = np.zeros((8, 9))
        for p in range(8):
            point1 = kp1[good[index[p]][0].queryIdx].pt
            point2 = kp2[good[index[p]][0].trainIdx].pt
            a[p, 0] = point1[0] * point2[0]
            a[p, 1] = point1[0] * point2[1]
            a[p, 2] = point1[0]
            a[p, 3] = point1[1] * point2[0]
            a[p, 4] = point1[1] * point2[1]
            a[p, 5] = point1[1]
            a[p, 6] = point2[0]
            a[p, 7] = point2[1]
            a[p, 8] = 1
        u, s, v = np.linalg.svd(a)
        f = v[:, 8].reshape((3, 3))
        u, s, v = np.linalg.svd(f)
        s[2] = 0
        f = u.dot(np.diag(s)).dot(v)
        num = 0
        for k in range(len(good)):
            point1 = (list(kp1[good[k][0].queryIdx].pt))
            point2 = (list(kp2[good[k][0].trainIdx].pt))
            point1.append(1)
            point2.append(1)
            d1 = np.linalg.norm(point1 - f.T.dot(point2))
            d2 = np.linalg.norm(f.dot(point1) - point2)
            d = d1 + d2
            if d < dist:
                num += 1
        if num > max_num:
            max_num = num
            final_f = f
    return final_f, max_num


def gatherPoints(kp1, kp2, good):
    points1 = []
    points2 = []
    for p in range(len(good)):
        points1.append(kp1[good[p][0].queryIdx].pt)
        points2.append(kp2[good[p][0].trainIdx].pt)
    points1 = np.array(points1)
    points2 = np.array(points2)
    return points1, points2


def deriveRotate(E):
    w = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    u, s, v = np.linalg.svd(E)
    r1 = u.dot(w).dot(v)
    r2 = u.dot(w.T).dot(v)
    tx1 = u.dot(w).dot(np.diag(s)).dot(u.T)
    tx2 = u.dot(w.T).dot(np.diag(s)).dot(u.T)
    t1 = np.array([tx1[2, 1], tx1[0, 2], tx1[1, 0]])
    t2 = np.array([tx2[2, 1], tx2[0, 2], tx2[1, 0]])
    return r1, r2, t1, t2


def v2m(v):
    m = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return m


def triangulation(mtx, kp1, kp2, good, r1, r2, t1, t2):
    t1 = np.expand_dims(t1, axis=1)
    t2 = np.expand_dims(t2, axis=1)
    P = mtx.dot(np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1))
    P1 = mtx.dot(np.concatenate((r1, t1), axis=1))
    P2 = mtx.dot(np.concatenate((r1, t2), axis=1))
    P3 = mtx.dot(np.concatenate((r2, t1), axis=1))
    P4 = mtx.dot(np.concatenate((r2, t2), axis=1))
    x1s = []
    x2s = []
    x3s = []
    x4s = []
    flag = np.zeros(4)
    for p in range(len(good)):
        point1 = kp1[good[p][0].queryIdx].pt
        point2 = kp2[good[p][0].trainIdx].pt
        point1 = np.concatenate((point1, [1]))
        point2 = np.concatenate((point2, [1]))
        a1 = np.concatenate((v2m(point1).dot(P), v2m(point2).dot(P1)), axis=0)
        b1 = a1[:, 3]
        a1 = a1[:, 0:3]
        x1 = np.linalg.inv(a1.T.dot(a1)).dot(a1.T.dot(b1))
        x1s.append(x1)
        if x1[2] < 0:
            flag[0] += 1
        a2 = np.concatenate((v2m(point1).dot(P), v2m(point2).dot(P2)), axis=0)
        b2 = a2[:, 3]
        a2 = a2[:, 0:3]
        x2 = np.linalg.inv(a2.T.dot(a2)).dot(a2.T.dot(b2))
        x2s.append(x2)
        if x2[2] < 0:
            flag[1] += 1
        a3 = np.concatenate((v2m(point1).dot(P), v2m(point2).dot(P3)), axis=0)
        b3 = a3[:, 3]
        a3 = a3[:, 0:3]
        x3 = np.linalg.inv(a3.T.dot(a1)).dot(a3.T.dot(b3))
        x3s.append(x3)
        if x3[2] < 0:
            flag[2] += 1
        a4 = np.concatenate((v2m(point1).dot(P), v2m(point2).dot(P4)), axis=0)
        b4 = a4[:, 3]
        a4 = a4[:, 0:3]
        x4 = np.linalg.inv(a4.T.dot(a4)).dot(a4.T.dot(b4))
        x4s.append(x4)
        if x4[2] < 0:
            flag[3] += 1
    choice = np.argmin(flag)
    if choice == 0:
        x = x1s
    elif choice == 1:
        x = x2s
    elif choice == 2:
        x = x3s
    else:
        x = x4s
    x = np.array(x)
    return x


def cvTrangulate(mtx, kp1, kp2, good, r1, r2, t1, t2):
    t1 = np.expand_dims(t1, axis=1)
    t2 = np.expand_dims(t2, axis=1)
    P = mtx.dot(np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1))
    P1 = mtx.dot(np.concatenate((r1, t1), axis=1))
    P2 = mtx.dot(np.concatenate((r1, t2), axis=1))
    P3 = mtx.dot(np.concatenate((r2, t1), axis=1))
    P4 = mtx.dot(np.concatenate((r2, t2), axis=1))
    points1, points2 = gatherPoints(kp1, kp2, good)
    points1 = points1.T
    points2 = points2.T
    points4d1 = cv2.triangulatePoints(P, P1, points1, points2)
    points4d2 = cv2.triangulatePoints(P, P2, points1, points2)
    points4d3 = cv2.triangulatePoints(P, P3, points1, points2)
    points4d4 = cv2.triangulatePoints(P, P4, points1, points2)
    points4d1[0:3] /= points4d1[3]
    points4d2[0:3] /= points4d2[3]
    points4d3[0:3] /= points4d3[3]
    points4d4[0:3] /= points4d4[3]
    flag = np.zeros(4)
    flag[0] = (points4d1[2] < 0).sum()
    flag[1] = (points4d2[2] < 0).sum()
    flag[2] = (points4d3[2] < 0).sum()
    flag[3] = (points4d4[2] < 0).sum()
    idx = np.argmin(flag)
    if idx == 0:
        return points4d1[0:3]
    elif idx == 1:
        return points4d2[0:3]
    elif idx == 2:
        return points4d3[0:3]
    else:
        return points4d4[0:3]




def findMatch(good1, good2, kp):
    match = []
    kp1 = []
    for p1 in range(len(good1)):
        idx = good1[p1][0].queryIdx
        for p2 in range(len(good2)):
            if good1[p2][0].queryIdx == idx:
                match.append(p1)
                kp1.append(list(kp[good2[p2][0].trainIdx].pt))
                break
    kp1 = np.array(kp1)
    return match, kp1






