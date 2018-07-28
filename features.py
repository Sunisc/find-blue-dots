import cv2
import numpy as np


def isBlue(im):
    mblue = np.array([60.4, -12.2, -35.7], dtype=np.float32)
    sblue = np.array([4.1, 3.2, 8.5], dtype=np.float32)
    d2 = np.sum((im - mblue)**2 / sblue**2, axis=2)
    return np.exp(-d2 / 20)


def circularity(contour):
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    area = cv2.contourArea(contour)
    result = 4 * np.pi * (area / perimeter ** 2)
    return result


def isCircular(contour, hull=False):
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    if hull:
        contour = cv2.convexHull(contour)
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * (area / perimeter ** 2)
    return 0.7 <= circularity <= 1.2
    # return 0.5 <= circularity <= 1.4


def isInside(contour, size):
    r = np.zeros(size[:2], dtype=np.uint8)
    cv2.drawContours(r, [contour - np.min(contour, axis=0)],
                     -1, 255, cv2.FILLED)
    return r == 255


def getFeatures(df):
    result = []
    for row in df.itertuples():
        lab = cv2.cvtColor(np.float32(row.pixels) / 255, cv2.COLOR_RGB2LAB)
        bim = isBlue(lab)
        h, w = bim.shape
        circ = circularity(row.contour)
        peri = cv2.arcLength(row.contour, True)
        area = cv2.contourArea(row.contour)
        roughness = cv2.arcLength(cv2.convexHull(row.contour), True) / peri
        isin = isInside(row.contour, bim.shape)
        fracin = np.sum(isin) / (h * w)
        std0 = np.std(bim[isin])
        std1 = np.std(cv2.dilate(bim, np.ones((3, 3)))[isin])
        fcount = len(row.fnos)

        result.append([circ, peri, area, roughness, fracin, std0, std1, fcount])
    return np.array(result)
