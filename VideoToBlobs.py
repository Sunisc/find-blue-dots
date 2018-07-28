'''
Extract fixations from videos by finding the blue dots

Gary Bishop July 2018
'''

import cv2
import numpy as np
import pandas as pd
import Args

sampleVideo = (
    '/home/gb/Dropbox/Karen and Gary Shared Files/'
    'Videos & Transcripts/MSB/MSB_Video 1 (09-30-17).mp4')

args = Args.Parse(
    video=sampleVideo,
    start=100,
    end=3680,
    blobs='output.blobs.bz2'
)

vc = cv2.VideoCapture(args.video)


def grabFrame(fn):
    '''
    Extract a frame and convert it to LAB float32 format
    '''
    vc.set(cv2.CAP_PROP_POS_FRAMES, fn)
    rval, im = vc.read()
    im = cv2.cvtColor(im.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    return im


def isBlue(im):
    '''
    probability of that special blue
    '''
    mblue = np.array([60.4, -12.2, -35.7], dtype=np.float32)
    sblue = np.array([4.1, 3.2, 8.5], dtype=np.float32)
    d2 = np.sum((im - mblue)**2 / sblue**2, axis=2)
    return np.exp(-d2 / 20)


def findContours(image):
    '''
    Return the contours of regions of special blue
    '''
    # find the special blue
    blue = isBlue(image)
    _, blue = cv2.threshold(blue, 0.6, 255, cv2.THRESH_BINARY)
    blue = blue.astype(np.uint8)
    # dilate a bit to fill in noise
    blue = cv2.dilate(blue, np.ones((3, 3), dtype=np.uint8), iterations=1)
    # erode to get back
    blue = cv2.erode(blue, np.ones((3, 3), dtype=np.uint8), iterations=1)

    im2, contours, hierarchy = cv2.findContours(blue, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    # filter out tiny ones
    minArea = np.pi * 4**2
    contours = [contour for contour in contours
                if minArea < cv2.contourArea(contour)]

    return contours


def rms(v):
    return np.sqrt(np.mean(v**2))


results = []
for fno in range(args.start, args.end):
    if fno % 100 == 0:
        print(fno, len(results))
    frame = grabFrame(fno)
    for blob in findContours(frame):
        x, y, w, h = cv2.boundingRect(blob)
        pixels = frame[y:y + h, x:x + w]
        pixels = cv2.cvtColor(pixels, cv2.COLOR_LAB2RGB)
        pixels = np.uint8(255 * pixels)
        # eliminate duplicates
        for l2, f2, b2, p2 in results:
            if (b2.shape == blob.shape and np.all(b2 == blob) and
                    p2.shape == pixels.shape and np.all(p2 == pixels)):
                f2.append(fno)
                break
        else:
            results.append((0, [fno], blob, pixels))
results = pd.DataFrame(results, columns=['isdot', 'fnos', 'contour', 'pixels'])
results.to_pickle(args.blobs)
