# find-blue-dots
Find blue dots in video for CLDS

Mostly experimental code for finding little blue markers in hours of video for
our friends at CLDS. **Very** application specific with a number of hacks and
constants pulled from my ear.

**VideoToBlobs.py** reads the video and outputs a pickle of the blobs it
found. Each contour includes the frames it was found on, the contour, the
pixels in the bounding box of the contour (in rgb uint8), and a label
that is 1 if we believe it to be a dot, 0 if not.

**LabelBlobs.py** reads the blobs and sets the labels.

**EditLabels.py** allows editing the labels by clicking on the images. 

**Training.ipynb** is the training for the logistic regression.

**features.py** encapsulates the code to compute features from the blobs.
The features currently include:
* circularity of the contour
* perimeter of the contour
* area of the contour
* roughness of the contour
* fraction of pixels inside the contour
* standard deviation of the pixels in the contour
* standard deviation of the pixels in the contour after one dilation
* number of frames containing the blob

These features are used to train a logistic regression classifier from
sklearn. 
