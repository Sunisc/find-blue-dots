import PIL.Image
from io import BytesIO
import ipywidgets as widgets
import cv2
import numpy as np


def array2PNGBytes(a, isLAB=True):
    '''
    Convert an array to png bytes to use with image
    '''
    if isLAB:
        a = cv2.cvtColor(a, cv2.COLOR_LAB2RGB)
    if a.dtype is not np.uint8:
        a = np.uint8(255 * a)
    h, w = a.shape[:2]
    # figure out the aspect ratio
    s = 100 / max(w, h)
    w *= s
    h *= s
    img = PIL.Image.fromarray(a, mode='RGB')
    f = BytesIO()
    img.save(f, 'png')
    return widgets.Image(
        value=f.getvalue(),
        format='png',
        width=w,
        height=h,
        layout=widgets.Layout(margin='auto')
    )


class ImageChooser(object):
    def __init__(self):
        self.checkBoxes = []
        self.images = []

    def show(self, images, labels=None, size=100, perRow=10):
        self.images = images
        rows = []
        row = []
        for i, image in enumerate(images):
            cb = widgets.Checkbox(
                value=False,
                description=labels[i] if labels else str(i),
                layout=widgets.Layout(width='{}px'.format(size + 10))
            )
            self.checkBoxes.append(cb)
            im = array2PNGBytes(image)
            vb = widgets.VBox([im, cb])
            row.append(vb)
            if len(row) == perRow:
                rows.append(widgets.HBox(row))
                row = []
        if len(row) > 0:
            rows.append(widgets.HBox(row))
        return widgets.VBox(rows)

    def extract(self):
        '''
        return two lists with chosen images in first, others in second
        '''
        chosen = []
        rejected = []
        for i, cb in enumerate(self.checkBoxes):
            if cb.value:
                chosen.append(self.images[i])
            else:
                rejected.append(self.images[i])
        return chosen, rejected
