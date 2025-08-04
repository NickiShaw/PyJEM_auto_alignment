try:
    from PyJEM import TEM3, detector
    offline = False
except:
    from PyJEM.offline import TEM3, detector
    offline = True

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math as m
from scipy.optimize import curve_fit
from statistics import mean
from matplotlib import pyplot as plt
from scipy.special import expit, logit
from PIL import Image


class TEMControl:
    def __init__(self):
        self.stage_object = TEM3.Stage3()

    '''
    self.stage_object.SetX(val)
    self.stage_object.SetY(val)
    self.stage_object.SetZ(val)
    self.stage_object.SetTiltXAngle(val)
    self.stage_object.SetTiltYAngle(val)
    '''

def collectPhosphorImage(self, show = False):
    bufferimg = detector.Detector(0).snapshot_rawdata()
    img = np.frombuffer(bufferimg, dtype='uint16')
    # c = img.reshape(580, 580)
    if show:
        plt.imshow(img, cmap=plt.cm.gray)
        plt.show()
    return img

def confirm_tilt_state(offline=False):
    '''
    Confirms the tilt state of the TEM.
    :return: True if the TEM is in tilt state, False otherwise.
    '''
    if offline:
        return True

    # HT and emission on.
    if TEM3.HT3().GetHtValue() != 200000.0:
        return False
    # TODO look for GetEmissionOnStatus
    # Beam valve open.
    if TEM3.FEG3().GetBeamValve() == 0:
        return False
    # Beam not blanked.
    if TEM3.Def3().GetBeamBlank() == 1: # 1=ON
        return False
    # Holder is inserted.
    if TEM3.Stage3().GetHolderStts() == 0: # 0=OUT
        return False
    # CL 2 inserted, no other apertures are inserted.
    # 1=CLA, 2=OLA, 3=HCA, 4=SAA, 5=ENTA (N/A), 6=EDS
    for apt in [1,2,3,4,6]:
        TEM3.Apt3().SelectKind(apt)
        if apt == 1:
            TEM3.Apt3().SelectSize(2)
        else:
            TEM3.Apt3().SelectSize(0)
    # Focusing screen not inserted.
    # TEM mode, TEM illumination.
    if TEM3.EOS3.GetFunctionMode()[0] == 0: # TEM MAG mode
        return False
    # Mag should be 100kx.
    if TEM3.EOS3().GetMagValue() != 100000:
        return False
    # TODO set brightness value

    detector.Detector(0).livestart()
    detector.Detector(0).set_exposuretime_value(120)  # 0.12 s
    detector.Detector(0).set_gainindex(0.5)  # actually sets gain to 1

    return True



image_dir = "test.tif"

# Get the contours for an image with a set threshold.
def getContours(image, low, high):
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded_img = cv2.threshold(grey_img, low, high, cv2.THRESH_BINARY)
    contour, _ = cv2.findContours(thresholded_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contour, thresholded_img


class ContourSet:
    def __init__(self, contours, width, height):
        self.contours = contours
        self.width = width
        self.height = height
        self.area = width * height

    # Obtain index of the contour corresponding to the scalebar.
    def getScalebar(self):
        contour_widths = []
        for i in range(len(self.contours)):
            x, y, w, h = cv2.boundingRect(self.contours[i])
            # Only include contour widths with width < width of the image, and the height < 5% height of the image.
            if w < self.width and h < self.height / 20:
                contour_widths.append(w)
        # Get index of longest contour and the length of that contour.
        return contour_widths.index(max(contour_widths)), max(contour_widths)

    # Filter out small contours and parts touching sides of image.
    def filterSmallContours(self, minimum_contour):
        new_prtcle_contours = []
        pointlist = []
        for c in self.contours:
            _, _, w, h = cv2.boundingRect(c)
            rect_area = w * h
            # Keep contours that area above the specified area.
            if self.area * minimum_contour < rect_area:
                # For new contour object.
                new_prtcle_contours.append(c)
                # For getting contour pixels.
                pointlist.append(c.tolist())
        flat_pointlist = []
        for e1 in pointlist:
            for e2 in e1:
                for e3 in e2:
                    if e3[0] != 0 and e3[0] != self.width and e3[1] != 0 and e3[0] != self.height:
                        flat_pointlist.append(e3)
        return new_prtcle_contours, flat_pointlist




if __name__ == "__main__":

    image = cv2.imread(image_dir, 1)
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.imshow(grey_img, cmap='gray')
    plt.show()

    ksize = (2, 2)  # (8, 8)
    blur_image = cv2.blur(grey_img, ksize)

    height, width, channels = image.shape

    low = 0
    high = 190
    _, thresholded_img = cv2.threshold(blur_image, low, high, cv2.THRESH_BINARY)
    contour, _ = cv2.findContours(thresholded_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    plt.imshow(thresholded_img, cmap='gray')
    plt.show()

    # kernel = np.ones((10, 10), np.uint8)
    # gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    #
    # plt.imshow(gradient, cmap='gray')
    # plt.show()

    filtered_contours = []
    for c in contour:
        # remove small contours
        if cv2.contourArea(c) > 0:
            filtered_contours.append(c)

    # for i, c in enumerate(filtered_contours):
    #     # remove contours touching edges.
    #     flat_contours = np.concatenate(c)
    #     for x in [0, int(width), int(height)]:
    #         if x in flat_contours:
    #             filtered_contours.pop(i)
    #             break

    # draw contours.
    for c in contour:
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

    plt.imshow(image)
    plt.show()